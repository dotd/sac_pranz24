import torch


def compute_state_reconstruction_loss(mext_obs_pred: torch.Tensor, next_obs: torch.Tensor):
    """ Compute state reconstruction loss.
    (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """

    state_loss = (mext_obs_pred - next_obs).pow(2).mean(dim=-1)
    return state_loss


def compute_reward_reconstruction_loss(rewards_pred: torch.Tensor, rewards: torch.Tensor):
    """ Compute reward reconstruction loss.
    (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """

    reward_loss = (rewards_pred - rewards).pow(2).mean(dim=-1)
    return reward_loss

def compute_kl_loss(latent_mean, latent_logvar):

    # -- KL divergence
    gauss_dim = latent_mean.shape[-1]

    # https://arxiv.org/pdf/1811.09975.pdf
    # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))

    mu = latent_mean[:, 1:, :] # curr_posterior_mean
    m = latent_mean[:, :-1, :] # prev_posterior_mean

    logE = latent_logvar[:, 1:, :] # curr_posterior_logvar
    logS = latent_logvar[:,:-1, :] # prev_posterior_logvar

    kl_divergences = 0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - gauss_dim +
                            torch.sum(torch.exp(logE) / torch.exp(logS), dim=-1) +
                            ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=-1))

    return kl_divergences


def compute_loss(self, latent_mean,
                 latent_logvar,
                 vae_prev_obs,
                 vae_next_obs,
                 vae_actions,
                 vae_rewards,
                 vae_tasks,
                 trajectory_lens):
    """
    Computes the VAE loss for the given data.
    Batches everything together and therefore needs all trajectories to be of the same length.
    (Important because we need to separate ELBOs and decoding terms so can't collapse those dimensions)
    """

    num_unique_trajectory_lens = len(np.unique(trajectory_lens))

    assert (num_unique_trajectory_lens == 1) or (self.args.vae_subsample_elbos and self.args.vae_subsample_decodes)
    assert not self.args.decode_only_past

    # cut down the batch to the longest trajectory length
    # this way we can preserve the structure
    # but we will waste some computation on zero-padded trajectories that are shorter than max_traj_len
    max_traj_len = np.max(trajectory_lens)
    latent_mean = latent_mean[:max_traj_len + 1]
    latent_logvar = latent_logvar[:max_traj_len + 1]
    vae_prev_obs = vae_prev_obs[:max_traj_len]
    vae_next_obs = vae_next_obs[:max_traj_len]
    vae_actions = vae_actions[:max_traj_len]
    vae_rewards = vae_rewards[:max_traj_len]

    # take one sample for each ELBO term
    if not self.args.disable_stochasticity_in_latent:
        latent_samples = self.encoder._sample_gaussian(latent_mean, latent_logvar)
    else:
        latent_samples = torch.cat((latent_mean, latent_logvar), dim=-1)

    num_elbos = latent_samples.shape[0]
    num_decodes = vae_prev_obs.shape[0]
    batchsize = latent_samples.shape[1]  # number of trajectories

    # subsample elbo terms
    #   shape before: num_elbos * batchsize * dim
    #   shape after: vae_subsample_elbos * batchsize * dim
    if self.args.vae_subsample_elbos is not None:
        # randomly choose which elbo's to subsample
        if num_unique_trajectory_lens == 1:
            elbo_indices = torch.LongTensor(self.args.vae_subsample_elbos * batchsize).random_(0,
                                                                                               num_elbos)  # select diff elbos for each task
        else:
            # if we have different trajectory lengths, subsample elbo indices separately
            # up to their maximum possible encoding length;
            # only allow duplicates if the sample size would be larger than the number of samples
            elbo_indices = np.concatenate([np.random.choice(range(0, t + 1), self.args.vae_subsample_elbos,
                                                            replace=self.args.vae_subsample_elbos > (t + 1)) for t in
                                           trajectory_lens])
            if max_traj_len < self.args.vae_subsample_elbos:
                warnings.warn('The required number of ELBOs is larger than the shortest trajectory, '
                              'so there will be duplicates in your batch.'
                              'To avoid this use --split_batches_by_elbo or --split_batches_by_task.')
        task_indices = torch.arange(batchsize).repeat(self.args.vae_subsample_elbos)  # for selection mask
        latent_samples = latent_samples[elbo_indices, task_indices, :].reshape(
            (self.args.vae_subsample_elbos, batchsize, -1))
        num_elbos = latent_samples.shape[0]
    else:
        elbo_indices = None

    # expand the state/rew/action inputs to the decoder (to match size of latents)
    # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
    dec_prev_obs = vae_prev_obs.unsqueeze(0).expand((num_elbos, *vae_prev_obs.shape))
    dec_next_obs = vae_next_obs.unsqueeze(0).expand((num_elbos, *vae_next_obs.shape))
    dec_actions = vae_actions.unsqueeze(0).expand((num_elbos, *vae_actions.shape))
    dec_rewards = vae_rewards.unsqueeze(0).expand((num_elbos, *vae_rewards.shape))

    # subsample reconstruction terms
    if self.args.vae_subsample_decodes is not None:
        # shape before: vae_subsample_elbos * num_decodes * batchsize * dim
        # shape after: vae_subsample_elbos * vae_subsample_decodes * batchsize * dim
        # (Note that this will always have duplicates given how we set up the code)
        indices0 = torch.arange(num_elbos).repeat(self.args.vae_subsample_decodes * batchsize)
        if num_unique_trajectory_lens == 1:
            indices1 = torch.LongTensor(num_elbos * self.args.vae_subsample_decodes * batchsize).random_(0, num_decodes)
        else:
            indices1 = np.concatenate([np.random.choice(range(0, t), num_elbos * self.args.vae_subsample_decodes,
                                                        replace=True) for t in trajectory_lens])
        indices2 = torch.arange(batchsize).repeat(num_elbos * self.args.vae_subsample_decodes)
        dec_prev_obs = dec_prev_obs[indices0, indices1, indices2, :].reshape(
            (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
        dec_next_obs = dec_next_obs[indices0, indices1, indices2, :].reshape(
            (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
        dec_actions = dec_actions[indices0, indices1, indices2, :].reshape(
            (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
        dec_rewards = dec_rewards[indices0, indices1, indices2, :].reshape(
            (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
        num_decodes = dec_prev_obs.shape[1]

    # expand the latent (to match the number of state/rew/action inputs to the decoder)
    # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
    dec_embedding = latent_samples.unsqueeze(0).expand((num_decodes, *latent_samples.shape)).transpose(1, 0)

    if self.args.decode_reward:
        # compute reconstruction loss for this trajectory (for each timestep that was encoded, decode everything and sum it up)
        # shape: [num_elbo_terms] x [num_reconstruction_terms] x [num_trajectories]
        rew_reconstruction_loss = self.compute_rew_reconstruction_loss(dec_embedding, dec_prev_obs, dec_next_obs,
                                                                       dec_actions, dec_rewards)
        # avg/sum across individual ELBO terms
        if self.args.vae_avg_elbo_terms:
            rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
        else:
            rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
        # avg/sum across individual reconstruction terms
        if self.args.vae_avg_reconstruction_terms:
            rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
        else:
            rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
        # average across tasks
        rew_reconstruction_loss = rew_reconstruction_loss.mean()
    else:
        rew_reconstruction_loss = 0

    if self.args.decode_state:
        state_reconstruction_loss = self.compute_state_reconstruction_loss(dec_embedding, dec_prev_obs,
                                                                           dec_next_obs, dec_actions)
        # avg/sum across individual ELBO terms
        if self.args.vae_avg_elbo_terms:
            state_reconstruction_loss = state_reconstruction_loss.mean(dim=0)
        else:
            state_reconstruction_loss = state_reconstruction_loss.sum(dim=0)
        # avg/sum across individual reconstruction terms
        if self.args.vae_avg_reconstruction_terms:
            state_reconstruction_loss = state_reconstruction_loss.mean(dim=0)
        else:
            state_reconstruction_loss = state_reconstruction_loss.sum(dim=0)
        # average across tasks
        state_reconstruction_loss = state_reconstruction_loss.mean()
    else:
        state_reconstruction_loss = 0

    if self.args.decode_task:
        task_reconstruction_loss = self.compute_task_reconstruction_loss(latent_samples, vae_tasks)
        # avg/sum across individual ELBO terms
        if self.args.vae_avg_elbo_terms:
            task_reconstruction_loss = task_reconstruction_loss.mean(dim=0)
        else:
            task_reconstruction_loss = task_reconstruction_loss.sum(dim=0)
        # sum the elbos, average across tasks
        task_reconstruction_loss = task_reconstruction_loss.sum(dim=0).mean()
    else:
        task_reconstruction_loss = 0

    if not self.args.disable_kl_term:
        # compute the KL term for each ELBO term of the current trajectory
        # shape: [num_elbo_terms] x [num_trajectories]
        kl_loss = self.compute_kl_loss(latent_mean, latent_logvar, elbo_indices)
        # avg/sum the elbos
        if self.args.vae_avg_elbo_terms:
            kl_loss = kl_loss.mean(dim=0)
        else:
            kl_loss = kl_loss.sum(dim=0)
        # average across tasks
        kl_loss = kl_loss.sum(dim=0).mean()
    else:
        kl_loss = 0

    return rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss