import warnings

import numpy as np
import torch
import torch.nn as nn

from simplified_vae.models.decoder import StateTransitionDecoder, RewardDecoder
from simplified_vae.models.encoder import RNNEncoder

from simplified_vae.config.config import Config


class VAE(nn.Module):

    """
    VAE of VariBAD:
    - has an encoder and decoder
    - can compute the ELBO loss
    - can update the VAE (encoder+decoder)
    """

    def __init__(self, config: Config,
                       obs_dim: int = None,
                       action_dim: int = None):

        # initialise the encoder
        super().__init__()
        self.config = config
        self.encoder = RNNEncoder(config=config,
                                  state_dim=obs_dim,
                                  action_dim=action_dim).to(config.device).to(self.config.device)

        # initialise the decoders (returns None for unused decoders)
        self.state_decoder = StateTransitionDecoder(config=config, action_dim=action_dim, obs_dim=obs_dim).to(self.config.device)
        self.reward_decoder = RewardDecoder(config=config, action_dim=action_dim, obs_dim=obs_dim).to(self.config.device)


    def forward(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_obs: torch.Tensor):

        latent_sample, latent_mean, latent_logvar, output = self.encoder(obs=obs, actions=actions, rewards=rewards)

        next_obs_preds = self.state_decoder(latent_sample, obs, actions)
        rewards_pred = self.reward_decoder(latent_sample, obs, actions, next_obs)

        return next_obs_preds, rewards_pred, latent_mean, latent_logvar

    def compute_loss_split_batches_by_elbo(self, latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions,
                                           vae_rewards, vae_tasks, trajectory_lens):

        """
        Loop over the elbo_t terms to compute losses per t.
        Saves some memory if batch sizes are very large,
        or if trajectory lengths are different, or if we decode only the past.
        """

        rew_reconstruction_loss = []
        state_reconstruction_loss = []
        task_reconstruction_loss = []

        assert len(np.unique(trajectory_lens)) == 1
        n_horizon = np.unique(trajectory_lens)[0]
        n_elbos = latent_mean.shape[0]  # includes the prior

        # for each elbo term (including one for the prior)...
        for idx_elbo in range(n_elbos):

            # get the embedding values (size: traj_length+1 * latent_dim; the +1 is for the prior)
            curr_means = latent_mean[idx_elbo]
            curr_logvars = latent_logvar[idx_elbo]

            # take one sample for each task
            if not self.args.disable_stochasticity_in_latent:
                curr_samples = self.encoder._sample_gaussian(curr_means, curr_logvars)
            else:
                curr_samples = torch.cat((latent_mean, latent_logvar))

            # if the size of what we decode is always the same, we can speed up creating the batches
            if not self.args.decode_only_past:

                # expand the latent to match the (x, y) pairs of the decoder
                dec_embedding = curr_samples.unsqueeze(0).expand((n_horizon, *curr_samples.shape))
                dec_embedding_task = curr_samples

                dec_prev_obs = vae_prev_obs
                dec_next_obs = vae_next_obs
                dec_actions = vae_actions
                dec_rewards = vae_rewards

            # otherwise, we unfortunately have to loop!
            # loop through the lengths we are feeding into the encoder for that trajectory (starting with prior)
            # (these are the different ELBO_t terms)
            else:

                # get the index until which we want to decode
                # (i.e. eithe runtil curr timestep or entire trajectory including future)
                if self.args.decode_only_past:
                    dec_from = 0
                    dec_until = idx_elbo
                else:
                    dec_from = 0
                    dec_until = n_horizon

                if dec_from == dec_until:
                    continue

                # (1) ... get the latent sample after feeding in some data (determined by len_encoder) & expand (to number of outputs)
                # num latent samples x embedding size
                dec_embedding = curr_samples.unsqueeze(0).expand(dec_until - dec_from, *curr_samples.shape)
                dec_embedding_task = curr_samples
                # (2) ... get the predictions for the trajectory until the timestep we're interested in
                dec_prev_obs = vae_prev_obs[dec_from:dec_until]
                dec_next_obs = vae_next_obs[dec_from:dec_until]
                dec_actions = vae_actions[dec_from:dec_until]
                dec_rewards = vae_rewards[dec_from:dec_until]

            if self.args.decode_reward:
                # compute reconstruction loss for this trajectory (for each timestep that was encoded, decode everything and sum it up)
                # size: if all trajectories are of same length [num_elbo_terms x num_reconstruction_terms], otherwise it's flattened into one
                rrc = self.compute_rew_reconstruction_loss(dec_embedding, dec_prev_obs, dec_next_obs, dec_actions,
                                                           dec_rewards)
                # sum up the reconstruction terms; average over tasks
                rrc = rrc.sum(dim=0).mean()
                rew_reconstruction_loss.append(rrc)

            if self.args.decode_state:
                src = self.compute_state_reconstruction_loss(dec_embedding, dec_prev_obs, dec_next_obs, dec_actions)
                # sum up the reconstruction terms; average over tasks
                src = src.sum(dim=0).mean()
                state_reconstruction_loss.append(src)

            if self.args.decode_task:
                trc = self.compute_task_reconstruction_loss(dec_embedding_task, vae_tasks)
                # average across tasks
                trc = trc.mean()
                task_reconstruction_loss.append(trc)

        # sum the ELBO_t terms
        if self.args.decode_reward:
            rew_reconstruction_loss = torch.stack(rew_reconstruction_loss)
            rew_reconstruction_loss = rew_reconstruction_loss.sum()
        else:
            rew_reconstruction_loss = 0

        if self.args.decode_state:
            state_reconstruction_loss = torch.stack(state_reconstruction_loss)
            state_reconstruction_loss = state_reconstruction_loss.sum()
        else:
            state_reconstruction_loss = 0

        if self.args.decode_task:
            task_reconstruction_loss = torch.stack(task_reconstruction_loss)
            task_reconstruction_loss = task_reconstruction_loss.sum()
        else:
            task_reconstruction_loss = 0

        if not self.args.disable_kl_term:
            # compute the KL term for each ELBO term of the current trajectory
            kl_loss = self.compute_kl_loss(latent_mean, latent_logvar, None)
            # sum the elbos, average across tasks
            kl_loss = kl_loss.sum(dim=0).mean()
        else:
            kl_loss = 0

        return rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss

    def compute_vae_loss(self, update=False, pretrain_index=None):
        """ Returns the VAE loss """

        # get a mini-batch
        vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, vae_tasks, \
        trajectory_lens = self.rollout_storage.sample_batch(batch_size=self.args.vae_batch_num_trajs)
        # vae_prev_obs will be of size: max trajectory len x num trajectories x dimension of observations

        # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
        _, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions,
                                                        states=vae_next_obs,
                                                        rewards=vae_rewards,
                                                        hidden_state=None,
                                                        return_prior=True,
                                                        detach_every=self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None,
                                                        )

        if self.args.split_batches_by_task:
            raise NotImplementedError
            losses = self.compute_loss_split_batches_by_task(latent_mean, latent_logvar, vae_prev_obs, vae_next_obs,
                                                             vae_actions, vae_rewards, vae_tasks,
                                                             trajectory_lens, len_encoder)
        elif self.args.split_batches_by_elbo:
            losses = self.compute_loss_split_batches_by_elbo(latent_mean, latent_logvar, vae_prev_obs, vae_next_obs,
                                                             vae_actions, vae_rewards, vae_tasks,
                                                             trajectory_lens)
        else:
            losses = self.compute_loss(latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions,
                                       vae_rewards, vae_tasks, trajectory_lens)
        rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss = losses

        # VAE loss = KL loss + reward reconstruction + state transition reconstruction
        # take average (this is the expectation over p(M))
        loss = (self.args.rew_loss_coeff * rew_reconstruction_loss +
                self.args.state_loss_coeff * state_reconstruction_loss +
                self.args.task_loss_coeff * task_reconstruction_loss +
                self.args.kl_weight * kl_loss).mean()

        # make sure we can compute gradients
        if not self.args.disable_kl_term:
            assert kl_loss.requires_grad
        if self.args.decode_reward:
            assert rew_reconstruction_loss.requires_grad
        if self.args.decode_state:
            assert state_reconstruction_loss.requires_grad
        if self.args.decode_task:
            assert task_reconstruction_loss.requires_grad

        # overall loss
        elbo_loss = loss.mean()

        if update:
            self.optimiser_vae.zero_grad()
            elbo_loss.backward()
            # clip gradients
            if self.args.encoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.encoder_max_grad_norm)
            if self.args.decoder_max_grad_norm is not None:
                if self.args.decode_reward:
                    nn.utils.clip_grad_norm_(self.reward_decoder.parameters(), self.args.decoder_max_grad_norm)
                if self.args.decode_state:
                    nn.utils.clip_grad_norm_(self.state_decoder.parameters(), self.args.decoder_max_grad_norm)
                if self.args.decode_task:
                    nn.utils.clip_grad_norm_(self.task_decoder.parameters(), self.args.decoder_max_grad_norm)
            # update
            self.optimiser_vae.step()

        self.log(elbo_loss, rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss,
                 pretrain_index)


        return elbo_loss
