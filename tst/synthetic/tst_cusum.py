from src.synthetic.cusum_vanilla import CUSUMVanilla


def tst_cusum():
    cusum = CUSUMVanilla()
    cusum.reset()
    vec = [2, 3, -2, -3, -1, 4, -5, 10]

    for i, v in enumerate(vec):
        print("----------------------")
        cusum.add_sample(v)
        print(f"seq={vec[:(i+1)]}")
        print(f"i={i}, v={v}")
        print(f"cusum:\n{cusum}")


if __name__ == "__main__":
    tst_cusum()
