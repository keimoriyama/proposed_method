import argparse

from omegaconf import OmegaConf

from train.train import eval, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", help="hyperparams for exp")
    parser.add_argument("--model", help="choose model to trian and evaluate")
    parser.add_argument("--mode", help="choose train or evaluate")
    parser.add_argument("--exp_name", help="expriment name")

    config = OmegaConf.load("./config/config.yml")
    args = parser.parse_args()
    if args.alpha is not None:
        config.train.alpha = float(args.alpha)
    if args.model is not None:
        config.model = args.model
    if args.mode is not None:
        config["mode"] = args.mode
    if config.mode == "train":
        train(config=config)
    elif config.mode == "eval":
        eval(config=config)


if __name__ == "__main__":
    main()
