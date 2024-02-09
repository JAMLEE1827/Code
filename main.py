from traj import DiffuTraj
import argparse
import os
import yaml
# from pprint import pprint
from easydict import EasyDict
import numpy as np
import pdb

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of DiffuTraj')
    parser.add_argument('--config', default='configs/baseline.yaml')
    parser.add_argument('--dataset', default='vessel')
    return parser.parse_args()


def main():
    # parse arguments and load config
    args = parse_args()
    with open(args.config) as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    config["dataset"] = args.dataset[:-1]
    config = EasyDict(config)
    agent = DiffuTraj(config)

    sampling = "ddim"
    steps = 5

    if config["eval_mode"]:
        agent.eval(sampling, 100//steps)
    else:
        agent.train()

if __name__ == '__main__':
    main()
