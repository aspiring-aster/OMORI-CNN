#!/usr/bin/env python3

"""
A Python program that acts as an Image Classifier
using PyTorch
"""
import argparse

from src.train import CNN, DEVICE, try_gpu


def parse_args():
    """
    Parse command line arguments for use
    """
    parser = argparse.ArgumentParser(
        prog="Image Classifier", description="A Image Classifier using PyTorch")
    parser.add_argument('action', default="train", choices=[
        "train", "test", "predict"], nargs="?")
    parser.add_argument("--score", '-s',
                        default=0, type=int)
    parser.add_argument("--epochs", '-e',
                        default=500, type=int)
    parser.add_argument("--gpu", '-g',
                        default=0, type=int)
    parser.add_argument("--model-path", '-m',
                        default="./models/model1.pth", type=str)
    parser.add_argument("--data-dir", '-d',
                        default="./data", type=str)
    args = parser.parse_args()
    return args

def main(args):

    """
    Use the parsed arguments to call methods from the CNN
    """

    network = CNN(args.data_dir)
    if args.gpu != 0:
        try_gpu(args.gpu, DEVICE)
    if args.action == "train":
        network.train(args.model_path,args.epochs)
    elif args.action == "test":
        network.test(args.model_path)
    if args.score != 0:
        network.score()


if __name__ == "__main__":
    main(parse_args())
