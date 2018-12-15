import os
import tensorflow as tf
import numpy as np
import random
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--batchsize', type=int, default=-1)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--runcode', type=str, default='')
    parser.add_argument('--trainidx', type=str, default='')
    parser.add_argument('--dim', type=int, default='')
    parser.add_argument('--sdim', type=int, default='')
    parser.add_argument('--epoch', type=int, default='')
    parser.add_argument('--seed', type=int, default='')
    parser.add_argument('--trainertype', type=int, default='')
    args = parser.parse_args()

    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed + 1)
    random.seed(seed + 2)
    tf.set_random_seed(seed + 3)

    from generator import PreprocImg, Generator
    from loss import Loss
    from trainer import Trainer
    from trainer2 import Trainer2
    from mrnn_trainer import MRNNTrainer

    model = args.model
    weights = args.weights

    if weights == '':
        weights = None

    dim = args.dim
    sdim = args.sdim
    batch_size = args.batchsize
    epochs = args.epoch
    trainidx = args.trainidx
    trainertype = args.trainertype

    p1 = PreprocImg("resize_divide", [dim])
    if trainertype in [3, 4]:
        p1 = PreprocImg("resize_divide_histeq", [dim])
    p2 = PreprocImg("proc_annot_resize_one_hot", [sdim, 2])
    loss = Loss("categor_iou")

    aug_data = trainertype == 4

    if trainertype == 3:
        g1 = Generator("data/tv_split/train_" + trainidx + ".csv", "data/train/images/", batch_size, p1, p2, aug_data)
        g2 = Generator("data/tv_split/valid_" + trainidx + ".csv", "data/train/images/", batch_size, p1, p2, aug_data)

        train_steps = g1.img_cnt // batch_size
        val_steps = g2.img_cnt // batch_size
    else:
        g1 = Generator("data/train_" + trainidx + ".csv", "data/train/images/", batch_size, p1, p2, aug_data)
        g2 = Generator("data/validation.csv", "data/train/images/", batch_size, p1, p2, aug_data)
        train_steps = 10000 // batch_size
        val_steps = 1000 // batch_size

    run_code = args.runcode

    if trainertype in [1, 4]:
        train = Trainer(g1, g2, loss, model, weights, dim, epochs, train_steps, val_steps, run_code)
        train.train()
    elif trainertype == 2:
        train = MRNNTrainer("data/train/images/", "data/train_" + trainidx + ".csv",
                            "data/test/images/", "data/test.csv")
        train.train()
        train.test()
    elif trainertype == 3:
        train = Trainer2(g1, g2, loss, model, weights, dim, epochs, train_steps, val_steps, run_code)
        train.train()
    else:
        assert 0
