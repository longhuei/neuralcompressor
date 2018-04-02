import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import pandas as pd

import itertools
from tqdm import tqdm
from time import strftime

import os
import argparse
import logging
import pickle

parser = argparse.ArgumentParser(description='EmbedQuantize')
parser.add_argument('-M', type=int, default=20, help='number of subcodes')
parser.add_argument(
    '-K', type=int, default=20, help='number of vectors in each codebook')
parser.add_argument(
    '--epoch',
    metavar='EP',
    type=int,
    default=300,
    help='number of training epochs')
parser.add_argument(
    '--batch-size',
    metavar='BS',
    type=int,
    default=32,
    help='size of each mini-batch')
parser.add_argument(
    '--n-word',
    metavar='NW',
    type=int,
    default=50000,
    help='number of words to compress, 0 for all')
parser.add_argument(
    '--lr', metavar='LR', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--seed',
    metavar='S',
    type=int,
    default=None,
    help='seed for random initialization')
parser.add_argument('--cuda', metavar='C', type=int, help='CUDA device to use')
parser.add_argument(
    '--pretrained',
    metavar='P',
    type=str,
    default="data/glove.6B.300d.npy",
    help='pretrined word embeddings numpy array')
parser.add_argument(
    '--save-dir',
    metavar='SD',
    type=str,
    default='model',
    help='directory in which model states are to be saved')
parser.add_argument(
    '--save-every',
    metavar='SE',
    type=int,
    default=50,
    help='epoch frequncy of saving model state to directory')
args = parser.parse_args()


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:  # not important here since never used lol
        # shape = y.size()
        # _, ind = y.max(dim=-1)
        # y_hard = torch.zeros_like(y).view(-1, shape[-1])
        # y_hard.scatter_(1, ind.view(-1, 1), 1)
        # y_hard = y_hard.view(*shape)
        y_hard = y == y.max(dim=1, keepdim=True)
        y += (y_hard.type_as(y) - y).detach()

    return y


class EmbedQuantize(nn.Module):
    def __init__(self, embedding_npy, M, K, param_init=0.01):
        super(EmbedQuantize, self).__init__()
        vocab_size = embedding_npy.shape[0]
        emb_size = embedding_npy.shape[1]
        self.M = M
        self.num_centroids = 2**K

        # self.pretrained = nn.Embedding.from_pretrained(embedding_npy, freeze=True)  # next version
        self.pretrained = nn.Embedding(vocab_size, emb_size)
        self.pretrained.weight.data = torch.from_numpy(embedding_npy)
        self.pretrained.weight.requires_grad = False
        self.codebook = nn.Embedding(M * self.num_centroids, emb_size)
        self.linear1 = nn.Linear(emb_size, M * self.num_centroids / 2)
        self.linear2 = nn.Linear(M * self.num_centroids / 2,
                                 M * self.num_centroids)

        nn.init.uniform_(self.embed.weight, a=-param_init, b=param_init)
        nn.init.uniform_(self.codebook.weight, a=-param_init, b=param_init)
        nn.init.uniform_(self.linear1.weight, a=-param_init, b=param_init)
        nn.init.uniform_(self.linear1.bias, a=-param_init, b=param_init)
        nn.init.uniform_(self.linear2.weight, a=-param_init, b=param_init)
        nn.init.uniform_(self.linear1.bias, a=-param_init, b=param_init)

    def forward(self, wordin=np.array([3, 4, 5]), tau=1.):
        tau -= 0.1

        h = F.tanh(self.linear1(self.pretrained(wordin)))
        logits = F.log(F.softplus(F.tanh(self.linear2(h))) + 1e-8)
        logits = logits.reshape((-1, self.M, self.num_centroids))

        D = gumbel_softmax(logits_lookup, tau, hard=False)
        y = D.reshape(-1, M * num_centroids).matmul(A)
        return y, word_lookup, logits


def save_quantized():
    for x in loader:
        if args.cuda is not None:  # move to GPU
            x = x.cuda(args.cuda)
            # y = y.cuda(args.cuda)

        y, word_lookup = model(Variable(x))

        loss = torch.sum((y - word_lookup)**2, dim=1).mean() / 2
        optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
        optimizer.zero_grad()  # set Variables' gradient to zero
        loss.backward()  # backward pass calculates gradients
        batch_loss.append(loss.data.cpu().numpy()[0])
        if progbar:  # update tqdm progress bar
            loader.set_postfix(loss="{:2.2f}".format(batch_loss[-1]))

        optimizer.step()  # update model parameters according to gradients


def train(args, save_dir=None, logger=None, progbar=True):
    """ Train a EmbedQuantize model and save model states.

    Given args specifying training hyperparameters, train a NativeLanguageCNN
    model and save model states to specified directory

    Arguments:
        args: (ArgumentParser) argument parser containing training parameters
        save_dir: (str) file directory in which model training logs and model
            states are saved
        logger: (Logger) logger object to which logging descriptions are written
        progbar: (bool) whether to show a tqdm progress bar

    Returns:
        model: (EmbedQuantize) final model after training
        train_loss: (float) final training loss
        train_acc: (float) final train set accuracy
        val_acc: (float) final validation set accuracy
    """
    pretrained = np.load(args.pretrained)
    if args.n_word > 0:
        pretrained = pretrained[:args.n_word]
    batch_size = min(pretrained.shape[0], args.batch_size)
    if logger:
        logger.info("Reading pretrained from {:s}".format(args.pretrained))
        logger.debug("pretrained data of size {}".format(pretrained.shape))
        logger.debug("batch size = {:d}".format(batch_size))

    if logger:
        logger.info("Constructing EmbedQuantize model")
        logger.debug("M={:d}, K={}".format(args.M, args.K))
    model = EmbedQuantize(pretrained, args.M, args.K)

    if args.cuda is not None:  # Enable GPU computation
        if logger:
            logger.info("Enabling CUDA Device {:d}".format(args.cuda))
        model.cuda(args.cuda)  # place at CUDA Device with specified ID

    if logger:
        logger.info("Creating optimizer")
        logger.debug("model parameters: {}".format(
            list(zip(*model.named_parameters()))[0]))

    if logger:
        logger.debug("Adam optimizer w/ lr={:.2e}".format(args.lr))

    # Create data loader
    word_ids = list(range(pretrained.shape[0]))
    dataloader = DataLoader(
        TensorDataset(torch.from_numpy(word_ids)),
        batch_size=batch_size,
        shuffle=True)
    dataloader_no_shuf = DataLoader(
        TensorDataset(torch.from_numpy(word_ids)),
        batch_size=batch_size,
        shuffle=False)

    lr = args.lr  # adaptive learning rate
    # best_loss = np.inf
    # best_count = 0
    train_loss = []  # record loss at end of each epoch

    for ep in range(args.epoch):
        if logger:
            logger.info(
                "============ Epoch {:2d} of {:2d} ============".format(
                    ep + 1, args.epoch))

        # wrap tqdm progress bar around dataloader if necessary
        loader = tqdm(dataloader) if progbar else dataloader

        model.train()  # set model to train mode
        batch_loss = []
        for x in loader:
            if args.cuda is not None:  # move to GPU
                x = x.cuda(args.cuda)
                # y = y.cuda(args.cuda)

            y, word_lookup, _ = model(Variable(x))  # forward pass

            loss = torch.sum((y - word_lookup)**2, dim=1).mean() / 2
            optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
            optimizer.zero_grad()  # set gradients to zero
            loss.backward()  # backward pass
            batch_loss.append(loss.data.cpu().numpy()[0])
            if progbar:  # update tqdm progress bar
                loader.set_postfix(loss="{:2.2f}".format(batch_loss[-1]))

            optimizer.step()  # update model parameters according to gradients

        train_loss.append(np.mean(batch_loss))

        if logger:
            logger.info("Epoch {:d}: loss = {:.3f}".format(
                ep + 1, train_loss[-1]))

        if save_dir:  # save model state by frequency and at end of training
            if (ep + 1) % args.save_every == 0 or ep == args.epoch - 1:
                if logger:
                    logger.info("Evaluating...")
                model.eval()
                codebook = model.codebook.reshape((args.M, 2**args.K, -1))
                quantised = np.zeros_like(pretrained)

                for x in dataloader_no_shuf:
                    if args.cuda is not None:  # move to GPU
                        x = x.cuda(args.cuda)
                    _, _, logits = model(Variable(x))

                    for (i, w) in enumerate(x):
                        quantised[w] = sum(
                            codebook[m, np.argmax(logits[i, m])]
                            for m in range(args.M))

                if logger:
                    logger.info("Save model-state-{:04d}.pkl".format(ep + 1))
                save_path = os.path.join(
                    save_dir, "model-state-{:04d}.pkl".format(ep + 1))
                torch.save(model.state_dict(), save_path)
                save_path = os.path.join(save_dir, "model-quan-loss.pkl")
                pickle.dump((quantised, model.train_loss), open(
                    save_path, 'wb'))

        # if train_loss[-1] < best_loss * 0.99:
        #     best_loss = train_loss[-1]
        #     best_count = 0
        # else:
        #     best_count += 1
        #     if best_counter >= 100:
        #         best_counter = 0
        #         lr /= 2
        #         if lr < 1.e-5:
        #             print('learning rate too small - stopping now')
        #             break

    return model, quantised, train_loss


if __name__ == '__main__':
    # Create log directory + file
    timestamp = strftime("%Y-%m-%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir)

    # Setup logger
    logging.basicConfig(
        filename=os.path.join(args.save_dir, timestamp + ".log"),
        format=
        '[%(asctime)s] {%(pathname)s:%(lineno)3d} %(levelname)6s - %(message)s',
        level=logging.DEBUG,
        datefmt='%H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger("TRAIN")
    logger.info("Timestamp: {}".format(timestamp))

    # Set random seed
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    train(args, save_dir, logger)
