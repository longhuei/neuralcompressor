import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd

import itertools
from tqdm import tqdm
from time import strftime

import os
import argparse
import logging
import pickle


class EmbedQuantize(nn.Module):
    def __init__(self, pretrained, M, K, param_init=0.01, cuda=None):
        """Quantization of embeddings with composition.
        
        Arguments:
            pretrained: (array) pretrained embeddings
            M: (int) number of groups in cookbook
            K: (int) number of centroids
            param_init: (float) uniform initialization of parameters
            cuda (int): which CUDA device to place the model params
        """
        super(EmbedQuantize, self).__init__()
        self.M = M
        self.K = K
        self.cuda_device = cuda

        # self.pretrained = nn.Embedding.from_pretrained(pretrained, freeze=True)  # next version
        self.pretrained = nn.Embedding(*pretrained.shape)
        self.pretrained.weight.data = torch.from_numpy(pretrained)
        self.pretrained.weight.requires_grad = False

        embed_dim = pretrained.shape[1]
        self.linear1 = nn.Linear(embed_dim, int(self.M * self.K / 2))
        self.linear2 = nn.Linear(int(self.M * self.K / 2), self.M * self.K)
        self.codebook = nn.Linear(self.M * self.K, embed_dim, bias=False)

        # Uniform initialization
        nn.init.uniform(self.linear1.weight, a=-param_init, b=param_init)
        nn.init.uniform(self.linear1.bias, a=-param_init, b=param_init)
        nn.init.uniform(self.linear2.weight, a=-param_init, b=param_init)
        nn.init.uniform(self.linear2.bias, a=-param_init, b=param_init)
        nn.init.uniform(self.codebook.weight, a=-param_init, b=param_init)

    def forward(self, wordin, tau=1.):
        """Quantization of embeddings with composition.
        
        Arguments:
            word: (array) list of word ids in batch
            tau: (float) temperature for Gumbel softmax
        """
        tau -= 0.1

        pretrained = self.pretrained(wordin)
        hidden = F.tanh(self.linear1(pretrained))
        logits = torch.log(F.softplus(F.tanh(self.linear2(hidden))) + 1e-8)

        D = self._gumbel_softmax(
            logits.view((-1, self.M, self.K)), tau, hard=False)
        out = self.codebook(D.view(-1, self.M * self.K))
        return out, D, logits, pretrained

    def _gumbel_softmax(self, logits, temperature, eps=1e-20, hard=False):
        """Gumbel softmax """
        U = torch.rand(logits.size())
        if self.cuda_device is not None:
            U = U.cuda(self.cuda_device)
        y = logits - Variable(torch.log(-torch.log(U + eps) + eps))
        if hard:  # not important here since never used lol
            # shape = y.size()
            # _, ind = y.max(dim=-1)
            # y_hard = torch.zeros_like(y).view(-1, shape[-1])
            # y_hard.scatter_(1, ind.view(-1, 1), 1)
            # y_hard = y_hard.view(*shape)
            y_hard = y == y.max(dim=1, keepdim=True)
            y += (y_hard.type_as(y) - y).detach()
        return y


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
        quantized: (array) reconstructed quantized word embeddings
        train_loss: (list) of training loss
        train_pmax: (list) of training pmax
    """
    pretrained = np.load(args.pretrained)
    if args.n_word > 0:
        pretrained = pretrained[:args.n_word]
    batch_size = min(pretrained.shape[0], args.batch_size)
    if logger:
        logger.info("Reading pretrained from {:s}".format(args.pretrained))
        logger.debug("pretrained embedding of vocab={:d}, dim={:d}".format(
            pretrained.shape[0], pretrained.shape[1]))
        logger.debug("batch size = {:d}".format(batch_size))

    if logger:
        logger.info("Constructing EmbedQuantize model")
        logger.debug("M={:d}, K={}".format(args.M, args.K))
    model = EmbedQuantize(pretrained, args.M, args.K, cuda=args.cuda)

    if args.cuda is not None:  # Enable GPU computation
        if logger:
            logger.info("Enabling CUDA Device {:d}".format(args.cuda))
        model.cuda(args.cuda)  # place at CUDA Device with specified ID

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr)  # Adam optimizer
    if logger:
        logger.info("Creating Adam optimizer")
        logger.debug("lr={:.2e}, model parameters: {}".format(
            args.lr,
            list(
                zip(*filter(lambda p: p[1].requires_grad,
                            model.named_parameters())))[0]))

    # Create data loader
    word_id = list(range(pretrained.shape[0]))
    word_id_tensor = torch.from_numpy(np.array(word_id))
    dataloader = DataLoader(
        TensorDataset(word_id_tensor, word_id_tensor),
        batch_size=batch_size,
        shuffle=True)
    dataloader_no_shuf = DataLoader(
        TensorDataset(word_id_tensor, word_id_tensor),
        batch_size=batch_size,
        shuffle=False)

    # lr = args.lr  # adaptive learning rate
    # best_loss = np.inf
    # best_count = 0
    train_loss = []  # record loss at end of each epoch
    train_pmax = []

    for ep in range(args.epoch):
        if logger:
            logger.info(
                "============ Epoch {:3d} of {:3d} ============".format(
                    ep + 1, args.epoch))

        model.train()  # set model to train mode
        batch_loss = []
        if progbar:  # wrap with tqdm progress bar
            dataloader = tqdm(dataloader)

        for (x, y) in dataloader:
            if args.cuda is not None:  # move to GPU
                x = x.cuda(args.cuda)

            out, D, _, pretrained = model(Variable(x))  # forward pass

            loss = torch.sum((out - pretrained)**2, dim=1).mean() / 2
            pmax = D.data.cpu().numpy().max(axis=2).mean()
            batch_loss.append(loss.data.cpu().numpy()[0])

            if args.clip_norm:  # clip by gradient norm
                norm = nn.utils.clip_grad_norm(model.parameters(),
                                               args.clip_norm)
                if progbar:  # update tqdm progress bar
                    dataloader.set_postfix(
                        loss="{:02.2f}".format(batch_loss[-1]),
                        norm="{:01.2f}".format(norm))
            else:
                if progbar:
                    dataloader.set_postfix(
                        loss="{:02.2f}".format(batch_loss[-1]))

            optimizer.zero_grad()  # set gradients to zero
            loss.backward()  # backward pass
            optimizer.step()  # update model parameters according to gradients

        train_loss.append(np.mean(batch_loss))
        train_pmax.append(pmax)
        if logger:
            logger.info("Epoch {:d}: loss = {:.3f}, pmax = {:.3f}".format(
                ep + 1, train_loss[-1], train_pmax[-1]))

        if save_dir:  # save model state by frequency and at end of training
            if (ep + 1) % args.save_every == 0 or ep == args.epoch - 1:
                if logger:
                    logger.info("Evaluating...")
                model.eval()
                codebook = model.codebook.reshape((args.M, args.K, -1))
                quantised = np.zeros_like(pretrained)

                if progbar:  # wrap with tqdm progress bar
                    dataloader_no_shuf = tqdm(dataloader_no_shuf)

                for x in dataloader_no_shuf:
                    if args.cuda is not None:  # move to GPU
                        x = x.cuda(args.cuda)
                    _, _, logits, _ = model(Variable(x), tau)

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
                pickle.dump((quantised, model.train_loss, model.train_pmax),
                            open(save_path, 'wb'))

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

    return model, quantised, train_loss, train_pmax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EmbedQuantize')
    parser.add_argument('-M', type=int, default=32, help='number of subcodes')
    parser.add_argument(
        '-K', type=int, default=16, help='number of vectors in each codebook')
    parser.add_argument(
        '--tau', type=float, default=1., help='default temperature')
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
        '--lr', metavar='LR', type=float, default=1e-4, help='learning rate')
    parser.add_argument(
        '--clip-norm', type=float, default=1e-3, help='clip by total norm')
    parser.add_argument(
        '--seed',
        metavar='S',
        type=int,
        default=None,
        help='seed for random initialization')
    parser.add_argument(
        '--cuda', metavar='C', type=int, help='CUDA device to use')
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
