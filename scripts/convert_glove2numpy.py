from __future__ import absolute_import, division, print_function
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--path")
    ap.add_argument("--dim", default=300, type=int, help="dimension")
    args = ap.parse_args()

    glove_path = args.path
    embed_path = args.path.replace("txt", "npy")
    print("convert {} to {}".format(glove_path, embed_path))

    lines = list(open(glove_path))
    embed_matrix = np.zeros((len(lines), args.dim), dtype='float32')
    for i, line in enumerate(lines):
        parts = line.strip().split()
        vec = np.array(map(float, parts[1:]), dtype='float32')
        embed_matrix[i] = vec
    np.save(embed_path, embed_matrix)
