import argparse
import os
import datetime
import numpy as np
import GPUtil
from time import time
from data_loader import preprocess_data
from train import train, test

np.random.seed(555)

def select_gpu():
    try:
        # Get the first available GPU
        DEVICE_ID_LIST = GPUtil.getFirstAvailable()
        DEVICE_ID = DEVICE_ID_LIST[0]

        # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    except FileNotFoundError:
        print("GPU not found")

parser = argparse.ArgumentParser()


# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--model', type=str, default='ebm', help='which model to use')
parser.add_argument('--n_epochs', type=int, default=100, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_neg', type=int, default=128, help='number of negative samples for evaluation')
parser.add_argument('--K', type=int, default=4, help='number of vectors')
parser.add_argument('--appear_thre', type=int, default=5, help='threshold of knowledge entities')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--lam', type=int, default=1e-8, help='l2 regularization')
parser.add_argument('--beta', type=int, default=0.2, help='entity weight')
parser.add_argument('--dropout', type=int, default=0.5, help='dropout rate')


'''
# yelp
parser.add_argument('--dataset', type=str, default='yelp', help='which dataset to use')
parser.add_argument('--model', type=str, default='ebm', help='which model to use')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--dim', type=int, default=25, help='dimension of user and entity embeddings')
parser.add_argument('--n_neg', type=int, default=128, help='number of negative samples for evaluation')
parser.add_argument('--K', type=int, default=6, help='number of vectors')
parser.add_argument('--appear_thre', type=int, default=5, help='threshold of knowledge entities')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--lam', type=int, default=5e-9, help='l2 regularization')
parser.add_argument('--beta', type=int, default=0.2, help='control prior regularization')
parser.add_argument('--dropout', type=int, default=0.5, help='dropout rate')
'''

# The movielens data is too large to be uploaded to github, so this part can not be run at this moment
'''
# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--model', type=str, default='ebm', help='which model to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=4e-4, help='learning rate')
parser.add_argument('--dim', type=int, default=30, help='dimension of user and entity embeddings')
parser.add_argument('--n_neg', type=int, default=128, help='number of negative samples for evaluation')
parser.add_argument('--K', type=int, default=4, help='number of vectors')
parser.add_argument('--appear_thre', type=int, default=10, help='threshold of knowledge entities')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--lam', type=int, default=1e-8, help='l2 regularization')
parser.add_argument('--beta', type=int, default=0.2, help='entity weight')
parser.add_argument('--dropout', type=int, default=0.5, help='dropout rate')
'''

select_gpu()

show_loss = False
show_time = True
show_topk = True

t = time()


args = parser.parse_args()
LOG_DIR = '%s-%dE-%dB-%glr-%dD-%dK-%gb-%gs' % (
    args.dataset, args.n_epochs, args.batch_size, args.lr, args.dim, args.K, args.beta, args.seed)
LOG_DIR = os.path.join('../runs/', LOG_DIR)
print(LOG_DIR)

stats = preprocess_data(args)

# Comment the line below to directly test on trained models
#train(args, stats, LOG_DIR)

test(args, stats, LOG_DIR)

if show_time:
    print('time used: %d s' % (time() - t))
