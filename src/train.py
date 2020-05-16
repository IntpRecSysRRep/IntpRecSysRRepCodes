from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import csv
import os
import shutil
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
import tensorflow as tf
from tqdm import tqdm
from model_EBM import EBM
from utils import ndcg_binary_at_k_batch, recall_at_k_batch


def set_rng_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

def load_data_sparse(file_name, num_row, num_col):
    df = pd.read_csv(file_name)
    rows, cols = df['uid'], df['tid']
    data_mat = sparse.csr_matrix((np.ones_like(rows),
                              (rows, cols)), dtype='float64',
                             shape=(num_row, num_col))
    return data_mat

def load_kg_sparse(file_name, num_row, num_col):
    df = pd.read_csv(file_name)
    rows, cols = df['tid'], df['aid']
    data_mat = sparse.coo_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float64',
                                 shape=(num_row, num_col))
    return data_mat

def get_unique_ids(file_name, id_name):
    return pd.read_csv(file_name)[id_name].unique()

def load_data(data_name, num_user, num_item, num_attr):
    pro_dir = os.path.join('../data/%s/' % data_name, 'pro_sg')
    train_mat = load_data_sparse(os.path.join(pro_dir, 'train.csv'), num_user, num_item)
    vad_mat_tr = load_data_sparse(os.path.join(pro_dir, 'validation_tr.csv'), num_user, num_item)
    vad_mat_te = load_data_sparse(os.path.join(pro_dir, 'validation_te.csv'), num_user, num_item)
    test_mat_tr = load_data_sparse(os.path.join(pro_dir, 'test_tr.csv'), num_user, num_item)
    test_mat_te = load_data_sparse(os.path.join(pro_dir, 'test_te.csv'), num_user, num_item)
    kg_mat = load_kg_sparse(os.path.join(pro_dir, 'item_attr.csv'), num_item, num_attr)

    uid_uniq_train = get_unique_ids(os.path.join(pro_dir, 'train.csv'), 'uid')
    uid_uniq_vad = get_unique_ids(os.path.join(pro_dir, 'validation_tr.csv'), 'uid')
    uid_uniq_test = get_unique_ids(os.path.join(pro_dir, 'test_tr.csv'), 'uid')

    return train_mat, vad_mat_tr, vad_mat_te, test_mat_tr, test_mat_te, kg_mat, \
           uid_uniq_train, uid_uniq_vad, uid_uniq_test


def train(args, stats, LOG_DIR):
    set_rng_seed(args.seed)
    num_user, num_item, num_attr = stats[0], stats[1], stats[2]
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    # load data
    train_mat, vad_mat_tr, vad_mat_te, _, _, kg_mat, \
    uid_uniq_train, uid_uniq_vad, _ = load_data(args.dataset, num_user, num_item, num_attr)

    # training preparation
    len_train = uid_uniq_train.size
    num_batches = int(np.ceil(float(len_train)/args.batch_size))
    total_anneal_steps = 5 * num_batches
    len_vad = uid_uniq_vad.size

    # build model
    tf.reset_default_graph()
    ebm = EBM(num_item, num_attr, args.K, args.dim, kg_mat,
              lam=args.lam, coef_attr=args.beta)
    saver, logits, train_op = ebm.build_graph()

    # ndcg record
    ndcg_var = tf.Variable(0.0)
    ndcg_best_var = tf.placeholder(dtype=tf.float64, shape=None)
    ndcg_summary = tf.summary.scalar('vad/ndcg', ndcg_var)
    ndcg_best_summary = tf.summary.scalar('vad/ndcg_best', ndcg_best_var)
    merged_valid = tf.summary.merge([ndcg_summary, ndcg_best_summary])

    # training
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        best_ndcg = -np.inf
        update_count = 0.0

        # each epoch
        for epoch in range(args.n_epochs):
            print("Epoch: %d" % epoch)
            np.random.shuffle(uid_uniq_train)

            # each batch
            for bnum, st_idx in tqdm(enumerate(range(0, len_train, args.batch_size))):
                # read rows from source mat
                end_idx = min(st_idx + args.batch_size, len_train)
                x = train_mat[uid_uniq_train[st_idx:end_idx]]
                if sparse.isspmatrix(x):
                    x = x.toarray()
                x = x.astype('float64')

                if total_anneal_steps > 0:
                    anneal = min(args.beta,
                                 1. * update_count / total_anneal_steps)
                else:
                    anneal = args.beta
                update_count += 1

                feed_dict = {ebm.input_ph: x,
                             ebm.is_training_ph: 1,
                             ebm.beta_ph: anneal,
                             ebm.dropout_p_ph: args.dropout
                             }
                sess.run(train_op, feed_dict=feed_dict)

            # validation after each epoch
            ndcg_dist = []
            for bnum, st_idx in enumerate(range(0, len_vad, args.batch_size)):
                end_idx = min(st_idx + args.batch_size, len_vad)
                x = vad_mat_tr[uid_uniq_vad[st_idx:end_idx]]
                if sparse.isspmatrix(x):
                    x = x.toarray()
                x = x.astype('float64')

                pred = sess.run(logits, feed_dict={ebm.input_ph: x})
                pred[x.nonzero()] = -np.inf     # # exclude examples from training and validation (if any)
                ndcg_dist.append(ndcg_binary_at_k_batch(
                        pred, vad_mat_te[uid_uniq_vad[st_idx:end_idx]])
                )
            ndcg_dist = np.concatenate(ndcg_dist)
            ndcg = ndcg_dist.mean()
            print('ndcg: %g' % ndcg)
            if ndcg > best_ndcg:
                saver.save(sess, '{}/chkpt'.format(LOG_DIR))
                best_ndcg = ndcg

        return best_ndcg


def test(args, stats, LOG_DIR):
    set_rng_seed(args.seed)
    num_user, num_item, num_attr = stats[0], stats[1], stats[2]

    # load data
    _, _, _, test_mat_tr, test_mat_te, kg_mat, \
        _, _, uid_uniq_test = load_data(args.dataset, num_user, num_item, num_attr)
    len_test = uid_uniq_test.size
    print('There are %d users to test.' % len_test)

    # build model
    tf.reset_default_graph()
    ebm = EBM(num_item, num_attr, args.K, args.dim, kg_mat,
              lam=args.lam, coef_attr=args.beta)
    saver, logits, _ = ebm.build_graph()

    # evaluation
    n100_list, r2_list, r5_list, r10_list, r50_list, r100_list = [], [], [], [], [], []
    with tf.Session() as sess:
        print('Load from: ', LOG_DIR)
        saver.restore(sess, '{}/chkpt'.format(LOG_DIR))

        for bnum, st_idx in enumerate(range(0, len_test, args.batch_size)):
            # read rows from source mat
            end_idx = min(st_idx + args.batch_size, len_test)
            x = test_mat_tr[uid_uniq_test[st_idx:end_idx]]
            if sparse.isspmatrix(x):
                x = x.toarray()
            x = x.astype('float32')

            # inference
            pred = sess.run(logits, feed_dict={ebm.input_ph: x})
            pred[x.nonzero()] = -np.inf

            # evaluate
            n100_list.append(ndcg_binary_at_k_batch(
                pred, test_mat_te[uid_uniq_test[st_idx:end_idx]], k=100))
            r2_list.append(recall_at_k_batch(
                pred, test_mat_te[uid_uniq_test[st_idx:end_idx]], k=2))
            r5_list.append(recall_at_k_batch(
                pred, test_mat_te[uid_uniq_test[st_idx:end_idx]], k=5))
            r10_list.append(recall_at_k_batch(
                pred, test_mat_te[uid_uniq_test[st_idx:end_idx]], k=10))
            r50_list.append(recall_at_k_batch(
                pred, test_mat_te[uid_uniq_test[st_idx:end_idx]], k=50))
            r100_list.append(recall_at_k_batch(
                pred, test_mat_te[uid_uniq_test[st_idx:end_idx]], k=100))

    n100_list = np.concatenate(n100_list)
    r2_list = np.concatenate(r2_list)
    r5_list = np.concatenate(r5_list)
    r10_list = np.concatenate(r10_list)
    r50_list = np.concatenate(r50_list)
    r100_list = np.concatenate(r100_list)

    # mean and variance in test
    print("Test NDCG@100=%.5f (%.5f)" % (
        n100_list.mean(), np.std(n100_list) / np.sqrt(len(n100_list))),
          file=sys.stderr)
    print("Test Recall@2=%.5f (%.5f)" % (
        r2_list.mean(), np.std(r2_list) / np.sqrt(len(r2_list))),
          file=sys.stderr)
    print("Test Recall@5=%.5f (%.5f)" % (
        r5_list.mean(), np.std(r5_list) / np.sqrt(len(r5_list))),
          file=sys.stderr)
    print("Test Recall@10=%.5f (%.5f)" % (
        r10_list.mean(), np.std(r10_list) / np.sqrt(len(r10_list))),
          file=sys.stderr)
    print("Test Recall@50=%.5f (%.5f)" % (
        r50_list.mean(), np.std(r50_list) / np.sqrt(len(r50_list))),
          file=sys.stderr)
    print("Test Recall@100=%.5f (%.5f)" % (
        r100_list.mean(), np.std(r100_list) / np.sqrt(len(r100_list))),
          file=sys.stderr)