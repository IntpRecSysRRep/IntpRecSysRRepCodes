import os
import sys
import numpy as np
from scipy import sparse
import pandas as pd


def preprocess_data(args):
    '''
        train_df, vad_df, test_df: dataframe (df_index: uid, tid, rating)
        kg_df: dataframe (df_index: tid, aid)
    '''

    n_user, n_item = load_rating(args)
    n_attr = load_attr(args, n_item)
    print('data loaded.')
    print('Size: %d, %d, %d' % (n_user, n_item, n_attr))

    return n_user, n_item, n_attr


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    # user index and item index are not tight
    n_user = max(set(rating_np[:, 0])) + 1
    n_item = max(set(rating_np[:, 1])) + 1

    # remove zeros
    rating_np = rating_np[rating_np[:, 2] > 0]

    # split into train, valid, test data
    dataset_split(rating_np, args, n_user, n_item)

    return n_user, n_item


def dataset_split(rating_np, args, n_user, n_item, n_hold_user=200, inner_hold_ratio=0.2):
    print('splitting dataset ...')

    rating_df = pd.DataFrame(data=rating_np, columns=['uid', 'tid', 'rating'])

    # shuffle user id
    uids = np.arange(n_user)
    id_perm = np.random.permutation(n_user)
    uids = uids[id_perm]

    # split data to three parts, according to user id
    tr_users = uids[:(n_user - n_hold_user * 2)]
    vd_users = uids[(n_user - n_hold_user * 2):
                          (n_user - n_hold_user)]
    te_users = uids[(n_user - n_hold_user):]

    train_df = rating_df.loc[rating_df['uid'].isin(tr_users)]
    vad_df = rating_df.loc[rating_df['uid'].isin(vd_users)]
    test_df = rating_df.loc[rating_df['uid'].isin(te_users)]

    vad_tr, vad_te = split_train_test_proportion(vad_df, inner_hold_ratio)
    test_tr, test_te = split_train_test_proportion(test_df, inner_hold_ratio)

    pro_dir = os.path.join('../data/%s/' % args.dataset, 'pro_sg')
    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)
    train_df.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
    vad_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
    vad_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
    test_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
    test_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)


def split_train_test_proportion(df, test_prop=0.2):
    df_grouped_by_user = df.groupby('uid')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(df_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u),
                                 replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

    df_tr = pd.concat(tr_list)
    df_te = pd.concat(te_list)

    return df_tr, df_te


def load_attr(args, n_item):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    kg_np[:, 2] = kg_np[:, 2] - n_item
    kg_df = pd.DataFrame(data=kg_np, columns=['tid', 'link_type', 'aid'])
    df = kg_df[kg_df['tid'] < n_item]
    kg_df, n_attr = filter_entities(kg_df, args.appear_thre)

    aids_org = kg_df[['aid']].groupby('aid', as_index=False).size().index
    aid_org2cont = dict((aid_org, i) for (i, aid_org) in enumerate(aids_org))
    col_aid_new = list(map(lambda x: aid_org2cont[x], kg_df['aid']))
    kg_df = pd.DataFrame(data={'tid': kg_df['tid'], 'aid': col_aid_new}, columns=['tid', 'aid'])

    pro_dir = os.path.join('../data/%s/' % args.dataset, 'pro_sg')
    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)
    kg_df.to_csv(os.path.join(pro_dir, 'item_attr.csv'), index=False)

    return n_attr


def filter_entities(df, thre=5):
    playcount_groupbyid = df[['aid']].groupby('aid', as_index=False)
    count = playcount_groupbyid.size()
    df = df[df['aid'].isin(count.index[count >= thre])]
    n_attr = df[['aid']].groupby('aid', as_index=False).size().shape[0]

    return df, n_attr
