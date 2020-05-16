import tensorflow as tf
import numpy as np

from tensorflow.contrib.distributions import RelaxedOneHotCategorical
from tensorflow.contrib.layers import apply_regularization, l2_regularizer


class EBM:
    def __init__(self, num_items, num_attrs, K, dim, kg_mat,
                 lam=0.0, tau=0.1, gumbel=True, std=0.75, lr=1e-3, coef_attr=0.2,
                 seed=1):
        self.num_items = num_items
        self.num_attrs = num_attrs
        self.K = K
        self.dim = dim
        self.lam = lam
        self.tau = tau
        self.gumbel = gumbel
        self.lr = lr
        self.coef_attr = coef_attr
        self.seed = seed

        self.std = std      # Gaussian prior std

        self.qItem_dims_in = [self.num_items, self.dim]
        self.qItem_dims_out = [self.dim, self.dim]
        self.qAttr_dims_in = [self.num_attrs, self.dim]
        self.qAttr_dims_out = [self.dim, self.dim]

        # sparse item-attr matrix
        indices = np.stack((kg_mat.row, kg_mat.col), axis=0).T
        self.kg_mat = tf.SparseTensor(indices=indices, values=kg_mat.data, dense_shape=kg_mat.shape)

        # w_qItem[0] stores embedding of items
        self.w_qItem, self.b_qItem = [], []
        for i, (dim_in, dim_out) in enumerate(zip(self.qItem_dims_in, self.qItem_dims_out)):
            if i == len(self.qItem_dims_in) - 1:
                dim_out *= 2
            weight_key = "weight_qItem_{}to{}".format(i, i + 1)
            self.w_qItem.append(tf.get_variable(
                name=weight_key, shape=[dim_in, dim_out], dtype=tf.float64,
                initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
            )
            bias_key = "bias_qItem_{}".format(i + 1)
            self.b_qItem.append(tf.get_variable(
                name=bias_key, shape=[dim_out], dtype=tf.float64,
                initializer=tf.truncated_normal_initializer(stddev=0.001, seed=self.seed))
            )

        # w_qAttr[0] stores embedding of attributes
        self.w_qAttr, self.b_qAttr = [], []
        for i, (dim_in, dim_out) in enumerate(zip(self.qAttr_dims_in, self.qAttr_dims_out)):
            if i == len(self.qAttr_dims_in) - 1:
                dim_out *= 2
            weight_key = "weight_qAttr_{}to{}".format(i, i + 1)
            self.w_qAttr.append(tf.get_variable(
                name=weight_key, shape=[dim_in, dim_out], dtype=tf.float64,
                initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
            )
            bias_key = "bias_qAttr_{}".format(i + 1)
            self.b_qAttr.append(tf.get_variable(
                name=bias_key, shape=[dim_out], dtype=tf.float64,
                initializer=tf.truncated_normal_initializer(stddev=0.001, seed=self.seed))
            )

        # embedding tables
        self.embs_item_self = tf.get_variable(
            name="embs_item", shape=[self.num_items, self.dim], dtype=tf.float64,
            initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
        self.embs_item_enti = [tf.get_variable(
            name="embs_item_"+str(k), shape=[self.num_items, self.dim], dtype=tf.float64,
            initializer=tf.contrib.layers.xavier_initializer(seed=self.seed)) for k in range(self.K)]
        self.cate_item = tf.get_variable(
            name="embs_aspect", shape=[self.K, self.dim], dtype=tf.float64,
            initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
        self.cate_attr = tf.get_variable(
            name="attr_aspect", shape=[self.K, self.dim], dtype=tf.float64,
            initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))

        # placeholders
        self.input_ph = tf.placeholder(dtype=tf.float64, shape=[None, num_items])
        self.is_training_ph = tf.cast(tf.placeholder_with_default(0., shape=None), tf.float64)
        self.beta_ph = tf.cast(tf.placeholder_with_default(1., shape=None), tf.float64)
        self.dropout_p_ph = tf.cast(tf.placeholder_with_default(1., shape=None), tf.float64)


    def build_graph(self, save_emb=False):
        saver, logits, recon_loss, kl = self.forward(save_emb=False)

        reg_var = apply_regularization(
            l2_regularizer(self.lam),
            self.w_qItem + self.w_qAttr + self.embs_item_enti + [self.embs_item_self, self.cate_item, self.cate_attr])

        neg_elbo = recon_loss + self.beta_ph * kl + tf.cast(2. * reg_var, dtype=tf.float64)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(neg_elbo)

        return saver, logits, train_op


    def q_network(self, x, w_list, b_list):
        # The q-network contains two layers, an embedding layer and a mapping layer
        mu_q, std_q, kl = None, None, None
        repr = tf.nn.l2_normalize(x, 1)
        repr = tf.nn.dropout(repr, self.dropout_p_ph)
        for i, (w, b) in enumerate(zip(w_list, b_list)):
            repr = tf.matmul(repr, w, a_is_sparse=(i==0)) + b
            if i != len(w_list) - 1:
                repr = tf.nn.tanh(repr)
            else:
                mu_q = repr[:, :self.dim]
                mu_q = tf.nn.l2_normalize(mu_q, axis=1)
                lnvarq_sub_lnvar0 = -repr[:, self.dim:]
                std_q = tf.exp(0.5 * lnvarq_sub_lnvar0) * self.std
                kl = tf.reduce_mean(tf.reduce_sum(
                    0.5 * (-lnvarq_sub_lnvar0 + tf.exp(lnvarq_sub_lnvar0) - 1.),
                    axis=1))
        return mu_q, std_q, kl

    def q_network_sparse(self, x, w_list, b_list):
        # The q-network contains two layers, an embedding layer and a mapping layer
        mu_q, std_q, kl = None, None, None
        denom = tf.sparse_reduce_sum(tf.square(x), axis=1, keep_dims=True)
        repr = x.__div__(denom)
        for i, (w, b) in enumerate(zip(w_list, b_list)):
            if i != len(w_list) - 1:
                repr = tf.sparse_tensor_dense_matmul(repr, w, adjoint_a=False, adjoint_b=False)
                repr = tf.nn.tanh(repr)
            else:
                repr = tf.matmul(repr, w, a_is_sparse=(i == 0)) + b
                mu_q = repr[:, :self.dim]
                mu_q = tf.nn.l2_normalize(mu_q, axis=1)
                lnvarq_sub_lnvar0 = -repr[:, self.dim:]
                std_q = tf.exp(0.5 * lnvarq_sub_lnvar0) * self.std
                kl = tf.reduce_mean(tf.reduce_sum(
                    0.5 * (-lnvarq_sub_lnvar0 + tf.exp(lnvarq_sub_lnvar0) - 1.),
                    axis=1))
        return mu_q, std_q, kl

    def qInfer_network(self, w_list, b_list):
        mu_q = None
        for i, (w, b) in enumerate(zip(w_list, b_list)):
            if i != len(w_list) - 1:
                repr = w + b
                repr = tf.nn.tanh(repr)
            else:
                repr = tf.matmul(repr, w, a_is_sparse=(i == 0)) + b
                mu_q = repr[:, :self.dim]
                mu_q = tf.nn.l2_normalize(mu_q, axis=1)

        return mu_q

    def qInfer_network_sparse(self, w_list, b_list):
        mu_q = None
        for i, (w, b) in enumerate(zip(w_list, b_list)):
            if i != len(w_list) - 1:
                repr = w + b
                repr = tf.nn.tanh(repr)
            else:
                repr = tf.matmul(repr, w, a_is_sparse=(i == 0)) + b
                mu_q = repr[:, :self.dim]
                mu_q = tf.nn.l2_normalize(mu_q, axis=1)
        return mu_q


    def encoder_item(self, embs_cate, w_list, b_list):
        embs_cate = tf.nn.l2_normalize(embs_cate, axis=1)
        embs_cnxt = tf.nn.l2_normalize(self.qInfer_network(w_list, b_list), axis=1)
        p_assign = tf.matmul(embs_cnxt, embs_cate, transpose_b=True) / self.tau

        if not self.gumbel:
            cates = tf.nn.softmax(p_assign, axis=1)
        else:
            cates_dist = RelaxedOneHotCategorical(1, p_assign)
            cates_sample = cates_dist.sample()
            cates_mode = tf.nn.softmax(p_assign, axis=1)
            cates = (self.is_training_ph * cates_sample +
                     (1 - self.is_training_ph) * cates_mode)

        # VAE based encoding
        z_list, cate_list = [], []
        kl = None
        for k in range(self.K):
            cates_k = tf.reshape(cates[:, k], (1, -1))
            cate_list.append(cates_k)

            # q-network
            x_k = self.input_ph * cates_k
            mu_k, std_k, kl_k = self.q_network(x_k, w_list, b_list)
            eps = tf.random_normal(tf.shape(std_k), dtype=tf.float64)
            z_k = mu_k + self.is_training_ph * eps * std_k
            z_list.append(z_k)
            kl = (kl_k if (kl is None) else (kl + kl_k))

        return z_list, cate_list, kl


    def encoder_attr(self, embs_cate, w_list, b_list):
        embs_cate = tf.nn.l2_normalize(embs_cate, axis=1)
        embs_cnxt = tf.nn.l2_normalize(self.qInfer_network_sparse(w_list, b_list), axis=1)
        p_assign = tf.matmul(embs_cnxt, embs_cate, transpose_b=True) / self.tau

        if not self.gumbel:
            cates = tf.nn.softmax(p_assign, axis=1)
        else:
            cates_dist = RelaxedOneHotCategorical(1, p_assign)
            cates_sample = cates_dist.sample()
            cates_mode = tf.nn.softmax(p_assign, axis=1)
            cates = (self.is_training_ph * cates_sample +
                     (1 - self.is_training_ph) * cates_mode)

        # VAE based encoding
        z_list = []
        zItem_list, cateItem_list = [], []
        kl = None
        x_input2attr = tf.sparse_tensor_dense_matmul(self.kg_mat, self.input_ph, adjoint_a=True, adjoint_b=True)
        x_input2attr = tf.transpose(x_input2attr)
        for k in range(self.K):
            cates_k = tf.reshape(cates[:, k], (1, -1))

            # q-network for user aspects
            x_k = x_input2attr * cates_k
            mu_k, std_k, kl_k = self.q_network(x_k, w_list, b_list)
            eps = tf.random_normal(tf.shape(std_k), dtype=tf.float64)
            z_k = mu_k + self.is_training_ph * eps * std_k
            z_list.append(z_k)
            kl = (kl_k if (kl is None) else (kl + kl_k))

            # q-network for item aspects
            x_k = self.kg_mat.__mul__(cates_k)
            mu_k, std_k, kl_k = self.q_network_sparse(x_k, w_list, b_list)
            eps = tf.random_normal(tf.shape(std_k), dtype=tf.float64)
            z_k = mu_k + self.is_training_ph * eps * std_k
            zItem_list.append(z_k)
            cates_sum_k = tf.sparse_reduce_sum(x_k, axis=1)
            cates_sum_k = tf.reshape(cates_sum_k, (1, -1))
            cateItem_list.append(cates_sum_k / tf.reduce_sum(cates_sum_k))

        return z_list, zItem_list, cateItem_list, kl


    def decoder(self, zAct_list, cAct_list, embs_item_self, embs_item_enti,
                zAsp_list, zItem_list, cAsp_list):
        embs_item_self = tf.nn.l2_normalize(embs_item_self, axis=1)

        probs = None
        for k in range(self.K):
            z_k = zAct_list[k]
            z_k_asp = zAsp_list[k]
            z_k_aspItem = zItem_list[k]
            cates_k = cAct_list[k]
            cates_k_asp = cAsp_list[k]

            z_k = tf.nn.l2_normalize(z_k, axis=1)
            logits_k = tf.matmul(z_k, embs_item_self, transpose_b=True) / self.tau
            probs_k = tf.exp(logits_k)
            probs_k = probs_k * cates_k
            probs = (probs_k if (probs is None) else (probs + probs_k))

            z_k_asp = tf.nn.l2_normalize(z_k_asp, axis=1)
            z_k_aspItem = tf.nn.l2_normalize(embs_item_enti[k], axis=1)
            logits_k = tf.matmul(z_k_asp, z_k_aspItem, transpose_b=True) / self.tau
            probs_k = tf.exp(logits_k)
            probs_k = probs_k * cates_k_asp
            probs = (probs_k if (probs is None) else (probs + self.coef_attr * probs_k))

        logits = tf.log(probs)
        logits = tf.nn.log_softmax(logits)
        loss_recon = tf.reduce_mean(tf.reduce_sum(
            -logits * self.input_ph, axis=-1))

        return logits, loss_recon


    def forward(self, save_emb=False):
        denom = tf.reduce_sum(self.input_ph, axis=1, keep_dims=True)
        self.input_ph = self.input_ph.__div__(denom)
        denom = tf.sparse_reduce_sum(self.kg_mat, axis=1, keep_dims=True)
        self.kg_mat = self.kg_mat.__div__(denom)

        zAct_list, cAct_list, klAct = self.encoder_item(self.cate_item, self.w_qItem, self.b_qItem)

        zAsp_list, zItem_list, cateItem_list, klAsp = self.encoder_attr(self.cate_attr,
                                                                        self.w_qAttr, self.b_qAttr)

        logits, loss_recon = self.decoder(zAct_list, cAct_list, self.embs_item_self, self.embs_item_enti,
                                          zAsp_list, zItem_list, cateItem_list)

        return tf.train.Saver(), logits, loss_recon, klAct
