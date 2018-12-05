import tensorflow as tf
import tensorflow_probability as tfp
import typing as T
from tree.definition import Tree
from tree.batch import BatchOfTreesForDecoding, BatchOfTreesForEncoding


class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self,
                 latent_size: int,
                 det_encoder: tf.keras.Model,
                 det_decoder: tf.keras.Model):
        """
        :param latent_size: dimension of the latent (codes) space
        :param det_encoder: maps deterministically the desired input to some space, from which the latent distribution parameters are directly mapped
        :param det_decoder: construct a sample in the input space from a code sampled from the latent distribution
        """
        super(VariationalAutoEncoder, self).__init__()

        self._latent_size = latent_size

        self._det_encoder = det_encoder
        self._det_decoder = det_decoder

        # mean should be unbound => linear activation
        self._enc_mean_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_size * 2, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(latent_size, activation=None)], name="MeanProj")

        # covar should be positive and unbound => softplus activation
        self._enc_dvar_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_size * 2, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(latent_size, activation=tf.nn.softplus)], name="DiagVarProj")

        self._latent_prior = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros([latent_size]),
            scale_identity_multiplier=1.0)

    def get_latent_distrib(self, embedding):
        """

        :param embedding:
        :return:
        """

        return tfp.distributions.MultivariateNormalDiag(
            loc=self._enc_mean_proj(embedding),
            scale_diag=self._enc_dvar_proj(embedding),
            name="latent_distrib")

    def encoder(self, input):
        """Build a distribution over the latent space from the input

        :param input: a sample from the input space, it will be fed to the det_encoder
        :return: a distribution over latent (codes) space
        """
        enc = self._det_encoder(input)
        return self.get_latent_distrib(enc)

    def sample_code(self, code_distrib, n_samples=1):
        """

        :param code_distrib:
        :param n_samples:
        :return:
        """
        d = [tf.Dimension(n_samples), code_distrib.batch_shape[0]] if len(code_distrib.batch_shape) > 0 else [n_samples,
                                                                                                              1]
        random_seed = self._latent_prior.sample(
            d)  # actually sampling from a N(0,1). Just using _latent_prior since it's N(0,1)
        sampled_encoding = code_distrib.mean() + random_seed * code_distrib.stddev()

        sampled_encoding = tf.reshape(sampled_encoding, [-1] + list(sampled_encoding.shape[2:]))

        return sampled_encoding

    def decoder(self, code_distrib, n_samples=1):
        """Build samples in the original space starting from samples drawed fromcode_distrib

        :param code_distrib: distribution over the latent (codes) space
        :param n_samples: how many samples to draw from the code_distrib
        :return: [n_samples, batch_size] + input_size
        """

        sampled_encoding = self.sample_code(code_distrib, n_samples)
        reconstructed = self._det_decoder(sampled_encoding)

        if n_samples > 1:
            reconstructed = tf.reshape(reconstructed, [n_samples, -1] + list(reconstructed.shape[1:]))

        return reconstructed

    def __call__(self, input, **kwargs):
        codes = self.encoder(input)
        sampled_reconstruction = self.decoder(codes)
        return sampled_reconstruction

    def get_loss_components(self, input, n_samples=1):
        """

        :param input:  a sample from the input space, it will be fed to the det_encoder
        :param n_samples: how many samples to reconstruct
        :return: KL-divergence sample-wide, reconstructed samples in the shape of [n_samples, batch_size] + input_size
        """
        codes = self.encoder(input)
        sampled_reconstruction = self.decoder(codes, n_samples)

        kld = codes.kl_divergence(self._latent_prior)

        return kld, sampled_reconstruction


class ConditionalVariationalAutoEncoder(VariationalAutoEncoder):
    def __init__(self,
                 latent_size: int,
                 det_encoder: tf.keras.Model,
                 det_decoder: tf.keras.Model,
                 cond_encoder: tf.keras.Model):
        """
        :param latent_size: dimension of the latent (codes) space
        :param det_encoder: maps deterministically the desired input to some space, from which the latent distribution parameters are directly mapped along with condition encoding
        :param det_decoder: construct a sample in the input space from a code sampled from the latent distribution and the condition encoding
        :param cond_encoder: maps the condition (i.e. Y in P[X|Y]) to some space, from which the latent distribution parameters are directly mapped along with input encoding. it's also used during the decoding phase
        """
        super(ConditionalVariationalAutoEncoder, self).__init__(latent_size, det_encoder, det_decoder)
        self._cond_encoder = cond_encoder

    def encoder(self, input, condition_embedding):
        """

        :param input: [batch, (input_size)]
        :param condition_embedding: [batch, cond_emb_size]
        :return:
        """

        enc = self._det_encoder(input)

        both = tf.concat([tf.layers.flatten(enc), tf.layers.flatten(condition_embedding)], axis=1)

        return self.get_latent_distrib(both)

    def decoder(self, code_distrib, condition_embedding, n_samples=1, **kwargs):  # TODO pass target trees
        """

        :param code_distrib: distribution of size (batch), (latent_size)
        :param condition_embedding: [batch, embedding_size]
        :param n_samples: int
        :param **kwargs: forwarded to the det_decoder
        :return:
        """

        sampled_encoding = self.sample_code(code_distrib, n_samples)
        tiled_condition = tf.tile(condition_embedding, [n_samples] + ([1] * (len(condition_embedding.shape)-1) ))

        reconstructed = self._det_decoder(tf.concat([sampled_encoding, tiled_condition], axis=1), **kwargs)

        if n_samples > 1:
            reconstructed = tf.reshape(reconstructed, [n_samples, -1] + list(reconstructed.shape[1:]))

        return reconstructed

    def __call__(self, input, condition, **kwargs):
        cond = self._cond_encoder(condition)
        codes = self.encoder(input, cond)
        sampled_reconstruction = self.decoder(codes, cond)
        return sampled_reconstruction

    def get_loss_components_(self, input, condition, n_samples=1):
        """

        :param input:  a sample from the input space, it will be fed to the det_encoder
        :param condition:
        :param n_samples: how many samples to reconstruct
        :return: KL-divergence sample-wide, reconstructed samples in the shape of [n_samples, batch_size] + input_size
        """
        cond = self._cond_encoder(condition)
        codes = self.encoder(input, cond)

        # simply add the correct condition
        def augment_fn(input, batch_idx):
            return tf.concat([input, tf.gather(cond, tf.mod(batch_idx, len(condition)))], axis=1)

        sampled_reconstruction = self.decoder(codes, cond, n_samples, target_trees=input, augment_fn=augment_fn)

        kld = codes.kl_divergence(self._latent_prior)

        return kld, sampled_reconstruction


    def get_loss_components_trees(self, input: T.List[Tree], condition: T.List[Tree], n_samples):

        condition_batch = BatchOfTreesForEncoding(condition, self._cond_encoder.embedding_size)
        cond = self._cond_encoder(condition_batch)
        input_batch = BatchOfTreesForEncoding(input, self._det_encoder.embedding_size)
        codes = self.encoder(input_batch, cond)

        # simply add the correct condition
        def augment_fn(input, batch_idx):
            return tf.concat([input, tf.gather(cond, tf.mod(batch_idx, len(condition)))], axis=1)

        sampled_encoding = self.sample_code(codes, n_samples)

        replicated_input = []
        for _ in range(n_samples-1):
            for i in input:
                replicated_input.append(i.clone())
        replicated_input.extend(input)

        batch = BatchOfTreesForDecoding(sampled_encoding, self._det_decoder.tree_def, replicated_input)
        reconstructed = self._det_decoder(batch=batch, augment_fn=augment_fn,
                                          attention_batch=condition_batch if self._det_decoder.attention else None)

        kld = codes.kl_divergence(self._latent_prior)

        return kld, batch

    def assess_unsupervised(self, input: T.List[Tree], condition: T.List[Tree], n_samples):
        condition_batch = BatchOfTreesForEncoding(condition, self._cond_encoder.embedding_size)
        cond = self._cond_encoder(condition_batch)
        input_batch = BatchOfTreesForEncoding(input, self._det_encoder.embedding_size)
        codes = self.encoder(input_batch, cond)

        # simply add the correct condition
        def augment_fn(input, batch_idx):
            return tf.concat([input, tf.gather(cond, tf.mod(batch_idx, len(condition)))], axis=1)

        sampled_encoding = self.sample_code(codes, n_samples)

        replicated_input = []
        for _ in range(n_samples-1):
            for i in input:
                replicated_input.append(i.clone())
        replicated_input.extend(input)

        batch1 = BatchOfTreesForDecoding(sampled_encoding, self._det_decoder.tree_def)
        reconstructed1 = self._det_decoder(batch=batch1, augment_fn=augment_fn,
                                           attention_batch=condition_batch if self._det_decoder.attention else None)

        batch2 = BatchOfTreesForDecoding(tf.concat([self.sample_code(self._latent_prior, n_samples) for _ in range(len(input))], axis=0),
                                         self._det_decoder.tree_def)
        reconstructed1 = self._det_decoder(batch=batch2, augment_fn=augment_fn,
                                           attention_batch=condition_batch if self._det_decoder.attention else None)

        return batch1, batch2

    def sample(self, condition: T.List[Tree]):
        condition_batch = BatchOfTreesForEncoding(condition, self._cond_encoder.embedding_size)
        cond = self._cond_encoder(condition_batch)

        # simply add the correct condition
        def augment_fn(input, batch_idx):
            return tf.concat([input, tf.gather(cond, tf.mod(batch_idx, len(condition)))], axis=1)

        code = self.sample_code(self._latent_prior)
        batch = BatchOfTreesForDecoding(code, self._det_decoder.tree_def)
        reconstructed = self._det_decoder(batch=batch, augment_fn=augment_fn)
        return reconstructed