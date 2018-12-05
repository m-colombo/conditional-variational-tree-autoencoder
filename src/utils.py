import tensorflow as tf


def tf_random_choice_idx(p):
    """Given a batch of probability vector samples from each one an element (index) accordingly to the probability.
        much faster than using np.random.choice on each sample
    :param p: [b, n] b probability vector of size n. (each [i, :] sum up to 1)
    :return: [b] indexes of sampled elements
    """
    sample = tf.random_uniform((p.shape[0].value, 1))
    pivots = tf.concat([tf.cumsum(p, axis=1, exclusive=True), tf.ones((p.shape[0].value, 1))], axis=1)
    mask = tf.logical_and(sample < pivots[:, 1:], sample >= pivots[:, :-1])
    return tf.where(condition=mask)[:, 1]


class TwoWindowAverage:
    def __init__(self, size, inital_value=0.0):
        self._buffer = [inital_value for _ in range(size * 2)]
        self._first_index = 0
        self.size = size

    def add(self, v):
        self._buffer[self._first_index] = float(v)
        self._first_index = (self._first_index + 1) % len(self._buffer)

    def add_many(self, vs):
        for v in vs:
            self.add(v)

    def get_averages(self):
        old = 0.0
        new = 0.0

        for i in range(int(len(self._buffer) / 2)):
            old += self._buffer[(self._first_index + i) % len(self._buffer)]
            new += self._buffer[(self._first_index + i + self.size) % len(self._buffer)]

        return old / float(self.size), new / float(self.size)