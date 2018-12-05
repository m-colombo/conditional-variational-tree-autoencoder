"""
Train a conditional variational autoencoder to learn the distribution of P(english_sentence | german_sentence)
in order to be able to get an english translation decoding z ~ N(0,I) | german_sentence.

Due to the high number of categories we group all leaves in a single family and all internal nodes in another one.
Every node output the category as a 1ofk encoding. Total category are 82, to much paramters!
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from benchmark import Measure
from experiments.wmt_en_de import load_all_data_and_build_model, define_flags
from functools import  reduce
import numpy as np

# benchmark flags

tf.flags.DEFINE_integer(
    "benchmark_runs",
    default=1,
    help=""
)

FLAGS = tf.flags.FLAGS


def arities(ts):
    def _visit(t):
        if t.children:
            return [len(t.children)] + list(reduce(lambda x,y: x+y, map(_visit, t.children)))
        else:
            return []

    arities = []
    for t in ts:
        arities.extend(_visit(t))
    return arities

def main(argv):

    train_data, valid_data, cvae = load_all_data_and_build_model()

    times = {}
    global loss_times
    loss_times = {}
    optimizer = tf.train.AdamOptimizer()

    # training cycle
    node_count = []
    node_depth = []
    node_arity = []
    for i in range(FLAGS.benchmark_runs):

        with Measure('data', times):
            train_data_iter = train_data.iter(0, FLAGS.ignore_leaves)
            xs, ys = next(train_data_iter)

        node_count.append(list(map(lambda t: t.calculate_node_count(), xs + ys)))
        node_depth.append(list(map(lambda t: t.calculate_max_depth(), xs + ys)))
        node_arity.extend(arities(xs + ys))


        with tfe.GradientTape() as tape:

            with Measure('compute', times):
                kld_all, recons = cvae.get_loss_components_trees(xs, ys, FLAGS.n_sample)

            with Measure('loss', times):
                struct_loss_all, val_loss_all = recons.reconstruction_loss()
                loss = tf.reduce_sum(kld_all)+tf.reduce_sum(struct_loss_all+val_loss_all)

        with Measure('grad', times):
            grad = tape.gradient(loss, cvae.variables)

        with Measure('apply', times):
            optimizer.apply_gradients(zip(grad, cvae.variables), global_step=tf.train.get_or_create_global_step())

    # Printing output
    print("#ALL")

    tot_avg, tot_sum = Measure.print_times(times)

    print("\nNodes: {0:.1f} ({1:.1f})".format(np.mean(node_count), np.std(node_count)))
    print("Depths: {0:.1f} ({1:.1f})".format(np.mean(node_depth), np.std(node_depth)))
    print("Arities: {0:.1f} ({1:.1f})".format(np.mean(node_arity), np.std(node_arity)))
    print((np.sum(node_count) / tot_sum), FLAGS.batch_size * FLAGS.benchmark_runs / tot_sum)
    # print("\n#CVAE")
    # Measure.print_times(cvae.loss_times)
    #
    # print("\n#DEC")
    # Measure.print_times(cvae._det_decoder.__class__.times)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")   # deprecated stuff in TF spams the console
    define_flags()
    tfe.run()