import tensorflow as tf
from experiments.wmt_en_de import load_all_data_and_build_model, define_flags
import json
import os
import csv
import numpy as np

tf.flags.DEFINE_string(
    "worker_result",
    default="model_selection_result.json",
    help="")


FLAGS = tf.flags.FLAGS

import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main(argv):
    del argv

    with open(FLAGS.worker_result) as f:
        lines = list(map(json.loads, f.readlines()))
        worker_results = {int(w['model_id']): flatten(w) for w in lines if 'model_id' in w}

        worker_results_keys = list(worker_results[list(worker_results.keys())[0]].keys())

    override_flags = ["embedding_size", "activation", "max_arity", "hidden_cell_coef", "cut_arity", "max_node_count",
                      "enc_variable_arity_strategy", "dec_variable_arity_strategy"]

    with open(os.path.join(FLAGS.model_dir, "selection_results.csv"), 'w') as fout:

        fwriter = csv.DictWriter(fout, [
            'model_id',
            'struct_overlap_depth',
            'struct_overlap_nodes',
            'values_overlap_uns',
            'values_overlap_sup'
        ] + override_flags + worker_results_keys)

        fwriter.writeheader()

        all_models_dir = [os.path.join(FLAGS.model_dir+'/',d) for d in os.listdir(FLAGS.model_dir) if d[:5] == "model"]

        for d in all_models_dir:
            try:
                FLAGS.model_dir = d
                model_id = d.split('_')[-1]
                print("\033[92mTesting", model_id, "\033[0m")

                with open(os.path.join(FLAGS.model_dir, "flags_info.json")) as f:
                    info = json.load(f)

                for f in override_flags:
                    setattr(FLAGS, f, info[f])

                _, valid_data, cvae = load_all_data_and_build_model()

                checkpoint = tf.train.Checkpoint(model=cvae)
                checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(checkpoint, directory=FLAGS.model_dir, max_to_keep=1)
                checkpoint.restore(checkpoint_manager.latest_checkpoint)

                valid_data_iter = valid_data.iter()

                # assess ignoring leaves
                x, y = next(valid_data_iter)
                superv, unsuperv = cvae.assess_unsupervised(x, y, n_samples=1)
                overlaps_depth, overlaps_nodes = zip(*[x[i].structural_overlap(unsuperv.decoded_trees[i]) for i in range(len(x))])
                values_overlaps_uns = [x[i].values_overlap(unsuperv.decoded_trees[i], ignore_leaves=True) for i in range(len(x))]
                values_overlaps_sup = [x[i].values_overlap(superv.decoded_trees[i], ignore_leaves=True) for i in range(len(x))]

                row = {
                    'model_id': model_id,
                    'struct_overlap_depth': float(np.mean(overlaps_depth)),
                    'struct_overlap_nodes': float(np.mean(overlaps_nodes)),
                    'values_overlap_uns': float(np.mean(values_overlaps_uns)),
                    'values_overlap_sup': float(np.mean(values_overlaps_sup))
                }

                # adding model info
                for f in override_flags:
                    row[f] = info[f]

                if int(model_id) in worker_results_keys:
                    row = {**row, **(worker_results[int(model_id)])}

                fwriter.writerow(row)
            except Exception as e:
                print('\033[0;31mFailed ', model_id, '\033[0m', str(e))


if __name__ == "__main__":
    define_flags()

    config = tf.ConfigProto()
    tf.enable_eager_execution(config=config)
    tf.contrib.eager.run()
