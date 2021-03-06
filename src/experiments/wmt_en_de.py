"""
Train a conditional variational autoencoder to learn the distribution of P(english_sentence | german_sentence)
in order to be able to get an english translation decoding z ~ N(0,I) | german_sentence.

Due to the high number of categories we group all leaves in a single family and all internal nodes in another one.
Every node output the category as a 1ofk encoding. Total category are 82, to much paramters!
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow.contrib.summary as tfs

from data.wmt_en_de import Dataset, load_dictionary, build_tree_def, get_stat_from_tree_file, en_normalize_word, de_normalize_word
from vae import ConditionalVariationalAutoEncoder as CVAE
from tree_decoder import Decoder, DecoderCellsBuilder
from tree_encoder import Encoder, EncoderCellsBuilder
from utils import TwoWindowAverage
import numpy as np
import traceback
import os, json, itertools, time, glob


def define_flags():
    tf.flags.DEFINE_string(
        "data_dir",
        default="../experiment_data/",
        help="Directory where all the data files are stored")

    tf.flags.DEFINE_string(
        "dict_files_pattern",
        default="{{LANG}}.50k.embedding.dict.pickle",
        help="Pickled dictionary with the precomputed word embeddings for LANG = de | en")

    tf.flags.DEFINE_string(
        "dataset_files_pattern",
        default="{{SET}}.{{LANG}}.trees.shuffled.txt",
        help="Pattern to retrieve all the datasets formatted as generated from  data/wmt_en_de/prepare_dataset"
             ". SET = train | valid | test ; LANG = en | de ")

    tf.flags.DEFINE_integer(
        "dataset_sample_to_skip",
        default=0,
        help="")

    tf.flags.DEFINE_string(
        "model_dir",
        default="/tmp/wmt_en_de/",
        help="Directory to put the model's summaries, parameters and checkpoint.")

    tf.flags.DEFINE_integer(
        "checkpoint_every",
        default=1000,
        help="save the model ever n mini-batch")

    tf.flags.DEFINE_boolean(
        "restore",
        default=False,
        help="Whether to restore a previous saved model")

    tf.flags.DEFINE_boolean(
        "overwrite",
        default=False,
        help="Whether to overwrite existing model directory")

    tf.flags.DEFINE_integer(
        "check_every",
        default=250 * 64,   # as 250 batch of 64 samples
        help="Evaluate summaries every check_every samples")

    tf.flags.DEFINE_integer(
        "max_iter",
        default=int(4.5e6 * 10),     # ~ 10 epoch
        help="Maximum iteration  to run the training process")

    tf.flags.DEFINE_integer(
        "early_stop_window_size",
        default=50 * 250 * 64,      # as 50 checks runs every 250 batch of 64 samples
        help="how many samples are into an early stop window. (two consecutive windows are used to compute average improvement)")

    tf.flags.DEFINE_float(
        "early_stop_minimum_improvement",
        default=0.9999,
        help="")

    tf.flags.DEFINE_integer(
        "max_depth",
        default=28,
        help="Maximum tree depth - exceeding trees are skipped - generated trees are truncated")

    tf.flags.DEFINE_integer(
        "cut_arity",
        default=25,
        help="When using flat strategy the trees exceeding this cardinality are generated by the same cell")

    tf.flags.DEFINE_integer(
        "max_arity",
        default=100,
        help="Maximum tree arity - exceeding trees are skipped - generated trees are bound"
    )

    tf.flags.DEFINE_integer(
        "max_node_count",
        default=120,
        help="Maximum total node count. - exceeding trees are skipped - generated trees are truncated"
    )

    tf.flags.DEFINE_integer(
        "embedding_size",
        default=50,
        help="Size of the embedding size used during tree processing - also the latent space size")

    tf.flags.DEFINE_string(
        "enc_variable_arity_strategy",
        default="FLAT",
        help="FLAT or REC"
    )

    tf.flags.DEFINE_string(
        "dec_variable_arity_strategy",
        default="FLAT",
        help="FLAT or REC"
    )

    tf.flags.DEFINE_integer(
        "batch_size",
        default=16,
        help="")

    tf.flags.DEFINE_string(
        "activation",
        default='tanh',
        help="activation used where there are no constraints"
    )

    tf.flags.DEFINE_float(
        "kld_rescale",
        default=.0010,
        help="coefficient to rescale Kullback-Lieber divergence in the loss")

    tf.flags.DEFINE_integer(
        "n_sample",
        default=2,
        help="how many sampling to use when computing loss on decoding from code distribution")

    tf.flags.DEFINE_float(
        "hidden_cell_coef",
        default=0.3,
        help="user to linear regres from input-output size to compute hidden size")

    tf.flags.DEFINE_float(
        "memory_fraction",
        default=1.0,
        help="Maximum fraction of GPU memory")

    tf.flags.DEFINE_bool(
        "allow_memory_growth",
        default=False,
        help="Whether to incrementally allocate memory of get all in one shot")

    tf.flags.DEFINE_boolean(
        "ignore_leaves",
        default=False,
        help="whether to ignore leaves values, focusing only on the structure")

    tf.flags.DEFINE_boolean(
        "ignore_leaves_loss",
        default=False,
        help="whether to ignore leaves values contribution to loss")

    tf.flags.DEFINE_boolean(
        "encoder_gate",
        default=False,
        help="")

    tf.flags.DEFINE_boolean(
        "decoder_gate",
        default=False,
        help="")

    tf.flags.DEFINE_float(
        "clip_grad",
        default=1.0,
        help="")

    tf.flags.DEFINE_float(
        "adam_eps",
        default=1e-8,
        help="")

FLAGS = tf.flags.FLAGS


def model(tree_def, cond_tree_def, activation):
    return CVAE(
        FLAGS.embedding_size,
        det_encoder=Encoder(
            tree_def=tree_def,
            embedding_size=FLAGS.embedding_size,
            cut_arity=FLAGS.cut_arity, max_arity=FLAGS.max_arity,
            variable_arity_strategy=FLAGS.enc_variable_arity_strategy,
            cellsbuilder=EncoderCellsBuilder(
                EncoderCellsBuilder.simple_cell_builder(hidden_coef=FLAGS.hidden_cell_coef,
                                                        activation=activation,
                                                        gate=FLAGS.encoder_gate),
                EncoderCellsBuilder.simple_dense_embedder_builder(activation=activation),
                EncoderCellsBuilder.simple_categorical_merger_builder(hidden_coef=FLAGS.hidden_cell_coef,
                                                                      activation=activation),
            ),
            name='encoder'
        ),
        det_decoder=Decoder(
            tree_def=tree_def,
            embedding_size=FLAGS.embedding_size,
            max_node_count=FLAGS.max_node_count,
            max_depth=FLAGS.max_depth,
            max_arity=FLAGS.max_arity,
            cut_arity=FLAGS.cut_arity,
            cellbuilder=DecoderCellsBuilder(DecoderCellsBuilder.simple_distrib_cell_builder(FLAGS.hidden_cell_coef, activation=activation),
                                            DecoderCellsBuilder.node_map({
                                                'NODE': DecoderCellsBuilder.simple_1ofk_value_inflater_builder(FLAGS.hidden_cell_coef,activation=activation),
                                                'PRE_LEAF': DecoderCellsBuilder.simple_1ofk_value_inflater_builder(
                                                    FLAGS.hidden_cell_coef, activation=activation),
                                                'LEAF': DecoderCellsBuilder.simple_dense_value_inflater_builder(FLAGS.hidden_cell_coef, activation=activation)}),
                                            DecoderCellsBuilder.simple_node_inflater_builder(FLAGS.hidden_cell_coef,
                                                                                             activation=activation,
                                                                                             gate=FLAGS.decoder_gate)),
            variable_arity_strategy=FLAGS.dec_variable_arity_strategy)
        ,
        cond_encoder=Encoder(
            tree_def=cond_tree_def,
            embedding_size=FLAGS.embedding_size,
            cut_arity=FLAGS.cut_arity,
            cellsbuilder=EncoderCellsBuilder(
                EncoderCellsBuilder.simple_cell_builder(hidden_coef=FLAGS.hidden_cell_coef, activation=activation, gate=FLAGS.encoder_gate),
                EncoderCellsBuilder.simple_dense_embedder_builder(activation=activation),
                EncoderCellsBuilder.simple_categorical_merger_builder(hidden_coef=FLAGS.hidden_cell_coef, activation=activation)
            ), max_arity=FLAGS.max_arity, name="condition_encoder",
            variable_arity_strategy=FLAGS.enc_variable_arity_strategy))


def load_all_data_and_build_model(l='de', l_cond='en'):
    print("Loading data and building the model for P("+l+"|"+l_cond+")")
    normalize_dict = {
        'de': de_normalize_word,
        'en': en_normalize_word,
    }

    # Loading all the data
    with open(os.path.join(FLAGS.data_dir, "all_labels.json")) as f:
        labels = json.load(f)

    lDict = load_dictionary(os.path.join(FLAGS.data_dir, FLAGS.dict_files_pattern.replace("{{LANG}}", l)))
    condDict = load_dictionary(os.path.join(FLAGS.data_dir, FLAGS.dict_files_pattern.replace("{{LANG}}", l_cond)))

    # # normalize dicts
    # def f(d):
    #     max, min = tf.reduce_max(lDict[1]), tf.reduce_min(lDict[1])
    #     return ((d - min) / (max - min)) * 2 - 1 # [-1, 1]
    # lDict = lDict[0], f(lDict[1])
    # condDict = condDict[0], f(condDict[1])
    #
    lTreeDef = build_tree_def(labels, lDict[1], FLAGS.ignore_leaves)
    condTreeDef = build_tree_def(labels, condDict[1], FLAGS.ignore_leaves)

    train_data = Dataset(directory=FLAGS.data_dir,
                              x_file=FLAGS.dataset_files_pattern.replace("{{LANG}}", l).replace("{{SET}}", "train"),
                              y_file= FLAGS.dataset_files_pattern.replace("{{LANG}}", l_cond).replace("{{SET}}", "train"),
                              x_tree_def=lTreeDef, x_dict=lDict,
                              y_tree_def=condTreeDef, y_dict=condDict,
                              x_normalizer=normalize_dict[l], y_normalizer=normalize_dict[l_cond],
                              batch_size=FLAGS.batch_size,
                              max_arity=FLAGS.max_arity, max_depth=FLAGS.max_depth, max_node_count=FLAGS.max_node_count)

    valid_data = Dataset(directory=FLAGS.data_dir,
                              x_file=FLAGS.dataset_files_pattern.replace("{{LANG}}", l).replace("{{SET}}", "valid"),
                              y_file=FLAGS.dataset_files_pattern.replace("{{LANG}}", l_cond).replace("{{SET}}","valid"),
                              x_tree_def=lTreeDef, x_dict=lDict,
                              y_tree_def=condTreeDef, y_dict=condDict,
                              x_normalizer=normalize_dict[l], y_normalizer=normalize_dict[l_cond],
                              batch_size=FLAGS.batch_size,
                              max_arity=FLAGS.max_arity, max_depth=FLAGS.max_depth, max_node_count=FLAGS.max_node_count)

    cvae = model(lTreeDef, condTreeDef, getattr(tf.nn, FLAGS.activation))

    return train_data, valid_data, cvae


def main(argv):
    del argv # unused

    if tf.gfile.Exists(FLAGS.model_dir):
        if FLAGS.overwrite:
            tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
            tf.gfile.DeleteRecursively(FLAGS.model_dir)
            tf.gfile.MakeDirs(FLAGS.model_dir)
        elif not FLAGS.restore:
            raise ValueError("Log dir already exists!")
    else:
        tf.gfile.MakeDirs(FLAGS.model_dir)

    RESTORE = FLAGS.restore and (not hasattr(FLAGS, 'worker_id') or os.path.exists(os.path.join(FLAGS.model_dir, FLAGS.worker_id + '_run_info.json')))

    if not RESTORE:
        with open(os.path.join(FLAGS.model_dir, "flags_info.json"), 'w') as f:
            json.dump(FLAGS.flag_values_dict(), f)
    else:
        with open(os.path.join(FLAGS.model_dir, "flags_info.json")) as f:
            info = json.load(f)
            override_flags = ["embedding_size", "activation", "max_arity", "hidden_cell_coef", "cut_arity",
                              "max_node_count", "kld_rescale", "encoder_gate", "decoder_gate", "ignore_leaves", "ignore_leaves_loss",
                              "enc_variable_arity_strategy", "dec_variable_arity_strategy"]
        for f in override_flags:
            setattr(FLAGS, f, info[f])

    summary_writer = tfs.create_file_writer(FLAGS.model_dir, flush_millis=1000)
    summary_writer.set_as_default()
    print("Summaries in " + FLAGS.model_dir)
    with tfs.always_record_summaries():
        train_data, valid_data, cvae = load_all_data_and_build_model()

        optimizer = tf.train.AdamOptimizer(epsilon=FLAGS.adam_eps)

        best_dir = os.path.join(FLAGS.model_dir, "best_" + str(time.time()))
        os.mkdir(best_dir)

        checkpoint = tf.train.Checkpoint(model=cvae, optimizer=optimizer, optimizer_step=tf.train.get_or_create_global_step())
        checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(checkpoint, directory=FLAGS.model_dir, max_to_keep=2)
        best_checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(checkpoint, directory=best_dir, max_to_keep=2)

        just_restored = False

        if RESTORE:
            print("Restoring previously saved model")
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            info_file_names = glob.glob(os.path.join(FLAGS.model_dir, "*_run_info.json"))
            if len(info_file_names) > 0:
                info_file_name = info_file_names[0]
                print(info_file_name)
                with open(info_file_name) as f:
                    run_info = json.load(f)
            else:
                run_info = {
                    'total_sample': 0,
                    'skipped_train_data': 0,
                    'checks': 0,
                    'iteration': 0,
                    'best_perf': 0.0,
                    'window_average': {}
                }
            # trying to fix some issue when restarting from a checkpoint - it sometimes fail to save further checkpoints
            checkpoint_manager = None
            checkpoint = None
            just_restored = True

            run_info['best_perf'] = run_info.get('best_perf', 0)

            e = tf.Event()
            e.step = int(tf.train.get_or_create_global_step())
            e.session_log.status = e.session_log.SessionStatus.Value("START")
            tfs.import_event(e.SerializeToString())


        else:
            run_info = {
                'total_sample': 0,
                'skipped_train_data': 0,
                'checks': 0,
                'iteration': 0,
                'best_perf': 0.0,
                'window_average': {}
            }

        train_data_iter = train_data.iter((run_info['total_sample'] + run_info['skipped_train_data']) % 4468840, FLAGS.ignore_leaves)
        valid_data_iter = valid_data.iter(ignore_leaves=FLAGS.ignore_leaves)

        early_stop_window_average = TwoWindowAverage(FLAGS.early_stop_window_size // FLAGS.check_every, inital_value=float('inf'))
        if RESTORE and 'window_average' in run_info and 'buffer' in run_info['window_average']:
            early_stop_window_average._buffer = run_info['window_average']['buffer']
            early_stop_window_average._first_index = run_info['window_average']['first_index']

        # training cycle
        failed_count = 0
        consecutive_failed = 0
        failed_count_on_conf = 0        # count failed iteration with the specific configuration (batch size / number of samples)
        all_count_on_conf = 0           # count all the iteration with the specific configuration (batch size / number of samples)

        for i in itertools.count(run_info['iteration']):

            # adjust counting on successful iteration
            i -= failed_count

            # finish check
            if i > FLAGS.max_iter // FLAGS.batch_size:
                break

            # adjust configuration to fit memory (all failure so far are due to GPU OOM)
            if consecutive_failed > 2 or (all_count_on_conf > 10 and failed_count_on_conf / all_count_on_conf > 0.3):
                all_count_on_conf = 0
                failed_count_on_conf = 0

                if consecutive_failed > 10:
                    print("Probably GPU got stacked - quitting")
                    run_info['iteration'] = i
                    run_info['total_sample'] += FLAGS.batch_size # skipping some data
                    run_info['window_average'] = {}
                    run_info['window_average']['buffer'] = early_stop_window_average._buffer
                    run_info['window_average']['first_index'] = early_stop_window_average._first_index
                    with open(os.path.join(FLAGS.model_dir, getattr(FLAGS, 'worker_id', '') + '_run_info.json'),'w') as f:
                        json.dump(run_info, f)

                    exit(0)

                if FLAGS.n_sample > 2:
                    FLAGS.n_sample -= 1
                    print("# Reduced n_sample to ", str(FLAGS.n_sample))

                elif FLAGS.batch_size >= 16:
                    FLAGS.batch_size -= 8
                    train_data.batch_size = FLAGS.batch_size
                    valid_data.batch_size = FLAGS.batch_size
                    print("# Reduced batch_size to ", str(FLAGS.batch_size))

                else:
                    print("Model to big")
                    raise ValueError("Model too big!")

            xs, ys = next(train_data_iter)

            try:
                with tfe.GradientTape() as tape:

                    kld_all, recons = cvae.get_loss_components_trees(xs, ys, FLAGS.n_sample)
                    struct_loss, val_loss = recons.reconstruction_loss(ignore_values={"LEAF"} if FLAGS.ignore_leaves_loss else {})
                    kld = tf.reduce_mean(kld_all)
                    loss = kld * FLAGS.kld_rescale + struct_loss + val_loss

                if run_info['total_sample'] >= run_info['checks'] * FLAGS.check_every:
                    run_info['checks'] += 1
                    tfs.scalar("tr/loss/kld", kld)
                    tfs.scalar("tr/loss/struct", struct_loss)
                    tfs.scalar("tr/loss/val", val_loss)
                    tfs.scalar("tr/loss/loss", loss)

                    xs, ys = next(valid_data_iter)
                    kld_all, recons = cvae.get_loss_components_trees(xs, ys, FLAGS.n_sample)
                    vl_struct_loss, vl_val_loss = recons.reconstruction_loss(ignore_values={"LEAF"} if FLAGS.ignore_leaves_loss else {})
                    vl_kld = tf.reduce_mean(kld_all)
                    vl_loss = vl_kld * FLAGS.kld_rescale + vl_struct_loss + vl_val_loss

                    tfs.scalar("vl/loss/kld", vl_kld)
                    tfs.scalar("vl/loss/struct", vl_struct_loss)
                    tfs.scalar("vl/loss/val", vl_val_loss)
                    tfs.scalar("vl/loss/loss", vl_loss)

                    superv, unsuperv = cvae.assess_unsupervised(xs, ys, n_samples=1)
                    structural_overlap_uns, values_overlaps_uns = zip(*[xs[i].compute_overlaps(unsuperv.decoded_trees[i], True, skip_leaves_value=not FLAGS.ignore_leaves) for i in range(len(xs))])
                    values_overlaps_sup = [xs[i].compute_overlaps(superv.decoded_trees[i], True,skip_leaves_value=not FLAGS.ignore_leaves)[1] for i in range(len(xs))]

                    # looking for a perfect match
                    idx = np.argmax(np.multiply(structural_overlap_uns,values_overlaps_uns))
                    print(np.max(np.multiply(structural_overlap_uns, values_overlaps_uns)))
                    print(idx)
                    if idx:
                        print(str(xs[idx]))
                        print(str(unsuperv.decoded_trees[idx]))
                        print(str(ys[idx]))

                    avg_structural_overlap_uns = float(np.mean(structural_overlap_uns))
                    avg_values_overlaps_uns = float(np.mean(values_overlaps_uns))
                    avg_values_overlaps_sup = float(np.mean(values_overlaps_sup))
                    acc_struct_uns = np.sum(np.equal(structural_overlap_uns, 1.0)) / len(structural_overlap_uns)
                    acc_val_uns = np.sum(np.equal(values_overlaps_uns, 1.0)) / len(values_overlaps_uns)
                    acc_val_sup = np.sum(np.equal(values_overlaps_sup, 1.0)) / len(values_overlaps_sup)

                    tfs.scalar("vl/avg_structural_overlap_uns", avg_structural_overlap_uns)
                    tfs.scalar("vl/avg_values_overlaps_uns", avg_values_overlaps_uns)
                    tfs.scalar("vl/avg_values_overlaps_sup", avg_values_overlaps_sup)

                    tfs.scalar("vl/acc_struct_uns", acc_struct_uns)
                    tfs.scalar("vl/acc_val_uns", acc_val_uns)
                    tfs.scalar("vl/acc_val_sup", acc_val_sup)

                    tf.contrib.summary.flush()
                    summary_writer.flush()
                    # save new best model
                    if run_info['best_perf'] < avg_values_overlaps_uns:
                        run_info['best_perf'] = avg_values_overlaps_uns

                        with open(os.path.join(best_dir, 'info.json'), 'w') as finfo:
                            json.dump({
                                "loss": float(loss),
                                "kld": float(kld),
                                "struct_loss": float(struct_loss),
                                "val_loss": float(val_loss),
                                "vl_loss": float(vl_loss),
                                "vl_kld": float(vl_kld),
                                "vl_struct_loss": float(vl_struct_loss),
                                "vl_val_loss": float(vl_val_loss),
                                'iteration': i,
                                'structural_overlap_uns': avg_structural_overlap_uns,
                                'values_overlaps_uns': avg_values_overlaps_uns,
                                'values_overlaps_sup': avg_values_overlaps_sup,
                                "vl/acc_struct_uns": acc_struct_uns,
                                "vl/acc_val_uns": acc_val_uns,
                                "vl/acc_val_sup": acc_val_sup,
                            }, finfo)

                        best_checkpoint_manager.save()
                        print("\n### saved best checkpoint")

                    # early stop
                    early_stop_window_average.add(vl_loss)
                    old, new = early_stop_window_average.get_averages()
                    if new/old > FLAGS.early_stop_minimum_improvement:
                        print("### EARLY STOP %f - %f" % (old, new))
                        break

                    print("\n{0:05}\tTR {1:.3e}: {2:.3e} {3:.3e} {4:.3e} \t{12} ({10}): {13}x{14} \n{9:.3f}e\tVL {5:.3e}: {6:.3e} {7:.3e} {8:.3e}\t\t{11:.4f}\t[{15}, {16}]".format(i, loss, kld, struct_loss, val_loss, vl_loss, vl_kld, vl_struct_loss, vl_val_loss, i * FLAGS.batch_size / 4.5e6, failed_count, new/old, failed_count_on_conf, FLAGS.batch_size, FLAGS.n_sample, train_data.skipped_count, valid_data.skipped_count))
                    print(
                        "\t\t{0:.2f} ({3:.2f})\t{1:.2f} ({4:.2f})\t{2:.2f} ({5:.2f})".format(avg_structural_overlap_uns,
                                                                                             avg_values_overlaps_uns,
                                                                                             avg_values_overlaps_sup,
                                                                                             acc_struct_uns,
                                                                                             acc_val_uns, acc_val_sup))

                grad = tape.gradient(loss, cvae.variables)
                grad, norm = tf.clip_by_global_norm(grad, FLAGS.clip_grad)
                tfs.scalar("grad_norm", norm)
                optimizer.apply_gradients(zip(grad, cvae.variables), global_step=tf.train.get_or_create_global_step())
                
                if i > 0 and i % FLAGS.checkpoint_every == 0 and not just_restored:
                    if checkpoint_manager is None:
                        checkpoint = tf.train.Checkpoint(model=cvae, optimizer=optimizer,
                                                         optimizer_step=tf.train.get_or_create_global_step())
                        checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(checkpoint,
                                                                                     directory=FLAGS.model_dir,
                                                                                     max_to_keep=2)
                        best_checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(checkpoint,
                                                                                     directory=best_dir,
                                                                                     max_to_keep=2)
                    checkpoint_manager.save()
                    print("\n### checkpoint saved to ", FLAGS.model_dir)

                    run_info['iteration'] = i
                    run_info['window_average'] = {}
                    run_info['window_average']['buffer'] = early_stop_window_average._buffer
                    run_info['window_average']['first_index'] = early_stop_window_average._first_index
                    with open(os.path.join(FLAGS.model_dir, getattr(FLAGS, 'worker_id', '') + '_run_info.json'), 'w') as f:
                        json.dump(run_info, f)

                run_info['total_sample'] += FLAGS.batch_size
                all_count_on_conf += 1
                consecutive_failed = 0
                just_restored = False
                print('.', end='', flush=True)
            except Exception as e:
                failed_count_on_conf += 1
                failed_count += 1
                consecutive_failed += 1
                print("Iteration failed: ", str(e))  # TODO not sure that a failed iteration leaves consistent model
                print(traceback.format_exc())

        if checkpoint_manager is None:
            checkpoint = tf.train.Checkpoint(model=cvae, optimizer=optimizer,
                                             optimizer_step=tf.train.get_or_create_global_step())
            checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(checkpoint,
                                                                         directory=FLAGS.model_dir,
                                                                         max_to_keep=2)
            best_checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(checkpoint,
                                                                              directory=best_dir,
                                                                              max_to_keep=2)
        checkpoint_manager.save()
        print("### final checkpoint saved to ", FLAGS.model_dir)
        with open(os.path.join(FLAGS.model_dir, getattr(FLAGS, 'worker_id', '') + '_run_info.json'), 'w') as f:
            json.dump(run_info, f)

        input("Waiting to quit")

    return {
        'stop_reason': 'iteration' if i >= FLAGS.max_iter // FLAGS.batch_size else 'early_stop',
        'iteration': i,
        'last_losses': {
            "loss": float(loss),
            "kld": float(kld),
            "struct_loss": float(struct_loss),
            "val_loss": float(val_loss),
            "vl_loss": float(vl_loss),
            "vl_kld": float(vl_kld),
            "vl_struct_loss": float(vl_struct_loss),
            "vl_val_loss": float(vl_val_loss)
        }
    }

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")   # deprecated stuff in TF spams the console
    define_flags()

    config = tf.ConfigProto()
    tf.enable_eager_execution(config=config)
    config.gpu_options.allow_growth = FLAGS.allow_memory_growth
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory_fraction
    tfe.run()