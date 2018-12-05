import tensorflow as tf
from experiments.wmt_en_de import load_all_data_and_build_model, define_flags
from data.wmt_en_de import en_normalize_word, parse_tree
from pycorenlp import StanfordCoreNLP
import json
import os
import re

tf.flags.DEFINE_string(
    "corenlp_server",
    default="http://localhost:9000",
    help="")


FLAGS = tf.flags.FLAGS


def main(argv):
    del argv

    override_flags = ["embedding_size", "activation", "max_arity", "hidden_cell_coef", "cut_arity", "max_node_count",
                      "enc_variable_arity_strategy", "dec_variable_arity_strategy"]

    # load info
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
    overlaps_depth, overlaps_nodes = zip(*[superv.decoded_trees[i].structural_overlap(unsuperv.decoded_trees[i]) for i in range(len(unsuperv.decoded_trees))])
    values_overlaps = [x[i].values_overlap(unsuperv.decoded_trees[i], ignore_leaves=True) for i in range(len(unsuperv.decoded_trees))]

    corenlp = StanfordCoreNLP(FLAGS.corenlp_server)
    corenlp_properties = {
        'annotators': 'parse',
        'outputFormat': 'json',
        'parse.model': 'edu/stanford/nlp/models/srparser/englishSR.ser.gz'
    }

    all_words = {valid_data.x_dict[0][k]: k for k in valid_data.x_dict[0].keys()}  # not sure this preserves ordering

    while True:
        sentence = input("Enter english sentence: ")
        sentence = sentence.replace(' ##AT##-##AT## ', '-')
        annotated = [s['parse'] for s in corenlp.annotate(sentence, properties=corenlp_properties)['sentences']]
        clenaed = [re.sub(r'  +', ' ', s.replace("\n", " ")) for s in annotated]
        serialized = "%%NEWSENT%%".join(clenaed)+"\n"

        tree = parse_tree(serialized, cvae._cond_encoder.tree_def, valid_data.y_dict[0], en_normalize_word)

        reconstructed = cvae.sample([tree])
        word_values = reconstructed[0].leaves()
        words_idx = cvae._det_decoder.tree_def.leaves_types[0].value_type.representation_to_abstract_batch(tf.stack(list(map(lambda x: x.representation, word_values))))

        words = [all_words[w] if w in all_words.keys() else '#OOV#' for w in words_idx]
        print ("\t", " ".join(words))

if __name__ == "__main__":
    define_flags()

    config = tf.ConfigProto()
    tf.enable_eager_execution(config=config)
    tf.contrib.eager.run()