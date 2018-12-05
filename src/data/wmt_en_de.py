import typing as T

import numpy as np
from scipy.spatial import KDTree
import pickle

import tensorflow as tf
from pycorenlp import StanfordCoreNLP

# from stanfordcorenlp import StanfordCoreNLP

import multiprocessing as mpp
import re
import os
from tree.definition import Tree, TreeDefinition, NodeDefinition
import time
import nltk
import json
import itertools
# import pathos.multiprocessing as mpp
from functools import reduce

# TODO consider smarter normalization (umlaut/ß, acronyms). English: replace - and . with space. Number embedding ?!
# TODO -lrb- -rrb- -lcb- -rcb- not properly tokenized (appers as oov)


def de_normalize_word(s):
    return s.lower()


def en_normalize_word(s):
    return s.lower()


# things needed to multiprocesserly parse embeddings

def mp_init_worker(_word_normalizer, _selected_words):
    global word_normalizer, selected_words
    word_normalizer = _word_normalizer
    selected_words = _selected_words


def mp_parse_line(line):
    global word_normalizer
    global selected_words

    k, v = line.split(" ", 1)
    k = word_normalizer(k)
    if k in selected_words:
        return k, np.fromstring(v, dtype=np.float32, sep=' ')


def exploratory_analysis_and_dictionary_preparation():
    print("Reading top 50k words")
    with open('../../data/WMT14 English-German/vocab.50K.de.txt') as f:
        de50 = [de_normalize_word(l[:-1]) for l in f.readlines()]

    with open('../../data/WMT14 English-German/vocab.50K.en.txt') as f:
        en50 = [en_normalize_word(l[:-1]) for l in f.readlines()]

    # Build dictionary in multiprocess
    # probably it's no so effective, read by lines required to read every single characters.
    def mp_build_dict(embedding_file, _selected_words, _word_normalizer):
        with open(embedding_file) as f:
            pool = mp.Pool(initializer=mp_init_worker, initargs=(_word_normalizer, set(map(_word_normalizer, _selected_words))))

            labels, values = zip(*filter(lambda x: x is not None, pool.map(mp_parse_line, f)))
            return labels, values

    print("Building en dictionary")
    global enDict # dirty workaround workaround to make it 'pickletable'
    enDict = mp_build_dict('../../data/embeddings/wiki.en.vec', en50, en_normalize_word)
    print(len(set(map(en_normalize_word, en50))) - len(enDict[1]), "en oov")

    print("Storing en dictionary")
    with open('../../data/WMT14 English-German/en.50k.embedding.dict.pickle', 'wb') as f:
        pickle.dump(enDict, f)

    print("Building de dictionary")
    global deDict # dirty workaround to make it 'pickletable'
    deDict = mp_build_dict('../../data/embeddings/wiki.de.vec', de50, de_normalize_word)
    print(len(set(map(de_normalize_word, de50))) - len(deDict[1]), "de oov")

    print("Storing de dictionary")
    with open('../../data/WMT14 English-German/de.50k.embedding.dict.pickle', 'wb') as f:
        pickle.dump(deDict, f)

def load_dictionary(path):
    print("Loading dictionary ", path)
    with open(path, 'rb') as f:
        labels, values = pickle.load(f)
        # custom convention - 0 is oov embedding - all zero vector # TODO better handling
        return {l: i for l, i in zip(labels, itertools.count(1))}, tf.concat([tf.zeros([1, len(values[0])]), tf.convert_to_tensor(values)], axis=0)


def generate_trees_worker(t, retrying=0):
    try:
        global corenlp, corenlp_properties
        t = t.replace(' ##AT##-##AT## ', '-')

        a = [s['parse'] for s in corenlp.annotate(t, properties=corenlp_properties)['sentences']]
        return a
    except Exception as e:
        waits = [5,15,60]

        if retrying > 2:
            print(e)
            return None
        else:
            time.sleep(waits[retrying])
            return generate_trees_worker(t, retrying+1)


def generate_trees_worker_init(corenlp_server, lang):
    global corenlp, corenlp_properties

    parser ={
        'de': 'edu/stanford/nlp/models/srparser/germanSR.ser.gz',
        'en': 'edu/stanford/nlp/models/srparser/englishSR.ser.gz'
    }

    corenlp = StanfordCoreNLP(corenlp_server)
    corenlp_properties={
      'annotators': 'parse',
      'outputFormat': 'json',
      'parse.model': parser[lang]
    }


def generate_trees_mp(texts, corenlp_server, lang, processes=None):
    """

    run with (https://stackoverflow.com/a/42323459/7416971)
    export CLASSPATH="`find . -name '*.jar'`"
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
        with -serverProperties StanfordCoreNLP-german.properties for german

    return None elements on failure
    """
    if lang not in ['de', 'en']:
        raise ValueError('Not supported language')

    generate_trees_worker_init(corenlp_server, lang)
    return map(generate_trees_worker, texts)

    pool = mp.Pool(processes=min(mp.cpu_count(), processes),
                   initializer=generate_trees_worker_init,
                   initargs=(corenlp_server,lang))
    return pool.imap(generate_trees_worker, texts, chunksize=200)


def prepare_bin_embeddings(input_file, output_file, include_only=None):
    """ Read a embedding file, where each line is:
      tkn embedding and save it in a numpy format

    :param input_file:
    :param output_file: stored a picled dictionary mapping string to embeddings
    :param include_only: [str]. Limit only to those words (or a subset)
    """
    with open(input_file) as f:
      print("Reading file...")
      c = 0
      d = {}
      for l in f.readlines():
        c += 1
        tkns = l.split(" ")
        if include_only is None or tkns[0] in include_only:
          v = map(float, tkns[1:])
          d[tkns[0]] = np.array(v)
        if c % 500 == 0:
          print("Read "+str(c)+" lines")

    with open(output_file, 'wb') as of:
      pickle.dump(d, of)


def prepare_dataset(input_file, output_file, corenlp_server, lang, processes=None):
    """Parse a text file, each line a sample.
    Parse its constituency tree and store it in a file.

    Format is: each line a sample. Usual corenlp/nltk format
    TODO consider more efficient format (e.g. replacing words and category with index in the dictionary, store it in a binary format uin8,uint16). Or even better a custom variable length encoding
    """
    count = 0
    count_failed = 0
    count_failed_row = 0

    with open(output_file, 'w') as of:
        for samples in generate_trees_mp(open(input_file), corenlp_server, lang, processes):
            count += 1
            if samples is None:
                of.write("FAILED\n")
                count_failed_row+=1
                count_failed+=1
                print("Failed ", count_failed)
            else:
                count_failed_row=0
                clenaed = [ re.sub(r'  +', ' ', s.replace("\n", " ")) for s in samples] # one line without redundant characters
                of.write("%%NEWSENT%%".join(clenaed)+"\n")
            if count % 200 == 0 :
                print("Processed ", count, " lines (Failed ", count_failed, ")")
            if count_failed_row > 20:
                print("Something wrong, 20 request failed in a row")
                exit(1)


def prepare_labels_file(input_file_names, output_file):
    labels = {"ALL_SENT"}

    for f in input_file_names:
        c = 0
        for l in open(f):
            c+=1
            if c%10000 == 0:
                print(f, " ", c)

            for tstr in l.split("%%NEWSENT%%"):
                labels = labels.union(set(list(map(lambda x: x.split(' ')[0], tstr.split('(')))[1:]))

    with open(output_file, 'w') as f:
        json.dump(list(labels), f)


def translate_nltk_tree(tree: nltk.Tree, tree_def: TreeDefinition, label_map: T.Dict[str, int], normalizer: T.Callable[[str], str], ignore_leaves=False):
    if tree.height() > 2:
        return Tree(node_type_id="NODE",
                    children=list(map(lambda x: translate_nltk_tree(x, tree_def, label_map, normalizer, ignore_leaves), tree)),
                    value=tree_def.id_map["NODE"].value_type(abstract_value=tree.label()))
    else:
        normalized = normalizer(tree.leaves()[0])
        return Tree(node_type_id="PRE_LEAF",
                    children=[
                        Tree(
                            node_type_id="LEAF",
                            children=[],
                            value=tree_def.id_map["LEAF"].value_type(abstract_value=label_map.get(normalized,0))  # 0 is oov
                        )
                    ] if not ignore_leaves else [],
                    value=tree_def.id_map["PRE_LEAF"].value_type(abstract_value=tree.label()))

def parse_tree(tree_str: str, tree_def: TreeDefinition, label_map: T.Dict[str, int], normalizer: T.Callable[[str],str], ignore_leaves=False):
    """Construct a Tree from string representation"""
    try:
        sentences = tree_str.split("%%NEWSENT%%")
        trees = map(nltk.Tree.fromstring, sentences)

        return Tree(node_type_id="NODE",
                    children=list(map(lambda x: translate_nltk_tree(x, tree_def, label_map, normalizer, ignore_leaves), trees)),
                    value=tree_def.id_map["NODE"].value_type(abstract_value="ALL_SENT"))
    except:
        return None

def visit(t):
    if t.height() > 2:
        children_arities = reduce(lambda x,y: x+y, list(map(visit, t)))
        return [len(t)] + children_arities
    else:
        return [1]

def _worker(str_tree):
    sentences = str_tree.split("%%NEWSENT%%")
    if all(map(lambda x: x != 'FAILED\n', sentences)):
        trees = list(map(nltk.Tree.fromstring, sentences))
        # for t in trees:
        #     for l in t.leaves():
        #         ln = normalizer(l)
        #         if ln not in dictionary:
        #             oov[ln] = oov.get(ln, 0) + 1

        depth = max(list(map(lambda x: x.height(), trees))) + 1
        node_count = len(str_tree.replace("(","").replace(')', "").split(' '))
        leaves_count = sum(list(map(lambda x: len(x.leaves()), trees)))
        arities = reduce(lambda x, y: x+y, map(visit, trees), [])
        return depth, node_count, leaves_count, arities
    else:
        return None

def get_stat_from_tree_file(file_name, dictionary, normalizer):

    with open(file_name) as f:
        pool = mpp.Pool()
        res = pool.map(_worker, f.readlines(), chunksize=10000)
        depths, node_counts, leaves_counts, arities = zip(*(filter(None, res)))
        arities_ = []
        for a in arities:
            arities_.extend(a)
        arities = arities_
        print(min(depths), np.mean(depths), np.std(depths), max(depths))
        print(min(node_counts), np.mean(node_counts), np.std(node_counts), max(node_counts))
        print(min(leaves_counts), np.mean(leaves_counts), np.std(leaves_counts), max(leaves_counts))
        print(min(arities), np.mean(arities), np.std(arities), max(arities))

def build_tree_def(labels, all_word_embeddings, ignore_leaves=False):
    kdtree = KDTree(all_word_embeddings)

    class LabelValue(NodeDefinition.Value):

        label_map = dict(zip(labels, itertools.count(0)))   # 0 is oov
        representation_shape = len(labels)

        @staticmethod
        def abstract_to_representation_batch(vs: T.Any) -> tf.Tensor:
            return tf.one_hot([LabelValue.label_map[v] for v in vs], len(labels))

        @staticmethod
        def representation_to_abstract_batch(t: tf.Tensor) -> T.Any:
            return [labels[i] for i in tf.argmax(t, axis=1).numpy()]

    class WordValue(NodeDefinition.Value):

        representation_shape = all_word_embeddings[0].shape[0].value

        @staticmethod
        def representation_to_abstract_batch(t: tf.Tensor) -> T.List[T.Any]:
            return kdtree.query(t.numpy())[1].tolist()

        @staticmethod
        def abstract_to_representation_batch(vs: T.List[T.Any]) -> tf.Tensor:
            return tf.gather(all_word_embeddings, vs)

        @property
        def representation(self) -> tf.Tensor:
            if self._representation is None and self._abstract_value is not None:
                self._representation = self.__class__.abstract_to_representation(self._abstract_value)
            return self._representation

        @representation.setter
        def representation(self, t: tf.Tensor):
            self._abstract_value = None
            self._representation = t

    return TreeDefinition(
        node_types=[
            NodeDefinition("NODE", may_root=True, arity=NodeDefinition.VariableArity(), value_type=LabelValue),
            NodeDefinition("PRE_LEAF", may_root=False, arity=NodeDefinition.FixedArity(1 if not ignore_leaves else 0), value_type=LabelValue),
            NodeDefinition("LEAF", may_root=False, arity=NodeDefinition.FixedArity(0), value_type=WordValue)
        ],
        fusable_nodes_id=[("PRE_LEAF", "LEAF")])  # LEAF - PRE_LEAF are always linked 1-1

class Dataset:
    def __init__(self, directory, x_file, y_file, x_tree_def, x_dict, y_tree_def, y_dict, x_normalizer, y_normalizer,
            batch_size, max_arity, max_depth, max_node_count):
        self.directory = directory
        self.x_file = x_file
        self.y_file = y_file
        self.x_tree_def = x_tree_def
        self.x_dict = x_dict
        self.y_tree_def = y_tree_def
        self.y_dict = y_dict
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        self.batch_size = batch_size
        self.max_arity = max_arity
        self.max_depth = max_depth
        self.max_node_count = max_node_count
        self.skipped_count = 0

    def iter(self, initial_skip=0, ignore_leaves=False):

        # TODO shuffle
        self.skipped_count = 0

        def _data_iter():
            count_initial_skip = 0
            while True:  # keep iterating over it
                with open(os.path.join(self.directory, self.x_file), encoding='utf-8') as xlines:
                    with open(os.path.join(self.directory, self.y_file), encoding='utf-8') as ylines:
                        for x, y in zip(map(lambda x: parse_tree(x, self.x_tree_def, self.x_dict[0], self.x_normalizer, ignore_leaves), itertools.islice(xlines, initial_skip, None)),
                                        map(lambda y: parse_tree(y, self.y_tree_def, self.y_dict[0], self.y_normalizer, ignore_leaves), itertools.islice(ylines, initial_skip, None))):
                            if x is not None and y is not None:
                                if x.calculate_max_arity() < self.max_arity and y.calculate_max_arity() < self.max_arity and \
                                        x.calculate_max_depth() < self.max_depth and y.calculate_max_depth() < self.max_depth and \
                                        x.calculate_node_count() < self.max_node_count and y.calculate_node_count() < self.max_node_count:

                                    yield (x, y)
                                else:
                                    self.skipped_count += 1

        data_iter = _data_iter()
        while True:
            xs, ys = zip(*[next(data_iter) for _ in range(self.batch_size)])
            yield list(xs), list(ys)

def dataset(directory, x_file, y_file, x_tree_def, x_dict, y_tree_def, y_dict, x_normalizer, y_normalizer,
            batch_size, max_arity, max_depth, max_node_count, skip_first=0):
    """

    :param directory: root directory with all the files
    :param x_file: name of the file containing line by line the sentence in a language
    :param y_file: name of the file containing line by line the corresponding sentence in the other language
    :param x_dict:
    :param y_dict:

    """

    # TODO shuffle

    def _data_iter():
        count_skipped = 0
        count_initial_skip = 0
        while True: # keep iterating over it
            with open(os.path.join(directory, x_file), encoding='utf-8') as xlines:
                with open(os.path.join(directory, y_file), encoding='utf-8') as ylines:
                    for x, y in zip(map(lambda x: parse_tree(x, x_tree_def, x_dict[0], x_normalizer), xlines), map(lambda y: parse_tree(y, y_tree_def, y_dict[0], y_normalizer), ylines)):
                        if x is not None and y is not None:
                            if x.calculate_max_arity() < max_arity and y.calculate_max_arity() < max_arity and\
                                x.calculate_max_depth() < max_depth and y.calculate_max_depth() < max_depth and\
                                x.calculate_node_count() < max_node_count and y.calculate_node_count() < max_node_count:

                                if count_initial_skip < skip_first:
                                    count_initial_skip += 1
                                    continue

                                yield (x, y)
                            else:
                                count_skipped += 1
                                print("skipped ", count_skipped, " samples")

    data_iter = _data_iter()
    while True:
        xs, ys = zip(*[next(data_iter) for _ in range(batch_size)])
        yield list(xs), list(ys)

#
# from data.wmt_tree_pb2 import WMT_Tree, NodeType
#
# # /CONVERT DATASET
# def build_pb_sample(sample: Tree, LabelValueClass):
#     t = WMT_Tree()
#
#     t.node_type = NodeType.Value(sample.node_type_id)
#
#     if sample.node_type_id == 'LEAF':
#         t.value = sample.value.abstract_value
#     else:
#         t.value = LabelValueClass.label_map[sample.value.abstract_value]
#
#     t.children.MergeFrom([build_pb_sample(c, LabelValueClass) for c in sample.children])
#
#     return t
#
# def write_delimited(open_file, sample: WMT_Tree):
#     size = sample.ByteSize()
#     open_file.write(int.to_bytes(size, 4, 'big'))
#     open_file.write(sample.SerializeToString())
#
# def read_all_delimited(file_name):
#     with open(file_name, 'rb') as f:
#         size = int.from_bytes(f.read(4), 'big')
#         encoded_sample = f.read(size)
#         if len(encoded_sample) == size:
#             obj = WMT_Tree()
#             obj.ParseFromString(encoded_sample)
#             yield obj

#
# def prepare_all(data_dir, lang_a, lang_b, emb_files_patterns, dataset_files_pattern):
#     """Main function that prepare every file needed during training.
#         The only files this function assume to exists are:
#         - List of sentences in language A (one sent. per line)
#         - List of sentences in language B (one sent. per line)
#         - Embedding file in language A (word emb per line)
#         - Embedding file in language B (word emb per line)
#     """
#
#     # emb_a = os.path.join(data_dir, emb_files_patterns.replace("{{LANG}}", lang_a))
#     # if not os.path.exists(emb_a):
#     #     pass
