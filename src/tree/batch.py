import tensorflow as tf
import tensorflow.contrib.eager as tfe
import typing as T
from tree.definition import Tree, TreeDefinition, NodeDefinition
from functools import reduce

class BatchOfTrees:
    def __init__(self):
        self.stores = {}

        # count items (i.e. nodes/values) to index their values in a big tensor
        self.counters = {
            'embs': 1   # 0 by convention is always the tf.zeros tensor
        }

    def __getitem__(self, item):
        return self.stores[item]

    def __setitem__(self, key, value):
        self.stores[key] = value

    def scatter_update(self, store_key, indeces, values):
        self[store_key] = \
            tf.scatter_nd(tf.reshape(indeces, [-1, 1]), values, self[store_key].shape) + \
            (tf.scatter_nd(tf.reshape(indeces, [-1, 1]), tf.ones([values.shape[0], 1]), (self[store_key].shape[0], 1)) - 1) * -self[store_key]

        # tf.scatter_update(self[store_key], indeces, values)

    @staticmethod
    def map_to_all_nodes(map_f: T.Callable[[Tree], None], trees: T.List[Tree]):
        def visit(n, depth=0):
            map_f(n, depth=depth)
            list(map(lambda x: visit(x, depth=depth+1), n.children))

        list(map(visit, trees))


class BatchOfTreesForEncoding(BatchOfTrees):
    def __init__(self, trees: T.List[Tree], embedding_size: int):
        super(self.__class__, self).__init__()

        self.trees = trees

        def init(n: Tree, **kwargs):

            nonlocal self
            n.meta['emb_batch'] = self
            n.meta['node_numb'] = self.counters['embs']
            self.counters['embs'] += 1

        self.map_to_all_nodes(init, trees)

        self['embs'] = tf.zeros([self.counters['embs'] + 1, embedding_size])

    def get_all_embeggings(self):
        def visit(t):
            return [t.meta['node_numb']] + list(reduce(lambda x,y: x+y, map(visit, t.children), []))
        embs_per_batch = []
        for t in self.trees:
            idx = visit(t)
            embs = tf.gather(self['embs'], idx)
            embs_per_batch.append(embs)

        return embs_per_batch

class BatchOfTreesForDecoding(BatchOfTrees):
    def __init__(self, root_embeddings: tf.Tensor, tree_def: TreeDefinition, target_trees: T.List[Tree] = None):

        super(self.__class__, self).__init__()

        self.is_supervised = target_trees is not None

        self.root_embeddings = root_embeddings
        self.target_trees = target_trees
        self.decoded_trees = []     # populated by the decoder
        self.tree_def = tree_def

        self.deferred_value_types = {}
        for nt in tree_def.node_types:
            if nt.value_type is not None:
                self.counters['vals_' + nt.id] = 1
                self.deferred_value_types[nt.id] = self.BuildDeferredRepresentationValueType(nt)

        # supervised call
        if self.is_supervised:
            self.depths = {'embs': []}

            def init(n: Tree, depth=0):
                nonlocal self
                n.meta['dec_batch'] = self
                n.meta['node_numb'] = self.counters['embs']
                self.counters['embs'] += 1
                self.depths['embs'].append(depth)

                if n.value is not None:
                    k = 'vals_' + n.node_type_id
                    n.meta['value_numb'] = self.counters[k]

                    if k not in self.depths.keys():
                        self.depths[k] = []
                    self.depths[k].append(depth)

                    self.counters[k] += 1

            self.map_to_all_nodes(init, target_trees)

            self['embs'] = tf.zeros([self.counters['embs'] + 1, root_embeddings.shape[1]])

            # save the initial embeddings in the store se we can easily gather them afterwards
            self.scatter_update('embs', [t.meta['node_numb'] for t in target_trees], root_embeddings)

            _depths = {k: tf.convert_to_tensor(self.depths[k], dtype=tf.float32) for k in self.depths.keys()}
            self.depths = _depths
            self.max_depths = {k: tf.reduce_max(self.depths[k]) for k in self.depths.keys()}
            scale_start, scale_end, scale_exp, max_depth = 1.0, 0.0, 2.0, 70
            self.depth_scale_coefs = {
                k: scale_start + (scale_end - scale_start) * (self.depths[k] / max_depth) ** scale_exp for k in self.depths.keys()
            }
            # don't need to index them, never gather nor rewrite them  - only used altoghether for the loss.
            # thus incrementally stack-up into a constant
            # root distrib is saved with some zero padding
            # more over some are not associated to a real node, artificial nodes used to train the model
            # to -not- generate a node (special no-child/no-node)

            self.distribs = []
            self.distribs_gt = []

            for nt in tree_def.node_types:
                if nt.value_type is not None:
                    self['vals_'+nt.id] = tf.zeros([self.counters['vals_'+nt.id] + 1,nt.value_type.representation_shape])

        # unsupervised call
        else:
            self['embs'] = tf.zeros([1, root_embeddings.shape[1]])

            for nt in tree_def.node_types:
                if nt.value_type is not None:
                    self['vals_'+nt.id] = tf.zeros([1, nt.value_type.representation_shape])

    def add_rows(self, store_key, values):
        " Extend a store with new rows - assigning new indexes"

        n = values.shape[0].value
        new_indexes = list(range(self.counters[store_key], self.counters[store_key] + n))
        self.counters[store_key] += n

        self[store_key] = tf.concat([self[store_key], values], axis=0)

        return new_indexes

    def add_values(self, node_type: NodeDefinition, values):
        "Extend values building the DeferredValueType objects"
        new_indexes = self.add_rows('vals_'+node_type.id, values)
        return [
            self.deferred_value_types[node_type.id](index=i) for i in new_indexes
        ]

    def scatter_init_values(self, node_type: NodeDefinition, indeces, values):
        "Set initial values in a prebuilt tensor - return new DeferredValueType objects"
        self.scatter_update('vals_'+node_type.id, indeces, values)
        return [
            self.deferred_value_types[node_type.id](index=i) for i in indeces
        ]

    def stack_to(self, key, values):
        getattr(self, key).append(values)

    def get_stacked(self, key):
        return tf.concat(getattr(self, key), axis=0)

    def get_all_decoded_abstract_leaves(self):
        """ really slow """
        values = [t.leaves() for t in self.decoded_trees]

        # assuming all the leaves have the same type
        value_t = type(values[0][0])
        leaves = list(map(lambda x: tf.stack(list(map(lambda y: y.representation, x))), values))

        return list(map(lambda l: value_t.representation_to_abstract_batch(l), leaves[:4]))

    def compute_bleu_score(self, dictionary):
        pass

    def reconstruction_loss(self, ignore_values={}):

        value_gt = {nt.id: [] for nt in self.tree_def.node_types if nt.value_type is not None}
        value = {nt.id: [] for nt in self.tree_def.node_types if nt.value_type is not None}

        # first gather all the tensors and then compute the loss is ~3.5 time faster
        def gather(node):
            if node.value is not None:
                value[node.node_type_id].append(node.value.index)
                value_gt[node.node_type_id].append(node.tr_gt_value.abstract_value)
            list(map(gather, node.children))

        list(map(gather, self.decoded_trees))
        # looks like it doesn't really help to initially scale loss accordingly to depth in the tree, probably propagation
        # depth_scale_coefs = {
        #     k: scale_start + (scale_end - scale_start) * ((self.depths[k]) / float(max_depth)) ** (1/float(scale_exp)) for k in
        # self.depths.keys()
        # }

        sample_error = tf.square(self.get_stacked('distribs') - self.get_stacked('distribs_gt'))
        d_loss = tf.reduce_mean(tf.reduce_sum(sample_error, axis=1))

        v_loss = 0
        for k in value.keys():
            if len(value[k]) > 0 and k not in ignore_values:
                all_value_gt = self.tree_def.id_map[k].value_type.abstract_to_representation_batch(value_gt[k])
                all_value_gen = tf.gather(self['vals_'+k], value[k])
                v_loss += tf.reduce_mean(tf.reduce_mean(tf.square(all_value_gen - all_value_gt), axis=1))

        return d_loss, v_loss

    def BuildDeferredRepresentationValueType(batch, node_type: NodeDefinition):

        class DeferredRepresentationValueType(node_type.value_type):

            # doesn't directly store representations, it stores them in a contiguous tensor
            def __init__(self, abstract_value=None, representation=None, index=None):
                """
                Exactly one of the argument should be provided
                :param abstract_value: create a new index in the store and init it with the corresponding representation
                :param representation: create a new index in the store and init it with the representation
                :param index: use the indexed representation and stores there future changes
                """

                if len(list(filter(None, [abstract_value, representation, index]))) != 1:
                    raise ValueError('Expected exactly one initial value')

                if index is not None:
                    self.index = index
                    self._abstract_value = None
                else:
                    self.index = batch.add_rows('vals_'+node_type.id, tf.zeros([1, node_type.value_type.representation_shape]))[0]
                    node_type.value_type.__init__(abstract_value=abstract_value, representation=representation)
                    del self._representation

            def gather_representation(self):
                return batch['vals_'+node_type.id][self.index]

            def update_representation(self, representation):
                tf.scatter_update(batch['vals_'+node_type.id], [self.index], representation)

            @staticmethod
            def representation_to_abstract_batch(t: tf.Tensor) -> T.List[T.Any]:
                return node_type.value_type.representation_to_abstract_batch(t)

            @staticmethod
            def abstract_to_representation_batch(v: T.List[T.Any]) -> tf.Tensor:
                return node_type.value_type.abstract_to_representation_batch(v)

            @property
            def abstract_value(self) -> T.Any:
                if self._abstract_value is None:
                    self._abstract_value = self.__class__.representation_to_abstract(self.representation)
                return self._abstract_value

            @abstract_value.setter
            def abstract_value(self, v: T.Any):
                self._abstract_value = v
                self.update_representation(self.abstract_to_representation(v))

            @property
            def representation(self) -> tf.Tensor:
                return self.gather_representation()

            @representation.setter
            def representation(self, t: tf.Tensor):
                self._abstract_value = None
                self.update_representation(t)

        return DeferredRepresentationValueType
