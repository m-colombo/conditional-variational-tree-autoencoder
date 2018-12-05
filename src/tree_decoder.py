import tensorflow as tf
from tree.definition import TreeDefinition, Tree, TrainingTree, NodeDefinition
from tree.batch import BatchOfTreesForDecoding
import typing as T
from utils import tf_random_choice_idx
from benchmark import Measure
import itertools


class GatedFixedArityNodeDecoder(tf.keras.Model):
    """ Build a dense 2-layer which is optimized for left-0-padded input """

    def __init__(self, _no_positional=None, activation=None, hidden_coef: float= None, embedding_size: int = None,
                 arity: int = None,
                 **kwargs):

        if _no_positional != None:
            raise ValueError("Positional argument not allowed!")

        self.activation = activation
        self.hidden_coef = hidden_coef
        self.embedding_size = embedding_size
        self.arity = arity
        super(GatedFixedArityNodeDecoder, self).__init__(**kwargs)

    def build(self, input_shape):

        self.gating_f = tf.keras.Sequential([
            # tf.keras.layers.Dense(units=int(input_shape[1].value * self.hidden_coef), activation=tf.sigmoid),
                                             tf.keras.layers.Dense(units=1, activation=tf.sigmoid)])
        self.output_f = tf.keras.Sequential([
            tf.keras.layers.Dense(int((input_shape[1].value + self.embedding_size * self.arity) * self.hidden_coef),
                                  activation=self.activation, input_shape=input_shape),
            tf.keras.layers.Dense(self.embedding_size * self.arity, activation=self.activation)
        ])
        super(GatedFixedArityNodeDecoder, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        """

        :param x: zero padded input [batch,  <= input_size]
        :return: [clones, batch, output_size]
        """
        parent_embs = x[:,:self.embedding_size] # TODO check - concat order is importand
        output = self.output_f(x)  # [batch, emb * arity]
        childrens = tf.reshape(output, [x.shape[0], self.arity, -1])  # [batch, arity, children_emb]
        gating_inp = tf.reshape(tf.concat([childrens, tf.tile(tf.expand_dims(x, axis=1), [1,self.arity, 1])], axis=-1), [x.shape[0] * self.arity, -1])
        gatings = self.gating_f(gating_inp)
        corrected = tf.reshape(childrens, [x.shape[0] * self.arity, -1]) * gatings + (1 - gatings) * tf.reshape(tf.tile(tf.expand_dims(parent_embs, axis=1), [1,self.arity, 1]), [x.shape[0] * self.arity, -1])
        return tf.reshape(corrected, [x.shape[0], -1])


class ParallelDense(tf.keras.layers.Layer):
    """ Build n dense (two layer) parallel (independent) layers """

    def __init__(self, activation, hidden_size: int, output_size: int, parallel_clones: int, gated: bool = False, **kwargs):
        self.activation = activation
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.parallel_clones = parallel_clones
        self.gated = gated

        super(ParallelDense, self).__init__(**kwargs)

    def build(self, input_shape):

        self.hidden_kernel = self.add_weight(name='hidden_kernel',
                                             shape=[self.parallel_clones, input_shape[1].value, self.hidden_size],
                                             initializer='random_normal',
                                             trainable=True)

        self.hidden_bias= self.add_weight(name='hidden_bias',
                                          shape=(self.parallel_clones, 1, self.hidden_size),    # 1 is for broadcasting
                                          initializer='random_normal',
                                          trainable=True)

        self.out_kernel = self.add_weight(name='out_kernel',
                                          shape=(self.parallel_clones, self.hidden_size, self.output_size),
                                          initializer='random_normal',
                                          trainable=True)

        self.out_bias= self.add_weight(name='out_bias',
                                             shape=(self.parallel_clones, 1, self.output_size), # 1 is for broadcasting
                                             initializer='random_normal',
                                             trainable=True)

        if self.gated:
            self.gate_kernel = self.add_weight(name='gate_kernel',
                                               shape=[ (self.parallel_clones * self.output_size) + input_shape[1].value, self.parallel_clones],
                                               initializer='random_normal',
                                               trainable=True)

            self.gate_bias = self.add_weight(name='gate_bias',
                                             shape=[self.parallel_clones],
                                             initializer='random_normal',
                                             trainable=True)

        super(ParallelDense, self).build(input_shape)

    def call(self, x, n:int):
        """

        :param x: input [batch, input_size]
        :param n: compute only the first n clones
        :return: [clones, batch, output_size]
        """
        x_ = tf.tile(tf.expand_dims(x, axis=0), [n, 1, 1])    # [clones, batch, input]
        #  [clones, batch, input] * [clones, input, hidden] = [clones, batch, hidden]
        hidden_activation = self.activation(tf.matmul(x_, self.hidden_kernel[:n]) + self.hidden_bias[:n])

        # [clones, batch, hidden] * [clones, hidden, output] = [clones, batch, output]
        output = self.activation(tf.matmul(hidden_activation, self.out_kernel[:n]) + self.out_bias[:n])

        if self.gated:
            gate_inp = tf.concat([x, tf.reshape(output, [x.shape[0], -1])], axis=-1)
            gate = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(gate_inp, self.gate_kernel[:gate_inp.shape[1], :n]) + self.gate_bias[:n]), axis=-1)
            gate = tf.reshape(gate, [n, -1, 1])
            corrected = tf.reshape(x, [1, x.shape[0], -1])[:, :, :self.output_size] * gate + (1-gate) * output
            return corrected

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)   # TODO this is wrong, can't be used in Sequential composition


class DecoderCellsBuilder:
    """ Define interfaces and simple implementations for a cells builder, factory used for the decoder modules.
           - Distrib : generates a distribution over nodes types
           - Value infl. : projects from the embedding space to some value space
           - Node infl. : from parent embedding generates children embeddings (actual shapes depends on node arity)
       """

    def __init__(self,
                 distrib_builder: T.Callable[[T.Tuple[int, int], T.Optional[str]], tf.keras.Model],
                 value_inflater_builder: T.Callable[[NodeDefinition, T.Optional[str]], tf.keras.Model],
                 node_inflater_builder: T.Callable[[NodeDefinition, "Decoder", T.Optional[str]], tf.keras.Model]):
        """Simple implementation which just use callables, avoiding superfluous inheritance

        :param distrib_builder: see self.build_distrib_cell
        :param value_inflater_builder: see self.build_value_inflater
        :param node_inflater_builder: see self.build_node_inflater
        """
        self._distrib_builder = distrib_builder
        self._value_inflater_builder = value_inflater_builder
        self._node_inflater_builder = node_inflater_builder

    def build_distrib_cell(self, output_size: (int, int), decoder: "Decoder", name=None) -> tf.keras.Model:
        """Build a distribution cell that given an embedding returns a output_size[0] concatenated probability vector of size output_size[1]"""
        m = self._distrib_builder(output_size, decoder, name)
        # setattr(m, 'compiled_call', tf.contrib.eager.defun(m))   # it's actually slower
        setattr(m, 'compiled_call', m.__call__)
        return m

    def build_value_inflater(self, node_def: NodeDefinition, decoder: "Decoder", name=None) -> tf.keras.Model:
        """Build a cell that projects an embedding in the node value space"""
        m = self._value_inflater_builder(node_def, decoder, name)
        # setattr(m, 'compiled_call', tf.contrib.eager.defun(m))   # it's actually slower
        setattr(m, 'compiled_call', m.__call__)
        return m

    def build_node_inflater(self, node_def: NodeDefinition, decoder: "Decoder", name=None) -> tf.keras.Model:
        """Build a cell that given parent embedding returns children embeddings.
            - for FixedArity nodes the output is the concat of all the chlidren embeddings
            - for VariableArity nodes it's a RNN - take (state_embedding) as input and returns (child_embedding, new_state_embedding)
        """
        m = self._node_inflater_builder(node_def, decoder, name)
        if type(m) == tuple:
            for mi in m:
                # setattr(mi, 'compiled_call', tf.contrib.eager.defun(mi))   # it's actually slower
                setattr(mi, 'compiled_call', mi.__call__)
        else:
            # setattr(m, 'compiled_call', tf.contrib.eager.defun(m))   # it's actually slower
            setattr(m, 'compiled_call', m.__call__)

        return m

    @staticmethod
    def simple_distrib_cell_builder(hidden_coef, activation=tf.nn.relu):
        def f(output_size: (int, int), decoder: Decoder, name=None):
            total_output_size = output_size[0] * output_size[1]

            size1 = int((total_output_size + decoder.embedding_size) * hidden_coef)
            size2 = int((size1 + decoder.embedding_size) * hidden_coef)

            return tf.keras.Sequential([
                tf.keras.layers.Dense(300, activation=activation),
                # tf.keras.layers.Dense(200, activation=activation),
                tf.keras.layers.Dense(output_size[0] * output_size[1]),
                tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1, output_size[1]]), output_shape=(output_size[1],)),
                tf.keras.layers.Softmax()
            ], name=name)
        return f

    @staticmethod
    def simple_node_inflater_builder(hidden_coef: float, activation=tf.nn.relu, gate=True):
        def f(node_def: NodeDefinition, decoder, name=None):
            if type(node_def.arity) == NodeDefinition.FixedArity:
                if gate:
                    return GatedFixedArityNodeDecoder(activation=activation, hidden_coef=hidden_coef,
                                                  embedding_size=decoder.embedding_size,
                                                  arity=node_def.arity.value, name=name)
                else:
                    return tf.keras.Sequential([
                                        tf.keras.layers.Dense(
                        int(hidden_coef * decoder.embedding_size * node_def.arity.value), activation=activation),
                                        tf.keras.layers.Dense(decoder.embedding_size * node_def.arity.value,
                                                               activation=activation),
                    ], name=name)
            elif type(node_def.arity) == NodeDefinition.VariableArity and not decoder.use_flat_strategy:
                return tf.keras.Sequential([
                    tf.keras.layers.Dense(int(decoder.embedding_size * 2 * hidden_coef), activation=activation),
                    tf.keras.layers.Dense(decoder.embedding_size * 2, activation=activation),
                ], name=name)
            elif type(node_def.arity) == NodeDefinition.VariableArity and decoder.use_flat_strategy:
                return ParallelDense(activation=activation, hidden_size=decoder.embedding_size,
                                     output_size=decoder.embedding_size,
                                     parallel_clones=decoder.cut_arity, gated=gate, name=name), \
                       tf.keras.Sequential([
                           tf.keras.layers.Dense(int(decoder.embedding_size * (1+hidden_coef*0.5)), activation=activation),
                           tf.keras.layers.Dense(decoder.embedding_size, activation=activation),
                       ], name=name+'_extra')
        return f

    @staticmethod
    def simple_1ofk_value_inflater_builder(hidden_coef, activation=tf.nn.relu):
        def f(node_def: NodeDefinition, decoder: "Decoder", name=None):
            size1 = int((node_def.value_type.representation_shape + decoder.embedding_size) * hidden_coef)
            size2 = int((size1 + decoder.embedding_size) * hidden_coef)
            return tf.keras.Sequential([
                tf.keras.layers.Dense(2*size1, activation=activation),
                # tf.keras.layers.Dense(size2, activation=activation),
                tf.keras.layers.Dense(node_def.value_type.representation_shape),
                tf.keras.layers.Softmax()
            ], name=name)
        return f

    @staticmethod
    def simple_dense_value_inflater_builder(hidden_coef, activation=tf.nn.relu):
        def f(node_def: NodeDefinition, decoder: "Decoder", name=None):
            size1 = int((node_def.value_type.representation_shape + decoder.embedding_size) * hidden_coef)
            size2 = int((size1 + decoder.embedding_size) * hidden_coef)
            return tf.keras.Sequential([
                tf.keras.layers.Dense(size1, activation=activation),
                tf.keras.layers.Dense(size2, activation=activation),
                tf.keras.layers.Dense(node_def.value_type.representation_shape),
            ], name=name)
        return f

    @staticmethod
    def node_map(map):
        def f(node_def, *args, **kwargs):
            return map[node_def.id](node_def, *args, **kwargs)
        return f


class Decoder(tf.keras.Model):

    def __init__(self, _no_positional_arguments=None,
                 tree_def: TreeDefinition = None, embedding_size: int = None,
                 max_depth: int = None, max_arity: int = None, cut_arity: int = None,
                 cellbuilder: DecoderCellsBuilder = None, max_node_count: int = 1000, take_root_along=True,
                 variable_arity_strategy="FLAT", attention=False):
        """

        :param tree_def:
        :param embedding_size:
        :param max_depth: after the tree generation is truncated
        :param max_arity: limit the max arity of the generated tree, must be more than any provided tree
        :param cellbuilder:
        :param max_node_count: limit the number of ndoe when unsupervised - avoid children nodes explosion
        :param take_root_along: whether to concat initial embedding at every step
        :param variable_arity_strategy: FLAT | REC .
        """
        super(Decoder, self).__init__()

        if _no_positional_arguments is not None:
            raise ValueError("Call with positional arguments is not allowed")

        self.cut_arity = cut_arity
        self.max_arity = max_arity
        self.max_depth = max_depth
        self.max_node_count = max_node_count

        self.embedding_size = embedding_size

        self.use_flat_strategy = variable_arity_strategy == "FLAT"
        self.attention = attention

        self.tree_def = tree_def
        self.node_map = {n.id: n for n in tree_def.node_types}

        self.root_types = self.tree_def.root_types
        self.root_types_idx = {t.id: i for t, i in zip(self.root_types, range(len(self.root_types)))}

        self.root_distrib = cellbuilder.build_distrib_cell((1, len(self.root_types)), self, name='distrib_root')

        self.all_types = self.tree_def.node_types
        self.all_types_idx = {t.id: i for t, i in zip(self.all_types, range(len(self.all_types)))}

        self.take_root_along = take_root_along

        if attention:
            self.attention_f = tf.keras.layers.Dense(1, activation=tf.sigmoid)

        # if not attr, they don't get registered as variable by the keras model (dunno why)
        for t in tree_def.node_types:
            if type(t.arity) == NodeDefinition.FixedArity and t.arity.value > 0:
                setattr(self, 'dist_' + t.id, cellbuilder.build_distrib_cell((t.arity.value, len(self.all_types)), self, name="distrib_" + t.id))
                setattr(self, 'infl_' + t.id, cellbuilder.build_node_inflater(t, self, name="inflater_" + t.id))
            elif type(t.arity) == NodeDefinition.VariableArity:
                if self.use_flat_strategy:
                    first_children, extra_children = cellbuilder.build_node_inflater(t, self, name="inflater_" + t.id)

                    # for first children, up to cut_arity, are computed by dedicated cells
                    setattr(self, 'dist_' + t.id, cellbuilder.build_distrib_cell((self.cut_arity, len(self.all_types) + 1), self, name="distrib_" + t.id))  # one special output for - nochild
                    setattr(self, 'infl_' + t.id, first_children)

                    # the same cell is used to compute the tail of the children, those who are rare
                    setattr(self, 'extra_dist_' + t.id, cellbuilder.build_distrib_cell((1, len(self.all_types) + 1), self, name="extra_distrib_" + t.id))  # one special output for - nochild
                    setattr(self, 'extra_infl_' + t.id, extra_children)
                else:
                    setattr(self, 'dist_'+t.id, cellbuilder.build_distrib_cell((1, len(self.all_types)+1),self,  name="distrib_" + t.id))  # one special output for - nochild
                    setattr(self, 'infl_' + t.id, cellbuilder.build_node_inflater(t, self, name="inflater_" + t.id))

            if t.value_type is not None:
                setattr(self, 'value_'+t.id, cellbuilder.build_value_inflater(t, self, name='value_'+t.id))

    times = {}
    def __call__(self, batch: BatchOfTreesForDecoding, augment_fn=None, attention_batch=None):
        """ Batched computation. All fireable operations are grouped according to node kinds and the one single aggregated
        operation is ran. Turned out to give a ~2x speedup.

        :param embeddings: [batch_size, embedding_size]
        :param target_trees: [batch_size] of trees or None if is not a supervised call
        :param augment_fn: given a list of embeddings and their index in the batch returns a list of extra info to use in the decoding ([n, embedding_size], [n]) -> [n, some_size]. For instance useful for some attention mechanism
        :return: [batch_size] of training trees
        """
        all_ops = {nt.id: [] for nt in self.all_types}
        TR = batch.target_trees is not None

        if self.attention:
            all_encoding_embedding = attention_batch.get_all_embeggings()

        # augment embeddings
        if augment_fn is not None:
            root_inp = augment_fn(batch.root_embeddings, range(batch.root_embeddings.shape[0].value))
        else:
            root_inp = batch.root_embeddings

        # root distribs
        distribs = self.root_distrib(root_inp)
        if TR:
            node_idx = [self.root_types_idx[t.node_type_id] for t in batch.target_trees]

            distribs_gt = tf.one_hot(node_idx, len(self.all_types)+1)
            vals = list(map(lambda t: t.value, batch.target_trees))

            pad = len(self.all_types) + 1 - len(self.root_types)

            batch.stack_to('distribs', tf.pad(distribs, [[0, 0], [0, pad]]))
            batch.stack_to('distribs_gt', distribs_gt)

        else:
            node_idx = list(tf.argmax(distribs, axis=1))
            node_count = [1] * len(node_idx)    # count total nodes per tree in order to bound their growth
            new_indexes = batch.add_rows('embs', batch.root_embeddings)

        # aggregate ops and plant the trees
        for n,  i in zip(node_idx, range(len(node_idx))):
            t = TrainingTree(self.root_types[n].id,
                             meta={"depth": 0,      # keep track of the tree depth to truncate at the given depth
                                   "batch_idx": i,  # keep track of batch index in order to be able to concat root embeddings
                                   })
            if TR:
                t.meta['target'] = batch.target_trees[i]
                t.meta['node_numb'] = batch.target_trees[i].meta['node_numb']
                t.tr_gt_value=vals[i]

                if len(batch.target_trees[i].children) > self.max_arity:
                    raise ValueError("Maximum Arity Exceeded " + str(len(batch.target_trees[i].children)) + ' > ' + str(self.max_arity))
            else:
                t.meta['node_numb'] = new_indexes[i] # initial embeddings are stored contiguosly

            batch.decoded_trees.append(t)
            all_ops[self.root_types[n].id].append(t)

        # aggregate computations cycle
        while True:

            # get the most requested op
            op_id = max(all_ops.keys(), key=lambda x: len(all_ops[x]))
            node_type = self.all_types[self.all_types_idx[op_id]]
            ops = all_ops.pop(op_id)
            all_ops[op_id] = []
            if len(ops) == 0:
                break

            with Measure('inp', Decoder.times):
                # build the input form the node embeddings
                inp = tf.gather(batch['embs'], [o.meta['node_numb'] for o in ops])
                batch_idxs = list(map(lambda x: x.meta['batch_idx'], ops))

                if self.attention:
                    all_new_inp = []
                    for i, o in zip(itertools.count(), ops):
                        b = all_encoding_embedding[
                            o.meta['batch_idx'] % len(all_encoding_embedding)]  # TODO really hardcoded
                        with_emb = tf.concat([b, inp[i:i+1]], axis=0)
                        concated = tf.concat([with_emb, tf.tile(tf.reshape(inp[i], [1, -1]), [with_emb.shape[0], 1])], axis=1)
                        gates = tf.nn.softmax(self.attention_f(concated), axis=0)
                        all_new_inp.append(tf.reduce_sum(concated * gates, axis=0))
                    inp = tf.stack(all_new_inp)

                # add to the input the 'augmented info'
                if augment_fn is not None:
                    inp = augment_fn(inp, batch_idxs)

                # add to the input  the root embedding
                if self.take_root_along:
                    root_embs = tf.gather(batch.root_embeddings, batch_idxs)
                    inp = tf.concat([inp, root_embs], axis=1)

            # add to the input informations about the node type TODO not sure this is needed
            # inp = tf.concat([inp, tf.tile(tf.one_hot(self.all_types_idx[node_type.id], len(self.all_types)), [len(ops), 1])])

            # compute node value, if present
            if node_type.value_type is not None:
                with Measure('val', Decoder.times):
                    # ops_to_compute_mask = tf.convert_to_tensor(list(map(lambda x: x.value is None, ops)))
                    # not avoid recomputing recursive nodes - looks to be more efficient

                    infl = getattr(self, 'value_'+node_type.id)
                    vals = infl.compiled_call(inp)

                    if TR:
                        new_values = batch.scatter_init_values(node_type, [o.meta['target'].meta['value_numb'] for o in ops], vals)
                    else:
                        new_values = batch.add_values(node_type, vals)

                    for o, v in zip(ops, new_values):
                        o.value = v

                    inp = tf.concat([inp, vals], axis=-1)

            # based on the node type expand the frontier of the computed tree
            if type(node_type.arity) == node_type.FixedArity:
                with Measure('fix', Decoder.times):
                    if node_type.arity.value == 0:  # leaf
                        for o in ops:
                            o.meta = {}     # value already computed we can release the memory
                    else:
                        # retrieve distribution and inflater
                        dst = getattr(self, 'dist_' + node_type.id)
                        inf = getattr(self, 'infl_' + node_type.id)

                        all_children_distribs = dst.compiled_call(inp)  # [batch * arity, types]
                        if TR:
                            node_idx = [self.all_types_idx[c.node_type_id] for o in ops for c in o.meta["target"].children]
                        else:
                            # node_idx = list(tf_random_choice_idx(all_children_distribs).numpy())
                            node_idx = list(tf.argmax(all_children_distribs, axis=1))

                        # adding info on the sampled children type before generating their embeddings
                        oh_distrib_ = tf.one_hot(node_idx, len(self.all_types)+1)
                        oh_distrib = tf.reshape(oh_distrib_, tf.cast([len(node_idx) / node_type.arity.value, -1], tf.int32))
                        inp = tf.concat([inp, oh_distrib], axis=-1)

                        all_embeddings = inf.compiled_call(inp) # [batch, arity * embedding_size]
                        all_embeddings = tf.reshape(all_embeddings, [-1, self.embedding_size])

                        if TR:
                            batch.scatter_update('embs', [c.meta['node_numb'] for o in ops for c in o.meta['target'].children], all_embeddings)
                            batch.stack_to('distribs', tf.pad(all_children_distribs, [[0, 0], [0, 1]]))
                            batch.stack_to('distribs_gt', oh_distrib_)
                        else:
                            new_indexes = batch.add_rows('embs', all_embeddings)

                        i = 0
                        j = 0
                        for o in ops:
                            for c in range(node_type.arity.value):
                                c_type = self.all_types[node_idx[i]]
                                t = TrainingTree(c_type.id, meta={"depth": o.meta["depth"] + 1, "batch_idx": o.meta["batch_idx"]})

                                if TR:
                                    t.meta['target'] = o.meta['target'].children[c]
                                    t.meta['node_numb'] = o.meta['target'].children[c].meta['node_numb']
                                    t.tr_gt_value = o.meta["target"].children[c].value

                                    if len(o.meta['target'].children[c].children) > self.max_arity:
                                        raise ValueError("Maximum Arity Exceeded " + str(
                                            len(o.meta['target'].children[c].children)) + ' > ' + str(self.max_arity))
                                else:
                                    t.meta['node_numb'] = new_indexes[i]

                                    if node_count[o.meta['batch_idx']] > self.max_node_count or o.meta["depth"] + 1 > self.max_depth:
                                        t.node_type_id = "TRUNCATED"
                                        # tf.logging.warn("Truncated tree. Node count of {0} [max {1}], Depth of {2} [max{3}]".format(node_count[o.meta['batch_idx']], self.max_node_count,o.meta["depth"] + 1, self.max_depth))
                                    else:
                                        node_count[o.meta['batch_idx']] += 1

                                # make the tree grows
                                # o.children.append(t)  makes the debugger crash :\
                                # RecursionError: maximum recursion depth exceeded while calling a Python object
                                # this other variant works :/
                                o.children = o.children + [t]

                                if t.node_type_id != "TRUNCATED":
                                    all_ops[t.node_type_id].append(t)

                                i += 1
                            j += 1
                            o.meta = {}  # release memory

            elif type(node_type.arity) == NodeDefinition.VariableArity and not self.use_flat_strategy:
                with Measure('var', Decoder.times):
                    dst = getattr(self, 'dist_' + node_type.id)
                    infl = getattr(self, 'infl_' + node_type.id)

                    distribs = dst.compiled_call(inp)
                    no_child_idx = len(self.all_types)  # special idx to stop the children generations

                    if TR:
                        node_idx = [
                            self.all_types_idx[o.meta['target'].children[len(o.children)].node_type_id]
                                if len(o.children) < len(o.meta['target'].children)
                                else no_child_idx
                            for o in ops]
                    else:
                        # node_idx = list(tf_random_choice_idx(distribs).numpy())
                        node_idx = list(tf.argmax(distribs, axis=1))

                    distribs_oh = tf.one_hot(node_idx, len(self.all_types) + 1)

                    if TR:
                        batch.stack_to('distribs', distribs)
                        batch.stack_to('distribs_gt', distribs_oh)

                    # avoid computing those exceeding maximum arity
                    no_child_mask = tf.equal(node_idx, no_child_idx)
                    truncated_mask = [n != no_child_idx and len(o.children) == self.max_arity for n, o in zip(node_idx, ops)]
                    mask = tf.logical_not(tf.logical_or(no_child_mask, truncated_mask))

                    inp = tf.boolean_mask(inp, mask, axis=0)   # remove terminated computations

                    if inp.shape[0].value > 0:  # otherwise means no more children have to be generated

                        inp = tf.concat([inp, tf.boolean_mask(distribs_oh, mask, axis=0)], axis=1)
                        embs = infl.compiled_call(inp)    # compute the new embedding

                        # perform all the split at once is much more efficient then having multiple slicing
                        child, new_parent = tf.split(embs, 2, axis=1)

                        # TODO some are actually not useful (those for which the computation is finished)
                        if TR:
                            batch.scatter_update('embs', [o.meta['target'].meta['node_numb'] for o, b in zip(ops, mask) if b], new_parent)
                            batch.scatter_update('embs', [o.meta['target'].children[len(o.children)].meta['node_numb'] for o, b in zip(ops, mask) if b ], child)
                        else:
                            batch.scatter_update('embs',
                                                 [o.meta['node_numb'] for o, b in zip(ops, mask) if b],
                                                 new_parent)

                            new_indeces=batch.add_rows('embs', child)

                    res_idx = 0
                    for o, i, n in zip(ops, node_idx, range(len(ops))):
                        if not mask[n]:
                            o.meta = {} # release memory TODO not sure
                        else:

                            c_idx = len(o.children)

                            t = TrainingTree(self.all_types[i].id,
                                             meta={"depth": o.meta["depth"] + 1,
                                                   "batch_idx": o.meta["batch_idx"]},
                                             tr_gt_value=o.meta["target"].children[c_idx].value if TR else None)

                            if TR:
                                t.meta['target'] = o.meta['target'].children[c_idx]
                                t.meta['node_numb'] = o.meta['target'].children[c_idx].meta['node_numb']

                                if len(o.meta['target'].children[c_idx].children) > self.max_arity:
                                    raise ValueError("Maximum Arity Exceeded " + str(
                                        len(o.meta['target'].children[c_idx].children)) + ' > ' + str(self.max_arity))
                            else:
                                t.meta['node_numb'] = new_indeces[res_idx]

                                if node_count[o.meta['batch_idx']] > self.max_node_count or o.meta[
                                    "depth"] + 1 > self.max_depth:
                                    t.node_type_id = "TRUNCATED"
                                    # tf.logging.warn("Truncated tree. Node count of {0} [max {1}], Depth of {2} [max{3}]".format(node_count[o.meta['batch_idx']], self.max_node_count, o.meta["depth"] + 1, self.max_depth))
                                else:
                                    node_count[o.meta['batch_idx']] += 1

                            o.children += [t]
                            if t.node_type_id != "TRUNCATED":
                                all_ops[t.node_type_id].append(t)   # add the new children computation
                            all_ops[o.node_type_id].append(o)   # continue the generation

                            res_idx += 1
            elif type(node_type.arity) == NodeDefinition.VariableArity and self.use_flat_strategy:
                with Measure('var', Decoder.times):
                    # TODO warning if max_arity < actual_arity
                    dst = getattr(self, 'dist_' + node_type.id)
                    infl = getattr(self, 'infl_' + node_type.id)

                    batch_size = inp.shape[0]

                    distribs = dst.compiled_call(inp)   # [batch * cut_arity, types+1] one special types means no child
                    no_child_idx = len(self.all_types)
                    max_children_arity = max([len(o.meta['target'].children) for o in ops]) if TR else self.max_arity
                    EXTRA_CHILD = max_children_arity > self.cut_arity

                    # if we have no info on children arity or there are some node with more children than our cut_arity
                    # we gonna compute remaining children with the 'jolly' cell
                    n = self.max_arity - self.cut_arity
                    if n > 0:
                        extra_dst = getattr(self, 'extra_dist_' + node_type.id)
                        extra_infl = getattr(self, 'extra_infl_' + node_type.id)



                        children_1ofk = tf.tile(tf.diag(tf.ones(n)), [batch_size, 1])
                        tiled_embs = tf.tile(tf.expand_dims(inp, axis=1), [1, n, 1])
                        extra_inp = tf.concat([tf.reshape(tiled_embs, [batch_size * n, -1]), children_1ofk], axis=1)
                        extra_distrib = extra_dst(extra_inp)    # [batch * n, types + 1]

                        if EXTRA_CHILD:
                            distribs = tf.reshape(
                                tf.concat([
                                    tf.reshape(distribs, [batch_size, self.cut_arity, len(self.all_types)+1]),
                                    tf.reshape(extra_distrib, [batch_size, n, len(self.all_types)+1])],
                                    axis=1),
                                [batch_size * self.max_arity, len(self.all_types)+1])
                        elif TR:
                            # train to not generate children
                            batch.stack_to('distribs', extra_distrib)
                            batch.stack_to('distribs_gt', tf.one_hot([no_child_idx] * extra_distrib.shape[0].value, len(self.all_types)+1))

                    if TR:
                        # assuming no empty interleaving children - all stacked to the left
                        node_idx = [self.all_types_idx[o.meta['target'].children[c].node_type_id] if len(o.meta['target'].children) > c
                                    else no_child_idx
                                    for o in ops for c in range(self.max_arity if EXTRA_CHILD else self.cut_arity)]
                    else:
                        # node_idx = list(tf_random_choice_idx(distribs).numpy())
                        node_idx = list(tf.argmax(distribs, axis=1).numpy())

                    distrib_gt = tf.one_hot(node_idx, len(self.all_types)+1)

                    if TR:
                        batch.stack_to('distribs', distribs)
                        batch.stack_to('distribs_gt', distrib_gt)

                    # TODO check all the one_hot, are not differentiable!!

                    distrib_gt = tf.reshape(distrib_gt, [batch_size, self.max_arity if EXTRA_CHILD else self.cut_arity, len(self.all_types)+1])

                    first_distribs_gt = tf.reshape(distrib_gt[:, :self.cut_arity], [batch_size, -1])
                    first_inp = tf.concat([inp, first_distribs_gt], axis=1)
                    first_embs = infl.compiled_call(first_inp, min(max_children_arity, self.cut_arity))    #[arity, batch, embedding_size]
                    first_embs = tf.reshape(tf.transpose(first_embs, [1, 0, 2]), [-1, self.embedding_size]) # [batch * max_children_arity, embedding_size]

                    if TR:
                        n = min(max_children_arity, self.cut_arity)
                        children_indexes = tf.concat([
                            tf.range(i * n, i * n + min(len(o.meta['target'].children), self.cut_arity))
                            for i, o in zip(itertools.count(), ops)], axis=0)
                        first_embs = tf.gather(first_embs, children_indexes)
                        # retrieve the index where to store the results
                        children_node_numbs = [c.meta['node_numb'] for o in ops for c in o.meta['target'].children[:self.cut_arity]]
                        batch.scatter_update('embs', children_node_numbs, first_embs)
                    else:
                        # we're assuming children are born left to right
                        idx = tf.reshape(tf.reshape(node_idx, [-1, self.max_arity if EXTRA_CHILD else self.cut_arity])[:, :min(max_children_arity, self.cut_arity)], [-1])
                        children_indexes = [i for ni, i in zip(idx.numpy().tolist(), itertools.count()) if ni != no_child_idx]

                        if len(children_indexes) > 0:
                            first_embs = tf.gather(first_embs, children_indexes)
                            first_new_node_numb = batch.add_rows('embs', first_embs)

                    if EXTRA_CHILD:
                        extra_distribs_gt = tf.reshape(distrib_gt[:, self.cut_arity:], [batch_size * (self.max_arity - self.cut_arity), (len(self.all_types) + 1)])
                        extra_inp = tf.concat([extra_inp, extra_distribs_gt], axis=1)
                        extra_embs = extra_infl(extra_inp)  # [batch * (max_arity - cut_arity), embedding_size]

                        if TR:
                            n = self.max_arity - self.cut_arity
                            children_indexes = tf.concat([
                                tf.range(i * n, i * n + len(o.meta['target'].children) - self.cut_arity)
                                for i, o in zip(itertools.count(), ops) if len(o.meta['target'].children) > self.cut_arity], axis=0)
                            extra_embs = tf.gather(extra_embs, children_indexes)
                            extra_node_numbs = [c.meta['node_numb'] for o in ops for c in o.meta['target'].children[self.cut_arity:]]
                            batch.scatter_update('embs', extra_node_numbs, extra_embs)
                        else:
                            # we're assuming children are born left to right
                            idx = tf.reshape(tf.reshape(node_idx, [-1, self.max_arity ])[:,self.cut_arity:], [-1])
                            children_indexes = [i for ni, i in zip(idx.numpy().tolist(), itertools.count()) if
                                                ni != no_child_idx]

                            if len(children_indexes) > 0:
                                extra_embs = tf.gather(extra_embs, children_indexes)
                                extra_new_node_numb = batch.add_rows('embs', extra_embs)

                    first_count = 0
                    extra_count = 0
                    stride = self.max_arity if EXTRA_CHILD else self.cut_arity
                    for o, i in zip(ops, range(len(ops))):

                        for c in range(stride): # TODO cycle only over generated children
                            n = i * stride + c
                            if node_idx[n] != no_child_idx:

                                t = TrainingTree(self.all_types[node_idx[n]].id,
                                                 meta={"depth": o.meta["depth"] + 1,
                                                       "batch_idx": o.meta["batch_idx"]})

                                if TR:
                                    t.meta['target'] = o.meta['target'].children[c]
                                    t.meta['node_numb'] = o.meta['target'].children[c].meta['node_numb']
                                    t.tr_gt_value = o.meta["target"].children[c].value

                                    if len(o.meta['target'].children[c].children) > self.max_arity:
                                        raise ValueError("Maximum Arity Exceeded " + str(
                                            len(o.meta['target'].children[c].children)) + ' > ' + str(self.max_arity))
                                else:
                                    if c >= self.cut_arity:
                                        t.meta['node_numb'] = extra_new_node_numb[extra_count]
                                        extra_count += 1
                                    else:
                                        t.meta['node_numb'] = first_new_node_numb[first_count]
                                        first_count += 1

                                    if node_count[o.meta['batch_idx']] > self.max_node_count or o.meta["depth"] + 1 > self.max_depth :
                                        t.node_type_id = "TRUNCATED"
                                    else:
                                        node_count[o.meta['batch_idx']] += 1

                                if t.node_type_id != "TRUNCATED":
                                    all_ops[t.node_type_id].append(t)

                                o.children += [t]
            else:
                raise ValueError("Node arity type not handled")

        return batch.decoded_trees
