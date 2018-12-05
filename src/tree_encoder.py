import tensorflow as tf
from tree.definition import TreeDefinition, Tree, NodeDefinition
from tree.batch import BatchOfTreesForEncoding
import typing as T


class GatedFixedArityNodeEmbedder(tf.keras.Model):
    """ Build a dense 2-layer which is optimized for left-0-padded input """

    def __init__(self, _no_positional=None, activation=None, hidden_coef: float= None, embedding_size: int = None,
                 arity: int = None,
                 **kwargs):
        super(GatedFixedArityNodeEmbedder, self).__init__(**kwargs)

        if _no_positional != None:
            raise ValueError("Positional argument not allowed!")

        self.activation = activation
        self.hidden_coef = hidden_coef
        self.embedding_size = embedding_size
        self.arity = arity



    def build(self, input_shape):

        # self.gating_f.build(input_shape)
        self.gating_f = tf.keras.Sequential([
            # tf.keras.layers.Dense(units=int(input_shape[1].value * self.hidden_coef), activation=tf.sigmoid),
            tf.keras.layers.Dense(units=1 + self.arity, activation=tf.sigmoid)])

        self.output_f = tf.keras.Sequential([
            tf.keras.layers.Dense(min(int((input_shape[1].value + self.embedding_size) * self.hidden_coef), self.embedding_size),
                                  activation=self.activation, name='/1'),
            tf.keras.layers.Dense(self.embedding_size, activation=self.activation, name='/2')
        ])

        # self.output_f.build(input_shape)

        super(GatedFixedArityNodeEmbedder, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        """

        :param x: zero padded input [batch,  <= input_size]
        :return: [clones, batch, output_size]
        """
        output = self.output_f(x)  # [batch, emb]
        childrens = tf.reshape(x, [x.shape[0], self.arity, -1])  # [batch, arity, children_emb]
        gatings = tf.expand_dims(tf.nn.softmax(self.gating_f(tf.concat([x, output], axis=-1)), axis=-1), axis=-1)
        corrected = tf.concat([childrens, tf.expand_dims(output, axis=1)], axis=1) * gatings
        return tf.reduce_sum(corrected, axis=1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


class GatedValueMerger(tf.keras.Model):
    # TODO dunno why it doesn't work
    def __init__(self, _no_positional=None, activation=None, hidden_coef: float= None, embedding_size: int = None,
                 **kwargs):
        super(GatedValueMerger, self).__init__(**kwargs)
        if _no_positional != None:
            raise ValueError("Positional argument not allowed!")

        self.activation = activation
        self.hidden_coef = hidden_coef
        self.embedding_size = embedding_size


    # def build(self, input_shape):
    #     # self.gating_f = tf.keras.Sequential([
    #     #     # tf.keras.layers.Dense(units=int(input_shape[1].value * self.hidden_coef), activation=tf.sigmoid),
    #     #     tf.keras.layers.Dense(units=1 , activation=tf.sigmoid)])
    #     #
    #     # self.output_f = tf.keras.Sequential([
    #     #     tf.keras.layers.Dense(int((input_shape[0].value + self.embedding_size) * self.hidden_coef),
    #     #                           activation=self.activation, input_shape=input_shape, name='/1'),
    #     #     tf.keras.layers.Dense(self.embedding_size, activation=self.activation, name='/2')
    #     # ])
    #     super(GatedValueMerger, self).build(input_shape)

    # def call(self, x, *args, **kwargs):
    #     """
    #
    #     :param x: zero padded input [batch,  <= input_size]
    #     :return: [clones, batch, output_size]
    #     """
    #     # output = self.output_f(x)  # [batch, emb]
    #     # gatings = tf.nn.softmax(self.gating_f(tf.concat([x, output], axis=-1)), axis=-1)
    #     # corrected = output * gatings + (1 - gatings) * x[:, :self.embedding_size]
    #     # return corrected
    #     return tf.zeros([x.shape[0],self.embedding_size])

    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], self.embedding_size)


class NullableInputDenseLayer(tf.keras.layers.Layer):
    """ Build a dense layer which is optimized for left-0-padded input """

    def __init__(self, _no_positional=None, input_size: int = None,
                 hidden_activation=None, hidden_size: int = None,
                 **kwargs):
        super(NullableInputDenseLayer, self).__init__(**kwargs)

        if _no_positional != None:
            raise ValueError("Positional argument not allowed!")

        self.hidden_activation = hidden_activation
        self.hidden_size = hidden_size
        self.input_size = input_size


    def build(self, input_shape):

        self.hidden_kernel = self.add_weight(name='hidden_kernel',
                                             shape=[self.input_size, self.hidden_size],
                                             initializer='random_normal',
                                             trainable=True)

        self.hidden_bias= self.add_weight(name='hidden_bias',
                                          shape=(1, self.hidden_size),    # 1 is for broadcasting
                                          initializer='random_normal',
                                          trainable=True)

        super(NullableInputDenseLayer, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        """

        :param x: zero padded input [batch,  <= input_size, emb]
        :return: [clones, batch, output_size]
        """
        n = x.shape[1].value

        hidden_activation = self.hidden_activation(tf.matmul(x, self.hidden_kernel[:n]) + self.hidden_bias)

        return hidden_activation

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.hidden_size)


class GatedNullableInput(tf.keras.Model):

    def __init__(self, _no_positional=None, embedding_size:int=None, output_model_builder: tf.keras.Model = None, input_size: int = None, **kwargs):

        if _no_positional != None:
            raise ValueError("Positional argument not allowed!")

        self.embedding_size = embedding_size
        self.output_model_builder = output_model_builder
        self.input_size = input_size

        super(GatedNullableInput, self).__init__(**kwargs)

    def build(self, input_shape):

        self.gating_f = NullableInputDenseLayer(input_size=self.input_size + self.embedding_size, hidden_activation=tf.nn.leaky_relu, hidden_size=self.input_size // self.embedding_size + 1)
        self.output_model = self.output_model_builder()
        # self.gating_f = tf.keras.layers.Dense(units=input_shape[1].value // self.embedding_size + 1, activation=tf.sigmoid, name="gate")

        super(GatedNullableInput, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        """

        :param x: zero padded input [batch,  <= input_size * emb]
        :return: [clones, batch, output_size]
        """

        output = self.output_model(x)
        number_of_input = x.shape[1].value // self.embedding_size
        gating_inp = tf.concat([output, x], axis=-1)
        gatings = tf.nn.softmax(self.gating_f(gating_inp)[:, :number_of_input+1], axis=1)
        weighted = tf.reshape(gating_inp, [x.shape[0], number_of_input + 1, -1]) * tf.expand_dims(gatings, -1)
        return tf.reduce_sum(weighted, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.embedding_size)


class EncoderCellsBuilder:
    """ Define interfaces and simple implementations for a cells builder, factory used for the encoder modules.
        - Cells: merge children embeddings into parent embedding
        - Embedder: project leaves value into embedding space
        - Merger: merge node value and node embedding into embedding space.
    """

    def __init__(self,
                 cell_builder: T.Callable[[NodeDefinition, "Encoder", T.Union[str, None]], tf.keras.Model],
                 embedder_builder: T.Callable[[NodeDefinition, int, T.Union[str, None]], tf.keras.Model],
                 merger_builder: T.Callable[[NodeDefinition, int, T.Union[str, None]], tf.keras.Model]):
        """Simple implementation which just use callables, avoiding superfluous inheritance

        :param cell_builder: see CellsBuilder.build_cell doc
        :param embedder_builder: see CellsBuilder.build_embedder doc
        :param merger_builder: see CellsBuilder.build_merger doc
        """
        self._cell_builder = cell_builder
        self._embedder_builder = embedder_builder
        self._merger_builder = merger_builder

    def build_cell(self, parent_def: NodeDefinition, encoder: "Encoder", name=None) -> tf.keras.Model:
        """A cell is something that merge children embeddings into the parent embedding.
        Actual input shape shape depends from parent arity.
            - Fixed arity: input size = total children * embedding_size
            - Variable arity: input_size = 2 * embedding_size """

        m = self._cell_builder(parent_def, encoder, name)
        if type(m) == tuple:
            for mi in m:
                # setattr(mi, 'compiled_call', tf.contrib.eager.defun(mi))   # it's actually slower
                setattr(mi, 'optimized_call', mi.__call__)
        else:
            # setattr(m, 'compiled_call', tf.contrib.eager.defun(m))   # it's actually slower
            setattr(m, 'optimized_call', m.__call__)

        return m

    def build_embedder(self, leaf_def: NodeDefinition, embedding_size: int, name=None):
        """An embedder is something that projects leaf value into the embedding space"""
        m = self._embedder_builder(leaf_def, embedding_size, name)
        # setattr(m, 'optimized_call', tf.contrib.eager.defun(m))   # it's actually slower
        setattr(m, 'optimized_call', m)
        return m

    def build_merger(self, node_def: NodeDefinition, embedding_size: int, name=None):
        """A merger is something that take a node value, its embedding and merge them into a single embedding"""
        m = self._merger_builder(node_def, embedding_size, name)
        # setattr(m, 'optimized_call', tf.contrib.eager.defun(m))   # it's actually slower
        setattr(m, 'optimized_call', m)
        return m

    @staticmethod
    def simple_categorical_embedder_builder(hidden_coef: int, activation=tf.nn.leaky_relu):
        def f(leaf_def: NodeDefinition, embedding_size, name=None):
            return tf.keras.Sequential([
                tf.keras.layers.Dense(int(embedding_size * hidden_coef),
                                      activation=activation,
                                      input_shape=(leaf_def.value_type.representation_shape,),
                                      name=name+'/1'),
                tf.keras.layers.Dense(embedding_size,
                                      activation=activation,
                                      name=name+"/2")
            ])
        return f

    @staticmethod
    def simple_dense_embedder_builder(activation=tf.nn.leaky_relu):
        def f(leaf_def: NodeDefinition, embedding_size, name=None):

            return tf.keras.layers.Dense(embedding_size,
                                         activation=activation,
                                         input_shape=(leaf_def.value_type.representation_shape,),
                                         name=name)
        return f

    @staticmethod
    def simple_categorical_merger_builder(hidden_coef, activation=tf.nn.leaky_relu):
        """
        """
        def f(node_def: NodeDefinition, embedding_size, name=None):
            input_size = embedding_size + node_def.value_type.representation_shape
            # return GatedValueMerger(activation=activation, embedding_size=embedding_size, hidden_coef=hidden_coef, name=name)
            return tf.keras.Sequential([
                tf.keras.layers.Dense(int((input_size + embedding_size) * hidden_coef),
                                      activation=activation,
                                      input_shape=(input_size,),
                                      name=name + '/1'),
                tf.keras.layers.Dense(embedding_size,
                                      activation=activation,
                                      name=name + "/2")])
        return f

    @staticmethod
    def simple_cell_builder(hidden_coef, activation=tf.nn.leaky_relu, gate=True):
        def f(node_def: NodeDefinition, encoder: 'Encoder', name=None):
            if type(node_def.arity) == node_def.VariableArity:
                if not encoder.use_flat_strategy:
                    # TODO use rnn/lstm
                    input_shape = (encoder.embedding_size*2,)
                    if gate:
                        return GatedFixedArityNodeEmbedder(activation=activation, hidden_coef=hidden_coef, embedding_size=encoder.embedding_size, arity=2)
                else:
                    output_model_builder = lambda :tf.keras.Sequential([
                        NullableInputDenseLayer(input_size=encoder.embedding_size * (encoder.cut_arity + 1),  # 1 is the summarization of extra children
                                                hidden_activation=activation, hidden_size=encoder.embedding_size * int(encoder.cut_arity**hidden_coef)),
                        tf.keras.layers.Dense(encoder.embedding_size, activation=activation)
                    ], name=name)

                    return GatedNullableInput(output_model_builder=output_model_builder,
                                              input_size=encoder.embedding_size * (encoder.cut_arity + 1),
                                              embedding_size=encoder.embedding_size,
                                              name=name) if gate else output_model_builder(),\
                            tf.keras.Sequential([
                                # tf.keras.layers.Reshape([encoder.max_arity - encoder.cut_arity, encoder.embedding_size]),
                                tf.keras.layers.Dense(int(encoder.embedding_size * hidden_coef), activation=activation, input_shape=(encoder.embedding_size,),
                                                      name=name + '/extra_attention/1'),
                                tf.keras.layers.Dense(1, name=name + '/extra_attention/2')
                            ])
                    # input_shape = (encoder.embedding_size * encoder.max_arity, )
            elif type(node_def.arity) == node_def.FixedArity:
                if gate:
                    return GatedFixedArityNodeEmbedder(activation=activation, hidden_coef=hidden_coef, embedding_size=encoder.embedding_size, arity=node_def.arity.value)
                else:
                    return tf.keras.Sequential([
                        tf.keras.layers.Dense(int((encoder.embedding_size) * hidden_coef),
                                              activation=activation, name='/1'),
                        tf.keras.layers.Dense(encoder.embedding_size, activation=activation, name='/2')
                    ])

            return tf.keras.Sequential([
                tf.keras.layers.Dense(int((input_shape[0] + encoder.embedding_size) * hidden_coef), activation=activation, input_shape=input_shape, name=name+'/1'),
                tf.keras.layers.Dense(encoder.embedding_size, activation=activation, name=name+'/2')
            ])
        return f


class Encoder(tf.keras.Model):
    def __init__(self, _no_positional = None,
                 tree_def: TreeDefinition = None, embedding_size: int = None, cellsbuilder: EncoderCellsBuilder = None, cut_arity: int = None, max_arity = None, name='',
                 variable_arity_strategy="FLAT"
                 ):
        """

        :param tree_def:
        :param embedding_size:
        """
        super(Encoder, self).__init__()

        self.tree_def = tree_def
        self.node_map = {n.id: n for n in tree_def.node_types}

        self.use_flat_strategy = variable_arity_strategy == "FLAT"
        self.max_arity = max_arity
        self.cut_arity = cut_arity

        self.embedding_size = embedding_size

        # if not attr, they don't get registered as variable by the keras model (dunno why)
        for t in tree_def.node_types:
            if self.use_flat_strategy and type(t.arity) == t.VariableArity:
                c, e = cellsbuilder.build_cell(t, self, name=name+"C_" + t.id)
                setattr(self, 'C_'+t.id, c)
                setattr(self, 'C_extra_' + t.id, e)

            elif not (type(t.arity) == t.FixedArity and t.arity.value == 0):
                setattr(self, 'C_'+t.id, cellsbuilder.build_cell(t, self, name=name+"C_" + t.id))

            if t.value_type is not None:
                setattr(self, 'M_' + t.id, cellsbuilder.build_merger(t, embedding_size, name=name+"M_" + t.id))

        for l in tree_def.leaves_types:
            setattr(self, 'E_'+l.id, cellsbuilder.build_embedder(l, embedding_size, name=name+"E_" + l.id))


    def _c_fixed_op(self, inp, node_id, ops, network):

        res = network.optimized_call(inp)

        if self.node_map[node_id].value_type is None:

            ops[0].meta['emb_batch'].scatter_update('embs', [op.meta['node_numb'] for op in ops], res)
            for op in ops:
                op.meta['computed'] = True

        # compute merging with value straight on - input already in place
        else:
            values = self.node_map[node_id].value_type.abstract_to_representation_batch([x.value.abstract_value for x in ops])
            self._m_op(res, values, getattr(self, 'M_'+node_id), ops)

        # recursive fused call is ignored - not useful in our use cases

    def _m_op(self, embs, values, network, ops):

            inp = tf.concat([embs, values], axis=-1)
            res = network.optimized_call(inp)

            ops[0].meta['emb_batch'].scatter_update('embs', [op.meta['node_numb'] for op in ops], res)

            for op in ops:
                op.meta['computed'] = True

    def __call__(self, batch: BatchOfTreesForEncoding):
        all_ops = {}    # (op_type, node_id) -> [inputs]

        # 1) visit the trees, store leaves computation and create back-links to traverse the tree bottom-up
        def init(node, **kwargs):

            node.meta['computed'] = False

            if 'added' in node.meta.keys():
                del node.meta['added']

            # leaves are always computable with no dependencies
            if len(node.children) == 0:

                if ('E', node.node_type_id) not in all_ops:
                    all_ops[('E', node.node_type_id)] = []

                # store operation
                all_ops[('E', node.node_type_id)].append(node)

            else:
                # children recursion and back-links
                for c in node.children:
                    c.meta['parent'] = node

        batch.map_to_all_nodes(init, batch.trees)

        while len(all_ops) > 0:
            # 2) compute the aggregated most-required computation

            op_t, node_id = max(all_ops.keys(), key=lambda k: len(all_ops[k]))
            ops = all_ops.pop((op_t, node_id))
            if len(ops) == 0:
                break

            node_t = self.node_map[node_id]
            network = getattr(self, op_t + '_' + node_t.id)

            if op_t == 'E':

                inp = node_t.value_type.abstract_to_representation_batch([x.value.abstract_value for x in ops])
                res = network.optimized_call(inp)

                #  superflous when node is fused
                if node_id not in self.tree_def.fusable_nodes_id_child_parent.keys():
                    batch.scatter_update('embs',
                                      [op.meta['node_numb'] for op in ops],
                                      res)

                    for op in ops:
                        op.meta['computed'] = True

                else:
                    rec_ops = [o.meta['parent'] for o in ops]
                    network = getattr(self, 'C_'+ops[0].meta['parent'].node_type_id)
                    self._c_fixed_op(res, ops[0].meta['parent'].node_type_id, rec_ops, network)

            elif op_t == 'M':
                    values = node_t.value_type.abstract_to_representation_batch([x.value.abstract_value for x in ops])
                    embs = tf.gather(batch['embs'], [x.meta['node_numb'] for x in ops])
                    self._m_op(embs, values, network, ops)

            elif op_t == 'C':
                if type(node_t.arity) == NodeDefinition.FixedArity:
                    inp = tf.gather(batch['embs'], [c.meta['node_numb'] for op in ops for c in op.children])
                    inp_1 = tf.reshape(inp, [len(ops), -1])
                    self._c_fixed_op(inp_1, node_id, ops, network)

                elif type(node_t.arity) == NodeDefinition.VariableArity and not self.use_flat_strategy:
                    # TODO rnn (e.g. lstm) ?

                    idx = tf.cast(tf.reshape([[o.meta['node_numb'], o.children[o.meta.get('next_child', 0)].meta['node_numb']]
                                              for o in ops], [-1]), tf.int64)

                    inp = tf.gather(batch['embs'], idx)

                    inp = tf.reshape(inp, [len(ops), -1])
                    res = network.optimized_call(inp)

                    k = ('C', node_id)
                    if k not in all_ops.keys():
                        all_ops[k] = []

                    # add results
                    batch.scatter_update('embs',
                                      [op.meta['node_numb'] for op in ops],
                                      res)

                    for o in ops:

                        if o.meta.get('next_child', 0) == len(o.children) - 1:  # merged the last children
                            if self.node_map[node_id].value_type is not None:
                                all_ops[k].append(o)
                            else:
                                o.meta['computed'] = True
                        else:   # keep computing
                            o.meta['next_child'] = o.meta.get('next_child', 0) + 1
                            o.meta.pop('added')

                            # keep computing if other child is ready
                            if o.children[o.meta['next_child']].meta['computed']:
                                if ('C', o.node_type_id) not in all_ops:
                                    all_ops[('C', o.node_type_id)] = []

                                all_ops[('C', o.node_type_id)].append(o)
                                # avoid computing multiple times the parent when multiple children have been computed ad once
                                o.meta['added'] = None
                elif type(node_t.arity) == NodeDefinition.VariableArity and self.use_flat_strategy:
                    # TODO warning if max_arity < actual_arity (the code won't work anyway - crash)

                    max_children_arity = max([len(o.children) for o in ops])
                    first_arity = min(max_children_arity, self.cut_arity)
                    EXTRA_CHILD = max_children_arity > self.cut_arity

                    # extra children
                    if EXTRA_CHILD:
                        extra_network = getattr(self, op_t + '_extra_' + node_t.id)
                        # TODO we can use less zero padding
                        extra_inp = tf.gather(batch['embs'],tf.reshape(tf.convert_to_tensor([[c.meta['node_numb'] for c in o.children[self.cut_arity:]] + ([0] * (max_children_arity - len(o.children))) for o in ops if len(o.children) > self.cut_arity]), [-1]))
                        weights = extra_network(extra_inp)
                        weights = tf.reshape(tf.nn.softmax(tf.reshape(weights, [-1, max_children_arity - self.cut_arity])), [-1, max_children_arity - self.cut_arity, 1])
                        extra_inp = tf.reshape(extra_inp, [-1, max_children_arity - self.cut_arity, self.embedding_size])
                        weighted = tf.reduce_sum(weights * extra_inp, axis=1)
                        indeces = [[i] for o, i in zip(ops, range(len(ops))) if len(o.children) > self.cut_arity]
                        extra_children = tf.scatter_nd(indeces, weighted, [len(ops), self.embedding_size])
                    else:
                        extra_children = tf.zeros([len(ops), self.embedding_size])

                    # first children
                    inp = tf.gather(batch['embs'], tf.reshape(tf.convert_to_tensor([[c.meta['node_numb'] for c in o.children[:self.cut_arity]] + ([0] * (first_arity - len(o.children))) for o in ops]), [-1]))
                    inp = tf.reshape(inp, [len(ops), first_arity * self.embedding_size])
                    inp = tf.concat([inp, extra_children], axis=1)
                    res = network.optimized_call(inp)

                    if node_t.value_type is None:
                        batch.scatter_update('embs',
                                          [op.meta['node_numb'] for op in ops],
                                          res)
                        for o in ops:
                            o.meta['computed'] = True

                    else:
                            m_net = getattr(self, 'M_' + node_id)
                            inp = node_t.value_type.abstract_to_representation_batch([x.value.abstract_value for x in ops])
                            self._m_op(res, inp, m_net, ops)
            else:
                raise NotImplementedError()

            # 3) find new ready to go operations
            # res should be: [number_of_ops x output_size], with order preserved as in all_ops
            for op in ops:

                # use back-link to find ready-to-go operations
                if "parent" in op.meta.keys():  # otherwise is the root
                    parent = op.meta['parent']

                    # keep bubbling up if we already computed the parent (might be some fused op)
                    while parent.meta['computed'] and 'parent' in parent.meta.keys():
                        parent = parent.meta['parent']
                    if parent.meta['computed']:
                        continue    # we reached the root and it's computed, we're done for this tree

                    if ('added' not in parent.meta.keys() and
                        # For fixed arity node we need all children to be computed
                        (((type(self.node_map[parent.node_type_id].arity) == NodeDefinition.FixedArity or self.use_flat_strategy)
                            and all(map(lambda s: s.meta['computed'], parent.children)))
                        or
                        # For variable arity node we just need the next node
                        (type(self.node_map[parent.node_type_id].arity) == NodeDefinition.VariableArity
                            and not self.use_flat_strategy
                            and parent.children[parent.meta.get('next_child', 0)].meta['computed']))):

                        if ('C', parent.node_type_id) not in all_ops:
                            all_ops[('C', parent.node_type_id)] = []

                        all_ops[('C', parent.node_type_id)].append(parent)

                        # avoid computing multiple times the parent when multiple children have been computed ad once
                        parent.meta['added'] = None

        return tf.gather(batch['embs'], [t.meta['node_numb'] for t in batch.trees])

