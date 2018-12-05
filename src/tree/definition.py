import typing as T
import tensorflow as tf
from functools import reduce
from collections import namedtuple

class Tree:
    """Concrete tree instances"""
    def __init__(self, node_type_id: str, children: T.List["Tree"] = None, value=None, meta=None):
        """

        :param node_type_id:
        :param children:
        :param value:
        """
        self.node_type_id = node_type_id

        self.children = children if children is not None else []

        self.value = value
        self.meta = meta if meta is not None else {}  # just a place to store additional info

    def __str__(self):
        s = self.node_type_id

        if self.value is not None:
            s += ' ' + str(self.value.abstract_value)

        if len(self.children) > 0:
            s += " : [" + ", ".join(map(str, self.children)) + "]"

        return s


    def leaves(self):
        if len(self.children) == 0:
            return [self.value]
        else:
            return reduce(lambda x,y: x + y, map(Tree.leaves, self.children))

    def calculate_max_depth(self):
        return 1 + max(map(Tree.calculate_max_depth, self.children), default=0)

    def calculate_node_count(self):
        return 1 + sum(map(Tree.calculate_node_count, self.children))

    def calculate_max_arity(self):
        return max(len(self.children), max(map(Tree.calculate_max_arity, self.children), default=0))

    def calculate_mean_arity(self):
        def _visit(t):
            if t.children:
                return [len(t.children)] + list(reduce(lambda x,y: x+y, map(_visit, t.children), []))
            else:
                return []

        arities = _visit(self)
        return sum(arities)/len(arities) if arities else None

    def equiv(self, t2):
        return self.structural_equivalence(t2) and self.values_equivalence(t2)

    def compute_overlaps(self, t2, also_values=False, skip_leaves_value=False):
        t1_nodes, t2_nodes = 0, 0
        bcpt_nodes = 0
        all_value_count = 0
        matched_value_count = 0

        def visit(t1, t2):
            nonlocal t1_nodes, t2_nodes, bcpt_nodes, all_value_count, matched_value_count

            t1_nodes += len(t1.children)
            t2_nodes += len(t2.children)
            for c1, c2 in zip(t1.children, t2.children):
                if c1.node_type_id == c2.node_type_id:
                    bcpt_nodes += 1
                    visit(c1, c2)
                    if also_values and c1.value is not None and not (len(c1.children) == 0 and skip_leaves_value):
                        all_value_count += 1
                        if c1.value.abstract_value == c2.value.abstract_value:  # TODO need to be implemented by the abstract value
                            matched_value_count += 1

        if self.node_type_id == t2.node_type_id:
            bcpt_nodes = 1
            t1_nodes = 1
            t2_nodes = 1
            if also_values and self.value is not None:
                all_value_count += 1
                if self.value.abstract_value == t2.value.abstract_value:    # TODO need to be implemented by the abstract value
                    matched_value_count += 1
            visit(self, t2)
            s_acc = 2 * bcpt_nodes / float(t1_nodes + t2_nodes)
            if also_values:
                v_acc = s_acc * matched_value_count / float(all_value_count) if all_value_count > 0 else 1.0 * s_acc
            else:
                v_acc = None
            return s_acc, v_acc
        else:
            return 0, 0

    def structural_overlap(self, t2):
        """:return [0,1], [0,1]. percentage of matching structure (starting from the root) respectively computed on nodes and tree depth (of t1, it isn't commutative)"""
        # TODO really inefficient - implement using BatchForDecoding?
        # TODO implement correctly as in the thesis
        max_depth = self.calculate_max_depth()
        node_count = self.calculate_node_count()

        def visit(t1, t2):
            if t1.node_type_id == t2.node_type_id:
                if len(t1.children) > 0 and len(t2.children) > 0:
                    depths, counts = zip(*map(lambda c: visit(c[0], c[1]), zip(t1.children, t2.children)))
                    return 1 + max(depths), 1 + sum(counts)
                else:
                    return 1, 1
            else:
                return 0, 0

        depth, count = visit(self, t2)
        return depth / float(max_depth), count / float(node_count)

    def structural_equivalence(self, t2):
        return self.node_type_id == t2.node_type_id and\
               len(self.children) == len(t2.children) and \
               all(map(lambda x: x[0].structural_equivalence(x[1]), zip(self.children, t2.children)))

    def values_overlap(self, t2, ignore_leaves=False):
        """Assumes structural matching"""
        def visit(t1, t2):
            if t1.value is not None and t2.value is not None and (not ignore_leaves or len(t1.children) > 0):
                value_match = int(t1.value.abstract_value == t2.value.abstract_value)
                if len(self.children) > 0 and len(t2.children) > 0:
                    matching_values, values_count = map(sum, zip(*map(lambda c: visit(c[0], c[1]),zip(t1.children, t2.children))))
                else:
                    matching_values, values_count = 0, 0
                return value_match + matching_values, values_count + 1
            else:
                return 0, 0
        mv, av = visit(self, t2)
        return mv/float(av)

    def values_equivalence(self, t2):

        if (self.value is None and t2.value is None) or \
                self.value.abstract_value == t2.value.abstract_value:   # TODO not really safe nor general
            return all(map(lambda ts: ts[0].values_equivalence(ts[1]), zip(self.children, t2.children)))
        else:
            return False

    def clone(self, clone_value=False):
        return Tree(node_type_id=self.node_type_id,
                    children=list(map(Tree.clone, self.children)),
                    meta=self.meta.copy(),
                    value=type(self.value)(abstract_value=self.value.abstract_value) if clone_value else self.value)

    TreeComparisonInfo = namedtuple('TreesComparisonInfo', [
        "matching_struct",
        "matching_value",
        "struct_overlap_by_depth",
        "struct_overlap_by_node",
        "value_overlap"])

    @staticmethod
    def compare_trees(ts1: T.List["Tree"], ts2: T.List["Tree"]):
        # TODO consider using batch for decoding for more efficient comparison
        s_ok, v_ok = 0, 0
        s_overlaps = []
        v_overlaps = []
        for t1, t2 in zip(ts1, ts2):

            if t1.structural_equivalence(t2):
                s_ok += 1
                s_overlaps.append((1.0, 1.0))

                if t1.values_equivalence(t2):   # TODO will be deadly slow when they match!
                    v_ok += 1
                    v_overlaps.append(1.0)
                else:
                    v_overlaps.append(0)    # TODO deadly slow t1.values_overlap(t2)
            else:
                v_overlaps.append(0.0)
                s_overlaps.append(t1.structural_overlap(t2))

        avg_depth_overlap, avg_node_overlap = list(map(lambda x: sum(x) / float(len(x)), zip(*s_overlaps)))

        return Tree.TreeComparisonInfo(
            matching_struct=s_ok / float(len(ts1)),
            matching_value=v_ok / float(len(ts1)),
            struct_overlap_by_depth=avg_depth_overlap,
            struct_overlap_by_node=avg_node_overlap,
            value_overlap=sum(v_overlaps) / float(len(v_overlaps))
        )

class TrainingTree(Tree): # TODO remove class, no more needed. move tr_gt_value in Tree.meta
    def __init__(self, node_type_id: str, children: T.List["TrainingTree"] = None, meta=None,
                value=None, tr_gt_value = None):

        super(TrainingTree, self).__init__(node_type_id, children, value, meta)

        self.tr_gt_value = tr_gt_value


class NodeDefinition:

    class Value:
        representation_shape = None

        def __init__(self, abstract_value=None, representation=None):
            """

            :param abstract_value: actual value
            :param representation: tensor representation e.g. 1ofk representation
            """
            if abstract_value == representation or (abstract_value is not None and representation is not None):
                raise ValueError("Initialize with Value XOR Representation")

            self._abstract_value = None
            self._representation = None

            if abstract_value is not None:
                self.abstract_value = abstract_value

            if representation is not None:
                self.representation = tf.convert_to_tensor(representation)

        @classmethod
        def representation_to_abstract(cls, t: tf.Tensor) -> T.Any:
            return cls.representation_to_abstract_batch(tf.expand_dims(t, axis=0))[0]

        @classmethod
        def abstract_to_representation(cls, v: T.Any) -> tf.Tensor:
            return cls.abstract_to_representation_batch([v])[0]

        @staticmethod
        def representation_to_abstract_batch(t: tf.Tensor) -> T.List[T.Any]:
            raise NotImplementedError()

        @staticmethod
        def abstract_to_representation_batch(v: T.List[T.Any]) -> tf.Tensor:
            raise NotImplementedError()

        @property
        def abstract_value(self) -> T.Any:
            if self._abstract_value is None and self._representation is not None:
                self._abstract_value = self.__class__.representation_to_abstract(self._representation)
            return self._abstract_value

        @abstract_value.setter
        def abstract_value(self, v: T.Any):
            self._abstract_value = v
            self._representation = None

        @property
        def representation(self) -> tf.Tensor:
            if self._representation is None and self._abstract_value is not None:
                self._representation = self.__class__.abstract_to_representation(self._abstract_value)
            return self._representation

        @representation.setter
        def representation(self, t: tf.Tensor):
            self._abstract_value = None
            self._representation = t

    @staticmethod
    def NumValue(min_value, max_value):
        """Build a Value class that handle numbers in [min_value, max_value] encoded as 1ofk"""

        size = max_value - min_value + 1

        class NumValue(NodeDefinition.Value):
            representation_shape = size

            @staticmethod
            def representation_to_abstract_batch(t: tf.Tensor):
                return (tf.argmax(t, axis=-1) + min_value).numpy()

            @staticmethod
            def abstract_to_representation_batch(v: T.List[T.Any]):
                return tf.one_hot(list(map(lambda x: x- min_value, v)), size, axis=-1)

        return NumValue

    class Arity:
        pass

    class FixedArity(Arity):
        def __init__(self, value: int):
            if value < 0:
                raise ValueError()

            self.value = value

    class VariableArity(Arity):
        def __init__(self, min_value: int = 1, max_value=None):

            if (min_value is not None and min_value < 0) or (max_value is not None and max_value < 0):
                raise ValueError()

            if min_value is not None and max_value is not None and min_value > max_value:
                raise ValueError()

            self.min_value = min_value
            self.max_value = max_value

    def __init__(self, type_id: str, may_root: bool, arity: Arity, value_type: T.Type[Value] = None):
        self.id = type_id
        self.may_root = may_root
        self.arity = arity
        self.value_type = value_type

    def __str__(self):
        return self.id


class TreeDefinition:
    def __init__(self, node_types: T.List[NodeDefinition], fusable_nodes_id=None):
        """

        :param node_types:
        :param leaf_types:
         :param fusable_nodes_id: list of (parent node_type_id, child node_type_id) that can be fused - i.e. every time a node of that type is computed
        we can compute its parent straight on and viceversa. Only makes sense when the fusable has fixed size and the parent has one only child
        """
        self.node_types = node_types
        self.id_map = {n.id: n for n in node_types}

        self.fusable_nodes_id_parent_child = {} if fusable_nodes_id is None else {a: b for a,b in fusable_nodes_id}
        self.fusable_nodes_id_child_parent = {} if fusable_nodes_id is None else {b: a for a, b in fusable_nodes_id}

        if len(node_types) != len(set(map(lambda x: x.id, node_types))):
            raise ValueError("Types id must be unique!")

        self.root_types = list(filter(lambda nt: nt.may_root, node_types))
        if len(self.root_types) == 0:
            raise ValueError("Need at least a root node type")

        self.leaves_types = list(filter(lambda nt: (type(nt.arity) == NodeDefinition.FixedArity and nt.arity.value == 0) or (type(nt.arity) == NodeDefinition.VariableArity and nt.arity.min_value == 0), node_types))