import random
from tree.definition import TreeDefinition, NodeDefinition, Tree
import tensorflow as tf
import typing as T


class BinaryExpressionTreeGen:
    def __init__(self, min_value, max_value):

        if min_value < 0 or min_value >= max_value:
            raise ValueError()

        self.NumValue = NodeDefinition.NumValue(min_value, max_value)

        self.tree_def = TreeDefinition(
            node_types=[
                NodeDefinition("add_bin", may_root=True, arity=NodeDefinition.FixedArity(2)),
                NodeDefinition("sub_bin", may_root=True, arity=NodeDefinition.FixedArity(2)),
                NodeDefinition("num_value", may_root=True, arity=NodeDefinition.FixedArity(0), value_type=self.NumValue)

            ])

        self.one_step_derived_tree_def = TreeDefinition(
            node_types=[
                NodeDefinition("add_bin", may_root=True, arity=NodeDefinition.FixedArity(2)),
                NodeDefinition("sub_bin", may_root=True, arity=NodeDefinition.FixedArity(2)),
                NodeDefinition("num_value", may_root=True, arity=NodeDefinition.FixedArity(0), value_type=NodeDefinition.NumValue(min_value - max_value, max_value * 2))
            ]
        )

        self.leaf_values = list(range(min_value, max_value+1))
        self.node_types = self.tree_def.node_types

    def generate(self, max_depth, avoid_leaf=False):
        """Generate a random arithmetic expression tree,
            using just binary plus and minus
            Args:
                max_depth: integer > 0
            Returns:
                expression tree where leaves are int.
        """

        if max_depth == 1: # recursion base case
            v = random.sample(self.leaf_values, 1)[0]
            return Tree(node_type_id='num_value', value=self.NumValue(abstract_value=v))

        elif max_depth > 1:
            if avoid_leaf:
                types = self.node_types[:-1]
            else:
                types = self.node_types
            node_type = random.sample(types, 1)[0]

            if node_type.id == 'num_value':
                return self.generate(1)

            else:
                return Tree(node_type.id, children=[
                    self.generate(max_depth - 1),
                    self.generate(max_depth - 1)], value=None)

    def evaluate(self, et):
        """Evaluate the result of the arithmetic expression
            Args:
                et: expression tree
            Returns:
                an integer, the result
        """

        if et.node_type_id == 'num_value':
            return et.value.abstract_value
        elif et.node_type_id == 'sub_bin':
            return self.evaluate(et.children[0]) - self.evaluate(et.children[1])
        elif et.node_type_id == 'add_bin':
            return self.evaluate(et.children[0]) + self.evaluate(et.children[1])

    def left_most_reduction(self, et):
        """Reduce the leftmost subtree
            Args:
                et: expression tree
            Returns:
                an et with the leftmost reducible in one step subtree reduced
                (ie having two digit leaves)

                or identity with a single digit
        """

        leaf_value_type = self.one_step_derived_tree_def.leaves_types[0].value_type
        # adapt leaf embedding space
        done = False

        def _f(t):
            nonlocal done
            if t.node_type_id == 'num_value':
                return Tree(node_type_id='num_value', value=leaf_value_type(abstract_value=t.value.abstract_value))
            elif not done and all(map(lambda c: c.node_type_id == 'num_value', t.children)):
                done = True
                return Tree(node_type_id='num_value', children=[],
                            value=leaf_value_type(abstract_value=self.evaluate(t)))
            else:
                return Tree(node_type_id=t.node_type_id, children=list(map(_f, t.children)), value=None)

        return _f(et)


class NaryExpressionTreeGen(BinaryExpressionTreeGen):
    def __init__(self, min_value, max_value, max_arity):
        super(NaryExpressionTreeGen, self).__init__(min_value, max_value)

        self.tree_def = TreeDefinition(
            node_types=[
                NodeDefinition("add_n", may_root=True,
                               arity=NodeDefinition.VariableArity(min_value=2, max_value=max_arity),
                               value_type=None),
                NodeDefinition("sub_bin", may_root=True, arity=NodeDefinition.FixedArity(2),
                               value_type=None),
                NodeDefinition("num_value", may_root=True, arity=NodeDefinition.FixedArity(0),
                               value_type=self.NumValue)
            ]
        )

        self.one_step_derived_tree_def = TreeDefinition(
            node_types=[
                NodeDefinition("add_n", may_root=True, arity=NodeDefinition.VariableArity(min_value=2, max_value=max_arity)),
                NodeDefinition("sub_bin", may_root=True, arity=NodeDefinition.FixedArity(2)),
                NodeDefinition("num_value", may_root=True, arity=NodeDefinition.FixedArity(0),
                               value_type=NodeDefinition.NumValue(min_value - max_value, max_value * max_arity))
            ]
        )

        self.leaf_values = list(range(min_value, max_value + 1))
        self.node_types = self.tree_def.node_types

    def generate(self, max_depth, avoid_leaf=False):
        """Generate a random arithmetic expression tree,
            using just n-ary plus and binary minus
            Args:
                max_depth: integer > 0
            Returns:
                expression tree where leaves are int.
        """

        if max_depth == 1: # recursion base case
            v = random.sample(self.leaf_values, 1)[0]
            return Tree(node_type_id='num_value', value=self.NumValue(abstract_value=v))

        elif max_depth > 1:
            if avoid_leaf:
                types = self.node_types[:-1]
            else:
                types = self.node_types

            node_type = random.sample(types, 1)[0]
            if node_type.id == 'num_value':
                return self.generate(1)

            elif node_type.id == 'add_n':
                n = random.randint(node_type.arity.min_value, node_type.arity.max_value)
                return Tree(node_type.id, children=[self.generate(max_depth - 1) for _ in range(n)])

            elif node_type.id == 'sub_bin':
                return Tree(node_type.id, children=[
                    self.generate(max_depth - 1),
                    self.generate(max_depth - 1)])

    def evaluate(self, t: Tree):
        if len(t.children) > 0:
            if t.node_type_id == 'add_n':
                return sum(map(self.evaluate, t.children))
            if t.node_type_id == 'sub_bin':
                return self.evaluate(t.children[0]) - self.evaluate(t.children[1])
        else:
            return t.value.abstract_value
