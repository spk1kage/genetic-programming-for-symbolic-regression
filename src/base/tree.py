from __future__ import annotations

from ..utils.functional_operators import Operator, create_operator
from ..utils.functional_operators import cos, sin, save_div, save_exp, sqrtabs, logabs
from ..utils.randoms import flip_coin
from operator import add
from operator import sub
from operator import mul
from operator import abs

from copy import deepcopy
from typing import Union, Optional, Tuple, List

from numpy.typing import NDArray
import numpy as np

import random
import networkx as nx
import matplotlib.pyplot as plt


class Node:
    def __init__(self, value: Union[int, float, NDArray, Operator], name: str = None,
                 left: Node = None, right: Node = None, index: int = None, root: bool = None):
        self.root = root
        self.index = index
        self.value = value
        self.name = name
        self.left = left
        self.right = right

    def __str__(self) -> str:
        if isinstance(self.value, Operator):
            return str(self.value.sign)
        else:
            return str(self.name)


class UniversalSet:
    """
    A base class for creating a universal set used in expression trees.

    Parameters
    ----------
    array : NDArray[Union[np.float64, np.int64]]
        A one-dimensional or multi-dimensional array representing the terminal set. The values of
        this array can be used in the leaves of an expression tree.
    functions_names : Tuple[str, ...], optional, default=('cos', 'sin', 'add', 'sub', 'mul', 'div', 'abs')
        A tuple of strings representing the names of functions that will be included in the
        functional set.

    Attributes
    ----------
    terminal_set : NDArray[Union[np.float64, np.int64]]
        An array of terminal values created from the `array`.
    functional_set : Dict[str, Operator]
        A dictionary of Operator objects created based on the function names provided
        in `functions_names`.

    Examples
    --------
    >>> from src.base.tree import UniversalSet
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> uniset = UniversalSet(x)
    >>> random_functional_node = uniset.random_functional()
    >>> print(random_functional_node)
    sin
    >>> random_terminal_node = uniset.random_terminal()
    >>> print(random_terminal_node)
    np.array([1, 2, 3, 4, 5], dtype=np.int64)
    >>> random_constant_node = uniset.random_constant()
    >>> print(random_constant_node)
    8
    """
    def __init__(self, array: NDArray[Union[np.float64, np.int64]],
                 functions_names: Tuple[str, ...] = ('cos', 'sin', 'add', 'sub', 'mul', 'div', 'abs')):

        self.terminal_set = array

        self.ALL_FUNCTIONS = {
            "cos": create_operator("cos({})", "cos", "cos", cos, 1),
            "sin": create_operator("sin({})", "sin", "sin", sin, 1),
            "add": create_operator("({} + {})", "add", "+", add, 2),
            "sub": create_operator("({} - {})", "sub", "-", sub, 2),
            "mul": create_operator("({} * {})", "mul", "*", mul, 2),
            "div": create_operator("({} / {})", "div", "/", save_div, 2),
            "abs": create_operator("abs({})", "abs", "abs", abs, 1),
            "sqrt(abs)": create_operator("sqrt(abs({}))", "sqrt(abs)", "sqrt(abs)", sqrtabs, 1),
            "log(abs)": create_operator("log(abs({}))", "log(abs)", "log(abs)", logabs, 1),
            "exp": create_operator("exp({})", "exp", "exp", save_exp, 1),
        }

        invalid_functions = set(functions_names) - set(self.ALL_FUNCTIONS.keys())
        if invalid_functions:
            raise ValueError(f"Неизвестные функции: {invalid_functions}. "
                             f"Доступные функции: {list(self.ALL_FUNCTIONS.keys())}")

        self.functional_set = {
            name: self.ALL_FUNCTIONS[name] for name in functions_names
        }

    def get_available_functions(self) -> List[str]:
        """
        Returns
        -------
        List[str]
            Returns a list of available functions.
        """

        return list(self.functional_set.keys())

    def random_terminal(self) -> Node:
        """
        Creates a random terminal ``Node``

        Returns
        -------
        Node
            ``Node`` with random terminal value of dimension param ``array``
        """
        if self.terminal_set.ndim == 1:
            value = self.terminal_set
            name = "x"
        else:
            dimension = random.randint(1, self.terminal_set.ndim)

            value = self.terminal_set[:, dimension]
            name = f"x{dimension}"
        return Node(value, name)

    def random_functional(self, arity: int = None) -> Node:
        """
        Creates a random functional ``Node`` from the available functions

        Parameters
        ----------
        arity: int, default=None
            The required arity of the operator (number of arguments) from 1 to 2.
            Let's say `sin` has arity 1, and `div` has arity 2.

        Returns
        -------
        Node
            ``Node`` with random functional value of class ``Operator``
        """
        if arity is not None:
            available_functions = [
                name for name, operator in self.functional_set.items() if operator.arity == arity
            ]
            if not available_functions:
                raise ValueError(f"Нет доступных операторов с арностью {arity}")

            random_function = random.choice(available_functions)
        else:
            random_function = random.choice(list(self.functional_set))

        random_operator = self.functional_set[random_function]
        return Node(random_operator, random_operator.sign)

    @staticmethod
    def random_constant(a: Optional[Union[int, float]] = 0, b: Optional[Union[int, float]] = 10) -> Node:
        """
        Creates a random constant ``Node`` in range [a, b], including both end points.

        Parameters
        ----------
        a: int or float, default=0
        b: int or float, default=10

        Returns
        -------
        Node
            ``Node`` with random constant value.
        """
        if flip_coin(0.5):
            random_value = random.randint(a, b)
        else:
            random_value = round(random.uniform(a, b), 4)
        return Node(random_value, f"{random_value}")


class Tree:
    """
    Class for growing a tree and functions for transforming it.

    Parameters
    ----------
    nodes: Node, optional
        Tree nodes of class Node. Since a node can have either a right or left node,
        formally the tree is contained in the Node class. The first node has the parameter ``root`` = ``true``.
    nodes_counter: int, default=0
        The number of all nodes in the tree.
    depth: int, default=0
        Tree depth.

    Attributes
    ----------
    nodes: Node, optional
        Tree nodes of class Node. Since a node can have either a right or left node,
        formally the tree is contained in the Node class. The first node has the parameter ``root`` = ``true``.
    nodes_counter: int, default=0
        Since the number of nodes may change after the tree transformation, it is necessary to recalculate the nodes
    depth: int, default=0
        Also with depth.

    Examples
    --------
    Example for growing tree a `half on half`:

    >>> from src.base.tree import UniversalSet, Tree
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> uniset = UniversalSet(x)
    >>> depth = 5
    >>> tree = Tree().random_growing_method(uniset, depth)
    >>> print(tree)
    abs((sin(cos(0.1183)) * (cos(1.3692) * sin(x))))

    You can also visualize the grown tree:

    >>> tree.plot()

    Or:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.figure(figsize=(14, 8))
    >>> tree.plot(ax)
    >>> plt.show()

    References
    ----------
    - Binary expression tree (https://en.wikipedia.org/wiki/Binary_expression_tree)
    """
    def __init__(self, nodes: Node = None, nodes_counter: int = 0, depth: int = 0):
        self.nodes = nodes
        if self.nodes:
            self.nodes_counter = self.__inorder_traversal(self.nodes)
            self.depth = self.__calculate_depth(self.nodes)
        else:
            self.nodes_counter = nodes_counter
            self.depth = depth

    # PUBLIC METHODS

    def __len__(self) -> int:
        return self.get_depth()

    def __str__(self) -> str:
        return self.phenotype()

    def __eq__(self, other: Tree) -> bool:
        if isinstance(other, Tree):
            if len(self) == len(other) and self.get_nodes() == other.get_nodes():
                return True
            return False
        else:
            raise TypeError(f"Cannot compare Tree with {type(other).__name__}")

    def copy(self) -> deepcopy(Tree):
        """
        Creates a full deepcopy of the tree.
        Copies all nodes, the nodes_counter, and the depth of tree.
        """
        return Tree(deepcopy(self.nodes), deepcopy(self.nodes_counter), deepcopy(self.depth))

    def get_nodes(self) -> Node:
        """
        Returns the root node of the tree.
        Provides access to the tree structure through its root.
        """
        return self.nodes

    def get_nodes_counter(self) -> int:
        """
        Returns the current number of nodes in the tree.
        Allows you to track the size of the tree.
        """
        return self.nodes_counter

    def get_depth(self) -> int:
        """
        Returns the current tree depth.
        Depth is defined as the maximum distance from the root to the leaves.
        """
        return self.depth

    def random_growing_method(self, uniset: UniversalSet, max_depth: int) -> Tree:
        """
        Randomly chooses between the growing and full growing methods for building the tree.
        Uses a 50/50 probability to choose the method.

        Parameters
        ----------
        uniset: UniversalSet
            Universal set generated by the class ``UniversalSet``
        max_depth: int
            Depth of tree being grown

        Returns
        -------
        Tree
            Functions returns the value of self
        """
        if flip_coin(0.5):
            return self.growing_method(uniset, max_depth)
        else:
            return self.full_growing_method(uniset, max_depth)

    def growing_method(self, uniset: UniversalSet, max_depth: int) -> Tree:
        """
        Builds a tree using the growing method.
        With this method, nodes can be either terminal or non-terminal
        at any depth up to the maximum.

        Parameters
        ----------
        uniset: UniversalSet
            Universal set generated by the class ``UniversalSet``
        max_depth:
            Depth of tree being grown

        Returns
        -------
        Tree
            Functions returns the value of self
        """
        self.nodes = self.__growing_method(uniset=uniset, max_depth=max_depth)
        self.nodes_counter = self.__inorder_traversal(self.nodes)
        self.depth = self.__calculate_depth(self.nodes)
        return self

    def full_growing_method(self, uniset: UniversalSet, max_depth: int) -> Tree:
        """
        Builds a tree using the full growing method.
        With this method, all nodes up to the maximum depth will be non-terminal,
        and at the maximum depth - only terminal.

        Parameters
        ----------
        uniset: UniversalSet
            Universal set generated by the class ``UniversalSet``
        max_depth:
            Depth of tree being grown

        Returns
        -------
        Tree
            Functions returns the value of self
        """
        self.nodes = self.__full_growing_method(uniset=uniset, max_depth=max_depth)
        self.nodes_counter = self.__inorder_traversal(self.nodes)
        self.depth = self.__calculate_depth(self.nodes)
        return self

    def replace_subtree(self, old_subtree: Tree, new_subtree: Tree) -> None:
        """
        Replaces the subtree old_subtree with new_subtree in the current tree.

        Parameters
        ----------
        old_subtree: Tree
            A tree that needs to be replaced
        new_subtree: Tree
            A tree to replace the old_subtree with
        """
        self.nodes = self.__replace_subtree(self.nodes, old_subtree.get_nodes(), new_subtree.get_nodes())
        self.nodes_counter = self.__inorder_traversal(self.nodes)
        self.depth = self.__calculate_depth(self.nodes)

    def change_node(self, uniset: UniversalSet, point: int, node: Node = None) -> Node:
        """
        Changes the value and name of a specified node in the tree based on its type.

        Parameters
        ----------
        uniset: UniversalSet
            Universal set generated by the class ``UniversalSet``
        point: int
            Index of the node to be changed.
        node: Node, optional
            Current node being processed. If None, starts from root.

        Returns
        -------
        Node
            Modified node with updated value and name.

        Notes
        -----
        For operator nodes, changes to a new random functional node with same arity.
        For terminal/constant nodes, randomly changes to either a new constant or terminal
        with 50% probability each.
        """
        if node is None:
            node = self.nodes

        if node.index == point:
            if isinstance(node.value, Operator):
                random_functional_node = uniset.random_functional(node.value.arity)
                node.value = random_functional_node.value
                node.name = random_functional_node.name
            else:
                if flip_coin(0.5):
                    random_constant_node = uniset.random_constant()
                    node.value = random_constant_node.value
                    node.name = random_constant_node.name
                else:
                    random_terminal_node = uniset.random_terminal()
                    node.value = random_terminal_node.value
                    node.name = random_terminal_node.name

            return node

        if node.left:
            node.left = self.change_node(uniset, point, node.left)

        if node.right:
            node.right = self.change_node(uniset, point, node.right)

        return node

    def phenotype(self, node: Node = None) -> str:
        """
        Converts the tree into a string representation of the mathematical expression (phenotype).
        Recursively traverses the tree, converting operators and values into string form.

        Parameters
        ----------
        node: Node, optional
            Current node being processed. If None, starts from root.

        Returns
        -------
        str
            String representation of the mathematical expression.

        Examples
        --------
        For tree: mul(2, add(x, 3))
        Returns: "(2 * (x + 3))"
        """
        if node is None:
            node = self.nodes

        if isinstance(node.value, Operator):
            if node.value.arity == 1:
                if node.left:
                    return node.value.write(self.phenotype(node.left))
                else:
                    return node.value.write(self.phenotype(node.right))
            if node.value.arity == 2:
                return node.value.write(self.phenotype(node.left), self.phenotype(node.right))
        else:
            return str(node.name)

    def genotype(self, node: Node = None) -> NDArray[Union[np.float64, np.int64]] | np.float64 | np.int64:
        """
        Computes the value of the mathematical expression (genotype).
        Recursively traverses the tree, applying operators to node values.

        Parameters
        ----------
        node: Node, optional
            Current node being processed. If None, starts from root.

        Returns
        -------
        Union[NDArray, float, int]
            Result of expression evaluation, can be array, float, or int.

        Examples
        --------
        For tree: mul(2, add(3, 4))
        Returns: (2 * (3 + 4)) = 14
        """
        if node is None:
            node = self.nodes

        if isinstance(node.value, Operator):
            if node.value.arity == 1:
                if node.left:
                    return node.value(self.genotype(node.left))
                else:
                    return node.value(self.genotype(node.right))
            if node.value.arity == 2:
                return node.value(self.genotype(node.left), self.genotype(node.right))
        else:
            return node.value

    def find_subtree(self, point: int, node: Node = None) -> Tree:
        """
        Recursively searches for a node with the specified index.

        Parameters
        ----------
        point : int
            Index of the node to find.
        node : Node, optional
            Current node being processed. If None, starts from root.

        Returns
        -------
        Tree
            A tree with found node as root, or None if node not found.
        """
        if node is None:
            node = self.nodes

        if node.index == point:
            return Tree(node)

        if node.left:
            left_result = self.find_subtree(point, node.left)
            if left_result:
                return left_result

        if node.right:
            right_result = self.find_subtree(point, node.right)
            if right_result:
                return right_result

    def plot(self, ax=None) -> None:
        """
        Visualizes the tree structure using NetworkX and Matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the tree. If None, creates new figure.

        Notes
        -----
        - Operator nodes are shown in light blue
        - Terminal/constant nodes are shown in light green
        - The expression is shown as the plot title
        """
        G = nx.DiGraph()
        pos = {}

        def add_nodes_edges(node, x=0, y=0, layer=1, parent=None, is_left=True):
            node_id = id(node)
            if isinstance(node.value, Operator):
                G.add_node(node_id, color='lightblue', label=node.value.sign)
            else:
                G.add_node(node_id, color='lightgreen', label=str(node.name))

            pos[node_id] = (x, -y)
            if parent:
                G.add_edge(parent, node_id)

            spacing = 1.9 / (2 ** (layer - 1))

            if node.left:
                add_nodes_edges(node.left, x - spacing, y + 1, layer + 1, node_id, True)

            if node.right:
                add_nodes_edges(node.right, x + spacing, y + 1, layer + 1, node_id, False)

        node = self.nodes
        add_nodes_edges(node)

        if ax is None:
            plt.figure(figsize=(14, 8))

        nx.draw(G, pos,
                node_color=[G.nodes[node]['color'] for node in G.nodes()],
                labels={node: G.nodes[node]['label'] for node in G.nodes()},
                with_labels=True,
                node_size=1500,
                font_size=10,
                edgecolors="black",
                linewidths=1,
                ax=ax)

        plt.title(f"Выражение: {self.phenotype()}")
        plt.axis('off')

        if ax is None:
            plt.show()

    # PRIVATE METHODS

    def __replace_subtree(self, node: Optional[Node], old_node: Node, new_node: Node) -> Node | None:
        """
        Recursively finds and replaces subtree old_node with new_node.

        Parameters
        ----------
        node : Optional[Node]
            Current node being processed
        old_node : Node
            ``Node`` to be replaced
        new_node : Node
            ``Node`` to replace with

        Returns
        -------
        Node or None
            Node after replacement
        """
        if node is None:
            return None

        if node == old_node:
            return new_node

        node.left = self.__replace_subtree(node.left, old_node, new_node)
        node.right = self.__replace_subtree(node.right, old_node, new_node)
        return node

    def __growing_method(self, uniset: UniversalSet, max_depth: int,
                         current_depth: int = 0, previous_node: Node = None) -> Node:
        """
        Implements the growing method for tree generation with variable depth.

        Parameters
        ----------
        uniset: UniversalSet
            Universal set generated by the class ``UniversalSet``
        max_depth : int
            Maximum allowed depth of the tree
        current_depth : int, optional
            Current depth in the tree, by default 0
        previous_node : Node, optional
            Parent node in the tree, by default None

        Returns
        -------
        Node
            Generated node according to growing method rules

        Notes
        -----
        - At max_depth-1, creates terminal or constant node with 0.5 probability
        - For intermediate depths, creates operator nodes with varying arity
        - Uses coin flip (0.5 probability) for various decision points
        """
        if current_depth == max_depth - 1:
            if flip_coin(0.5):
                return uniset.random_terminal()
            else:
                return uniset.random_constant()

        if current_depth < max_depth - 1:
            if previous_node:
                if previous_node.left and isinstance(previous_node.left.value, Operator) \
                        or previous_node.right and isinstance(previous_node.right.value, Operator):
                    if flip_coin(0.5):
                        return uniset.random_terminal()
                    else:
                        return uniset.random_constant()

        functional_node = uniset.random_functional()

        if current_depth == 0:
            functional_node.root = True

        if functional_node.value.arity == 1:
            if flip_coin(0.5):
                functional_node.left = self.__growing_method(uniset, max_depth, current_depth + 1, functional_node)
            else:
                functional_node.right = self.__growing_method(uniset, max_depth, current_depth + 1, functional_node)
        if functional_node.value.arity == 2:
            if flip_coin(0.5):
                functional_node.left = self.__growing_method(uniset, max_depth, current_depth + 1, functional_node)
                functional_node.right = self.__growing_method(uniset, max_depth, current_depth + 1, functional_node)
            else:
                functional_node.right = self.__growing_method(uniset, max_depth, current_depth + 1, functional_node)
                functional_node.left = self.__growing_method(uniset, max_depth, current_depth + 1, functional_node)
        return functional_node

    def __full_growing_method(self, uniset: UniversalSet, max_depth: int, current_depth: int = 0) -> Node:
        """
        Implements the full growing method for tree generation with fixed depth.

        Parameters
        ----------
        uniset: UniversalSet
            Universal set generated by the class ``UniversalSet``
        max_depth : int
            Maximum allowed depth of the tree
        current_depth : int, optional
            Current depth in the tree, by default 0

        Returns
        -------
        Node
            Generated node according to full growing method rules

        Notes
        -----
        - Creates a full tree where all leaves are at max_depth-1
        - At max_depth-1, creates terminal or constant node with 0.5 probability
        - For intermediate depths, always creates operator nodes
        """
        if current_depth == max_depth - 1:
            if flip_coin(0.5):
                return uniset.random_terminal()
            else:
                return uniset.random_constant()

        functional_node = uniset.random_functional()

        if current_depth == 0:
            functional_node.root = True

        if functional_node.value.arity == 1:
            if flip_coin(0.5):
                functional_node.left = self.__full_growing_method(uniset, max_depth, current_depth + 1)
            else:
                functional_node.right = self.__full_growing_method(uniset, max_depth, current_depth + 1)
        if functional_node.value.arity == 2:
            if flip_coin(0.5):
                functional_node.left = self.__full_growing_method(uniset, max_depth, current_depth + 1)
                functional_node.right = self.__full_growing_method(uniset, max_depth, current_depth + 1)
            else:
                functional_node.right = self.__full_growing_method(uniset, max_depth, current_depth + 1)
                functional_node.left = self.__full_growing_method(uniset, max_depth, current_depth + 1)

        return functional_node

    def __inorder_traversal(self, node: Node, counter: int = 0) -> int:
        """
        Numbers each tree node using inorder traversal and returns total node count.

        Parameters
        ----------
        node : Node
            Current node being processed
        counter : int, optional
            Current node counter, by default 0

        Returns
        -------
        int
            Total number of numbered nodes

        Notes
        -----
        Inorder traversal (NLR):
        1. Check if current node is empty or null
        2. Traverse left subtree recursively
        3. Number current node
        4. Traverse right subtree recursively

        Node indexing starts from 1
        """
        if node is None:
            return counter

        counter = self.__inorder_traversal(node.left, counter)
        counter += 1
        node.index = counter
        counter = self.__inorder_traversal(node.right, counter)

        return counter

    def __calculate_depth(self, node: Node) -> int:
        """
        Calculates the maximum depth of the tree.

        Parameters
        ----------
        node : Node
            Root node of the tree or subtree

        Returns
        -------
        int
            Maximum depth of the tree (number of levels)
        """
        if node is None:
            return 0
        return max(self.__calculate_depth(node.left), self.__calculate_depth(node.right)) + 1
