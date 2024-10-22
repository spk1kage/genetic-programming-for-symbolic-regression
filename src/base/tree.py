from __future__ import annotations

from src.utils.functional_operators import Operator, create_operator
from src.utils.functional_operators import cos, sin, save_div, save_exp, sqrtabs, logabs
from src.utils.randoms import flip_coin
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
    Создание универсального множества для выращивания деревьев
    с возможностью выбора доступных функций.
    """
    def __init__(self, array: NDArray[Union[np.float64, np.int64]],
                 functions_names: Tuple[str, ...] = ('cos', 'sin', 'add', 'sub', 'mul', 'div', 'abs')):

        self.array = array

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

        self.SYMBOLIC_FUNCTIONS = {
            name: self.ALL_FUNCTIONS[name] for name in functions_names
        }

    def get_available_functions(self) -> List[str]:
        """
        Возвращает список имен доступных функций.

        :return: Список имен функций
        """
        return list(self.SYMBOLIC_FUNCTIONS.keys())

    def random_terminal(self) -> Node:
        """
        Создает случайный терминальный узел.

        :return: Узел с терминальным значением
        """
        if self.array.ndim == 1:
            value = self.array
            name = "x"
        else:
            dimension = random.randint(1, self.array.ndim)

            value = self.array[:, dimension]
            name = f"x{dimension}"
        return Node(value, name)

    def random_functional(self, arity: int = None) -> Node:
        """
        Создает случайный функциональный узел из доступных функций.

        :param arity: Требуемая арность оператора (количество аргументов)
        :return: Узел с функциональным оператором
        """
        if arity is not None:
            available_functions = [
                name for name, operator in self.SYMBOLIC_FUNCTIONS.items() if operator.arity == arity
            ]
            if not available_functions:
                raise ValueError(f"Нет доступных операторов с арностью {arity}")

            random_function = random.choice(available_functions)
        else:
            random_function = random.choice(list(self.SYMBOLIC_FUNCTIONS))

        random_operator = self.SYMBOLIC_FUNCTIONS[random_function]
        return Node(random_operator, random_operator.sign)

    @staticmethod
    def random_constant() -> Node:
        """
        Создает случайный узел-константу со значением от 0 до 10.

        :return: Узел с константным значением
        """
        if random.random() < 0.5:
            random_value = random.randint(0, 10)
        else:
            random_value = round(random.uniform(0, 10), 4)
        return Node(random_value, f"{random_value}")


class Tree:
    """
    Класс для реализации бинарного дерева выражений
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
        Создает полную глубокую копию дерева.
        Копирует все узлы, счетчик узлов и глубину дерева.

        :return: Новый экземпляр дерева, являющийся полной копией текущего
        """
        return Tree(deepcopy(self.nodes), deepcopy(self.nodes_counter), deepcopy(self.depth))

    def get_nodes(self) -> Node:
        """
        Возвращает корневой узел дерева.
        Предоставляет доступ к структуре дерева через его корень.

        :return: Корневой узел дерева (объект класса Node)
        """
        return self.nodes

    def get_nodes_counter(self) -> int:
        """
        Возвращает текущее количество узлов в дереве.
        Позволяет отслеживать размер дерева.

        :return: Целое число - количество узлов в дереве
        """
        return self.nodes_counter

    def get_depth(self) -> int:
        """
        Возвращает текущую глубину дерева.
        Глубина определяется как максимальное расстояние от корня до листьев.

        :return: Целое число - глубина дерева
        """
        return self.depth

    def random_growing_method(self, uniset: UniversalSet, max_depth: int) -> Tree:
        """
        Случайно выбирает между методом growing и full growing для построения дерева.
        Использует вероятность 50/50 для выбора метода.

        :param uniset: Универсальное множество доступных узлов
        :param max_depth: Максимальная допустимая глубина дерева
        :return: Построенное дерево (self)
        """
        if flip_coin(0.5):
            return self.growing_method(uniset, max_depth)
        else:
            return self.full_growing_method(uniset, max_depth)

    def growing_method(self, uniset: UniversalSet, max_depth: int) -> Tree:
        """
        Строит дерево используя метод growing.
        При этом методе узлы могут быть как терминальными, так и нетерминальными
        на любой глубине до достижения максимальной.

        :param uniset: Универсальное множество доступных узлов
        :param max_depth: Максимальная допустимая глубина дерева
        :return: Построенное дерево (self)
        """
        self.nodes = self.__growing_method(uniset=uniset, max_depth=max_depth)
        self.nodes_counter = self.__inorder_traversal(self.nodes)
        self.depth = self.__calculate_depth(self.nodes)
        return self

    def full_growing_method(self, uniset: UniversalSet, max_depth: int) -> Tree:
        """
        Строит дерево используя метод full growing.
        При этом методе все узлы до максимальной глубины будут нетерминальными,
        а на максимальной глубине - только терминальными.

        :param uniset: Универсальное множество доступных узлов
        :param max_depth: Максимальная допустимая глубина дерева
        :return: Построенное дерево (self)
        """
        self.nodes = self.__full_growing_method(uniset=uniset, max_depth=max_depth)
        self.nodes_counter = self.__inorder_traversal(self.nodes)
        self.depth = self.__calculate_depth(self.nodes)
        return self

    def replace_subtree(self, old_subtree: Tree, new_subtree: Tree) -> None:
        """
        Заменяет поддерево old_subtree на new_subtree в текущем дереве.

        :param old_subtree: Дерево, которое нужно заменить
        :param new_subtree: Дерево, на которое нужно заменить
        """
        self.nodes = self.__replace_subtree(self.nodes, old_subtree.get_nodes(), new_subtree.get_nodes())
        self.nodes_counter = self.__inorder_traversal(self.nodes)
        self.depth = self.__calculate_depth(self.nodes)

    def change_node(self, uniset: UniversalSet, point: int, node: Node = None):
        if node is None:
            node = self.nodes

        if node.index == point:
            print(node.name)
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
            print(node.name)
            return node

        if node.left is not None:
            node.left = self.change_node(uniset, point, node.left)

        if node.right is not None:
            node.right = self.change_node(uniset, point, node.right)

        return node

    def phenotype(self, node: Node = None) -> str:
        """
        Преобразует дерево в строковое представление математического выражения (фенотип).
        Рекурсивно обходит дерево, преобразуя операторы и значения в строковую форму.

        :param node: Текущий узел дерева (None для начала с корня)
        :return: Строковое представление математического выражения

        Пример:
        Для дерева: mul(2, add(x, 3))
        Вернет строку: "(2 * (x + 3))"
        """

        if node is None:
            node = self.nodes

        if isinstance(node.value, Operator):
            if node.value.arity == 1:  # Унарный оператор
                if node.left:
                    return node.value.write(self.phenotype(node.left))
                else:
                    return node.value.write(self.phenotype(node.right))
            if node.value.arity == 2:  # Бинарный оператор
                return node.value.write(self.phenotype(node.left), self.phenotype(node.right))
        else:
            # Терминальный узел
            return str(node.name)

    def genotype(self, node: Node = None) -> NDArray[Union[np.float64, np.int64]] | np.float64 | np.int64:
        """
        Вычисляет значение математического выражения (генотип).
        Рекурсивно обходит дерево, применяя операторы к значениям узлов.

        :param node: Текущий узел дерева (None для начала с корня)
        :return: Результат вычисления выражения (может быть массивом, float или int)

        Пример:
        Для дерева: mul(2, add(3, 4))
        Вернет значение: (2 * (3 + 4)) = 14
        """
        if node is None:
            node = self.nodes

        if isinstance(node.value, Operator):
            if node.value.arity == 1:
                # Унарный оператор
                if node.left:
                    return node.value(self.genotype(node.left))
                else:
                    return node.value(self.genotype(node.right))
            if node.value.arity == 2:
                # Бинарный оператор
                return node.value(self.genotype(node.left), self.genotype(node.right))
        else:
            # Терминальный узел
            return node.value

    def find_subtree(self, point: int, node: Optional[Node] = None) -> Optional[Tree]:
        """
        Рекурсивно ищет узел с заданным индексом.

        :param node: Текущий узел
        :param point: Индекс искомого узла
        :return: Найденный узел или None, если узел не найден
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

    def __replace_subtree(self, node: Optional[Node], old_node: Node, new_node: Node) -> Optional[Node]:
        """
        Рекурсивно находит и заменяет поддерево old_node на new_node.

        :param node: Текущий узел дерева
        :param old_node: Узел, который нужно заменить
        :param new_node: Узел, на который нужно заменить
        :return: Узел дерева после замены
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

        if functional_node.value.arity == 1:  # Унарный оператор
            if flip_coin(0.5):
                functional_node.left = self.__growing_method(uniset, max_depth, current_depth + 1, functional_node)
            else:
                functional_node.right = self.__growing_method(uniset, max_depth, current_depth + 1, functional_node)
        if functional_node.value.arity == 2:  # Бинарный оператор
            if flip_coin(0.5):
                functional_node.left = self.__growing_method(uniset, max_depth, current_depth + 1, functional_node)
                functional_node.right = self.__growing_method(uniset, max_depth, current_depth + 1, functional_node)
            else:
                functional_node.right = self.__growing_method(uniset, max_depth, current_depth + 1, functional_node)
                functional_node.left = self.__growing_method(uniset, max_depth, current_depth + 1, functional_node)
        return functional_node

    def __full_growing_method(self, uniset: UniversalSet, max_depth: int, current_depth: int = 0) -> Node:
        if current_depth == max_depth - 1:
            if flip_coin(0.5):
                return uniset.random_terminal()
            else:
                return uniset.random_constant()

        functional_node = uniset.random_functional()

        if current_depth == 0:
            functional_node.root = True

        if functional_node.value.arity == 1:  # Унарный оператор
            if flip_coin(0.5):
                functional_node.left = self.__full_growing_method(uniset, max_depth, current_depth + 1)
            else:
                functional_node.right = self.__full_growing_method(uniset, max_depth, current_depth + 1)
        if functional_node.value.arity == 2:  # Бинарный оператор
            if flip_coin(0.5):
                functional_node.left = self.__full_growing_method(uniset, max_depth, current_depth + 1)
                functional_node.right = self.__full_growing_method(uniset, max_depth, current_depth + 1)
            else:
                functional_node.right = self.__full_growing_method(uniset, max_depth, current_depth + 1)
                functional_node.left = self.__full_growing_method(uniset, max_depth, current_depth + 1)

        return functional_node

    def __inorder_traversal(self, node: Node, counter: int = 0) -> int:
        """
        Метод, который нумерует каждый узел дерева прямым обходом и возвращает общее количество узлов

        Прямой обход дерева (NLR):
        1. Проверяем, не является ли текущий узел пустым или null.
        2. Обходим левое поддерево рекурсивно, вызвав функцию прямого обхода.
        3. Нумеруем текущий узел.
        4. Обходим правое поддерево рекурсивно, вызвав функцию прямого обхода.
        :param node: Узел дерева
        :param counter: Текущий счетчик узлов
        :return: Общее количество пронумерованных узлов
        """
        if node is None:
            return counter

        counter = self.__inorder_traversal(node.left, counter)
        counter += 1
        node.index = counter  # Индексация узлов начинается с 1
        counter = self.__inorder_traversal(node.right, counter)

        return counter

    def __calculate_depth(self, node: Node) -> int:
        if node is None:
            return 0
        return max(self.__calculate_depth(node.left), self.__calculate_depth(node.right)) + 1
