# from src.base.tree import Tree
from typing import Union, Optional
import numpy as np
import random
import numbers


def random_weighted_sample(
    weights: np.ndarray, quantity: Union[int, np.int64] = 1, replace: bool = False
) -> np.ndarray:
    """
    Функция для генерации случайной выборки на основе весов.

    Parameters
    ----------
    weights : np.ndarray
        1D array of weights representing the probability of each element being selected.
    quantity : Union[int, np.int64]
        The number of elements to sample. Default is 1.
    replace : bool
        Whether sampling is done with replacement. Default is True.

    Returns
    -------
    np.ndarray
        An array of sampled indices.
    """

    population_size = len(weights)
    indices = np.arange(population_size)

    sampled_indices = np.random.choice(
        indices, size=quantity, replace=replace, p=weights / np.sum(weights)
    )

    return sampled_indices


def flip_coin(threshold):
    """
    Simulate a biased coin flip.

    Parameters
    ----------
    threshold : float64
        The threshold for the biased coin flip. Should be in the range [0, 1].

    Returns
    -------
    boolean
        Returns True with a probability equal to the given threshold and
        False with a complementary probability.

    Notes
    -----
    The function simulates a biased coin flip with a specified threshold.
    If the threshold is 0.5, it behaves like a fair coin flip.
    A threshold greater than 0.5 biases the result towards True,
    while a threshold less than 0.5 biases the result towards False.
    """
    return random.random() < threshold


def find_point(individ):
    """
    Find a random point in the tree different from the root node.

    Parameters
    ----------
    individ : Tree
        An object representing a tree structure. It must have methods
        get_nodes() and get_nodes_counter().

    Returns
    -------
    int
        A random node index different from the root node's index.

    Notes
    -----
    This function selects a random node from the tree, excluding the root node.
    It uses a while loop to ensure that the selected point is different from
    the root node's index. The function assumes that node indices start from 1.

    If the tree has only one node (the root), this function will enter an
    infinite loop. It's recommended to add a check for this case in production code.

    The randomness of the selection depends on the random.randint() function,
    which uses the Mersenne Twister as the core generator.

    Examples
    --------
    >>> tree = Tree()  # Assume Tree is a class representing your tree structure
    >>> random_point = find_point(tree)
    >>> print(random_point)
    """
    nodes = individ.get_nodes()
    nodes_counter = individ.get_nodes_counter()

    while True:
        point = random.randint(1, nodes_counter)
        if point != nodes.index:
            return point


def find_common_region(tree1, tree2):

    common_region = []

    def traverse(node1, node2):
        if node1 is None or node2 is None:
            return

        if not node1.root or not node2.root:
            common_region.append((node1.index, node2.index))
        traverse(node1.left, node2.left)
        traverse(node1.right, node2.right)

    traverse(tree1.get_nodes(), tree2.get_nodes())
    return common_region


def random_randint(a, b):
    return random.randint(a, b)


def random_choice(array):
    return random.choice(array)


def check_random_state(seed: Optional[Union[int, np.random.RandomState]] = None) -> int:
    """
    Преобразует seed в целочисленное значение и применяет его для random и numpy.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        Если seed равен None, возвращает текущее случайное состояние numpy.
        Если seed — целое число, то используется этот seed.
        Если seed уже является экземпляром RandomState, возвращает его seed.
        В противном случае генерирует ValueError.

    Returns
    -------
    int
        Возвращает целочисленное значение seed.
    """
    if seed is None:
        random_state = np.random.mtrand._rand
        seed = random_state.get_state()[1][0]
    elif isinstance(seed, (numbers.Integral, np.integer)):
        random_state = np.random.RandomState(seed)
        seed = random_state.get_state()[1][0]
        set_seed(seed)
    elif isinstance(seed, np.random.RandomState):
        random_state = seed
        seed = random_state.get_state()[1][0]
        set_seed(seed)
    else:
        raise ValueError(f"{seed} не может быть использован для генерации numpy.random.RandomState")

    return seed


def set_seed(seed_value: int):
    """
    Устанавливает seed для random и numpy.
    """
    random.seed(int(seed_value))
    np.random.seed(seed_value)
