from src.base.tree import Tree
from src.utils.randoms import find_point, flip_coin, random_choice, find_common_region
from src.utils.logs import info, success

from numpy.typing import NDArray
from typing import List


def standard_crossover(individs: NDArray[Tree], proba: float) -> Tree:
    """
    Стандартное скрещивание в генетическом программировании.
    Точки разрыва выбираются случайно

    :return: Случайно выбранный потомок, после операции скрещивания родителей
    """
    individ_1_copy: Tree = individs[0].copy()
    individ_2_copy: Tree = individs[1].copy()

    info(f"Chosen tree 1 for crossover: {individ_1_copy}")
    info(f"Chosen tree 2 for crossover: {individ_2_copy}")

    if flip_coin(proba):
        info('Starting standard crossover')
        point_crossover_1 = find_point(individ_1_copy)
        point_crossover_2 = find_point(individ_2_copy)

        individ_1_subtree = individ_1_copy.find_subtree(point_crossover_1)
        individ_2_subtree = individ_2_copy.find_subtree(point_crossover_2)

        info(f"Found subtree in tree 1: {individ_1_subtree}")
        info(f"Found subtree in tree 2: {individ_2_subtree}")

        individ_1_copy.replace_subtree(individ_1_subtree, individ_2_subtree)
        individ_2_copy.replace_subtree(individ_2_subtree, individ_1_subtree)

        info(f"Tree 1 after crossing: {individ_1_copy}")
        info(f"Tree 2 after crossing: {individ_2_copy}")

        if flip_coin(0.5):
            chosen_individ = individ_1_copy
        else:
            chosen_individ = individ_2_copy

        success(f"Chosen tree: {chosen_individ}")

        return chosen_individ

    if flip_coin(0.5):
        chosen_individ = individ_1_copy
    else:
        chosen_individ = individ_2_copy

    success(f"Chosen tree: {chosen_individ}")

    return chosen_individ


def one_point_crossover(individs: NDArray[Tree], proba: float) -> Tree:
    """
    Одноточечное скрещивание в генетическом программировании.
    Точка разрыва выбирается случайно из общей области деревьев.

    :param individs: Массив из двух деревьев для скрещивания
    :param proba: Вероятность выполнения скрещивания
    :return: Случайно выбранный потомок после операции скрещивания родителей
    """
    individ_1_copy: Tree = individs[0].copy()
    individ_2_copy: Tree = individs[1].copy()

    info(f"Chosen tree 1 for crossover: {individ_1_copy}")
    info(f"Chosen tree 2 for crossover: {individ_2_copy}")

    if flip_coin(proba):
        info('Starting single-point crossover')
        common_region = find_common_region(individ_1_copy, individ_2_copy)
        if not common_region:
            # warning("No common region found. Returning random parent.")
            return individ_1_copy if flip_coin(0.5) else individ_2_copy

        crossover_point = random_choice(common_region)

        individ_1_subtree = individ_1_copy.find_subtree(crossover_point[0])
        individ_2_subtree = individ_2_copy.find_subtree(crossover_point[1])

        info(f"Found subtree in tree 1: {individ_1_subtree}")
        info(f"Found subtree in tree 2: {individ_2_subtree}")

        individ_1_copy.replace_subtree(individ_1_subtree, individ_2_subtree)
        individ_2_copy.replace_subtree(individ_2_subtree, individ_1_subtree)

        info(f"Tree 1 after crossing: {individ_1_copy}")
        info(f"Tree 2 after crossing: {individ_2_copy}")

        chosen_individ = individ_1_copy if flip_coin(0.5) else individ_2_copy
        success(f"Chosen tree: {chosen_individ}")

        return chosen_individ

    chosen_individ = individ_1_copy if flip_coin(0.5) else individ_2_copy
    success(f"Chosen tree: {chosen_individ}")

    return chosen_individ
