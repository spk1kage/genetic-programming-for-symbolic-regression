from src.base.tree import UniversalSet, Tree
from src.utils.randoms import find_point, flip_coin, random_randint
from src.utils.logs import info, success


def growing_mutation(tree: Tree, uniset: UniversalSet, proba: float) -> Tree:
    """
    Мутация выращиванием. Выбирается случайное поддерево и заменяется выращенным
    с глубиной не больше выбранного.
    :param tree:
    :param uniset:
    :param proba:
    :return:
    """
    mutated_tree = tree.copy()

    if flip_coin(proba):
        info("Starting growing mutation")
        info(f"Mutated individ: {mutated_tree}")

        point = find_point(mutated_tree)
        subtree = mutated_tree.find_subtree(point)
        info(f"Find subtree for mutation: {subtree}")

        depth = random_randint(1, len(subtree))
        info(f"Generating new subtree with depth: {depth}")

        growing_subtree = Tree().random_growing_method(uniset, depth)
        info(f"Growing mutated subtree: {growing_subtree}")

        mutated_tree.replace_subtree(subtree, growing_subtree)
        success(f"Individ after mutation: {mutated_tree}")

    return mutated_tree


def point_mutation(tree: Tree, uniset: UniversalSet, proba: float):
    """
    Точечная мутация. Выбирается случайный узел в дереве и заменяется значением
    из соответствующего множества. Если выбранный узел находится в функциональном множестве,
    то заменяется оператором такой же арности.
    :param tree:
    :param uniset:
    :param proba:
    :return:
    """
    mutated_tree = tree.copy()

    if flip_coin(proba):
        info("Starting point mutation")
        info(f"Mutated individ: {mutated_tree}")

        point = find_point(mutated_tree)

        mutated_tree.change_node(uniset, point)
        success(f"Individ after mutation: {mutated_tree}")

    return mutated_tree


# def hoist_mutation():
# Выбирается поддерево и оно поднимается в дереве
#
#
# def shrink_mutation():
# случайным образом выбирает узел в дереве и с заданной вероятностью заменяет его
# одним из его конечных узлов, эффективно обрезая поддерево с корнем в выбранном узле.
