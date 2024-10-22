from src.base.tree import Tree, UniversalSet
from src.operators.selections import proportional_selection, rank_selection, tournament_selection
from src.operators.crossovers import standard_crossover, one_point_crossover
from src.operators.mutations import growing_mutation, point_mutation
from src.utils.metrics import mean_square_error, root_mean_square_error, mean_absolute_error
from src.utils.randoms import random_randint, check_random_state
from src.utils.logs import enable_logging, info

from typing import Optional, Literal, Tuple, Callable, Dict
from functools import partial

from numpy.typing import NDArray
import numpy as np


class GeneticProgrammingSymbolRegression:
    """
    GeneticProgrammingSymbolRegression
    ----------------------------------

    Реализация алгоритма генетического программирования для задачи регрессии

    Атрибуты:
    ---------
    n_iters : int
        Количество итераций алгоритма.
    pop_size : int
        Размер популяции.
    max_depth : int
        Максимальная глубина дерева.
    elitism: float или None
        Сколько лучших индивидов оставлять в новую популяцию
    limit_depth : int или None
        Ограничение глубины (если есть).
    selection : str
        Выбранный метод отбора.
    tour_size : int
        Размер турнира для селекции.
    crossover : str
        Выбранный метод скрещивания.
    crossover_rate: float
        Вероятность скрещивания.
    mutation : str
        Выбранный метод мутации.
    mutation_rate : float
        Вероятность мутации.
    is_constant_rate: bool
        Постоянная ли константа мутации.
        Если True, то используется заданная константа mutation_rate,
        если False, то mutation_rate = mutation_rate / Tree.get_depth()
    metric : str
        Выбранная метрика оценки качества решения.
    is_logging : bool
        Записывать ли операции в файл с логами
    show_progress_each : int или None
        Частота вывода прогресса.
    termination: int или None
        Через сколько итераций завершать выполнение алгоритма,
        если лучших индивид не меняется
    """
    def __init__(
            self,
            *,
            n_iters: int = 500,
            pop_size: int = 100,
            elitism: float = None,
            max_depth: int = 5,
            limit_depth: int = 12,
            metric: Literal['MSE', 'RMSE', 'MAE'] = 'MSE',
            selection: Literal[
                'proportional', 'rank',
                'tournament_k', 'tournament_3',
                'tournament_5', 'tournament_7'] = 'proportional',
            tour_size: int = 5,
            crossover: Literal['standard', 'one_point'] = 'standard',
            crossover_rate: float = 0.9,
            mutation: Literal['grow', 'point'] = 'grow',
            mutation_rate: float = 0.4,
            is_const_mut_rate: bool = True,
            is_logging: bool = False,
            random_state: int = None,
            show_progress_each: Optional[int] = None,
            termination: int = None

    ):
        self.n_iters = n_iters
        self.pop_size = pop_size
        self.elitism = elitism
        self.max_depth = max_depth
        self.limit_depth = limit_depth
        self.mutation_rate = mutation_rate

        seed = check_random_state(random_state)
        info(f'Seed is {seed}')

        self.show_progress_each = show_progress_each
        self.termination = termination

        self.specified_selection = selection
        self.specified_crossover = crossover
        self.specified_mutation = mutation
        self.specified_metric = metric

        self.selection_pool: Dict[str, Tuple] = {
            'proportional': (proportional_selection, 0),
            'rank':         (rank_selection, 0),
            'tournament_k': (tournament_selection, tour_size),
            'tournament_3': (tournament_selection, 3),
            'tournament_5': (tournament_selection, 5),
            'tournament_7': (tournament_selection, 7),
        }

        self.crossover_pool: Dict[str, Tuple] = {
            'standard': (standard_crossover, 2, crossover_rate),
            'one_point': (one_point_crossover, 2, crossover_rate)
        }

        self.mutation_pool: Dict[str, Tuple] = {
            'grow': (growing_mutation, mutation_rate, is_const_mut_rate),
            'point': (point_mutation, mutation_rate, is_const_mut_rate)
        }

        self.metric_pool: Dict[str, Callable] = {
            'MSE': mean_square_error,
            'RMSE': root_mean_square_error,
            'MAE': mean_absolute_error
        }

        if is_logging:
            enable_logging(True)

        info(
            f"Initialized parameters: n_iter={n_iters}, pop_size={pop_size}, max_depth={max_depth}, "
            f"metric={metric}, selection={selection}, tour_size={self.selection_pool[selection][1]}, "
            f"crossover={crossover}, mutation={mutation}, mutation_rate={mutation_rate}"
        )

        self.population = None
        self.fitness = None

        self.__no_update_counter: int = 0

        self.best_fitness: float = float('inf')
        self.best_tree = None

    def half_and_half(self, uniset: UniversalSet) -> NDArray[Tree]:
        info(f'Start initializing first population "half and half" method')

        population = np.empty(shape=self.pop_size, dtype=object)
        for i in range(self.pop_size):
            population[i] = Tree().random_growing_method(uniset, random_randint(2, self.max_depth))

        return population

    def get_init_population(self, uniset: UniversalSet) -> None:
        self.population = self.half_and_half(uniset=uniset)

        for idx, tree in enumerate(self.population):
            info(f'Created tree {idx}: {tree}')

    def from_population_to_fitness(self, y_true: NDArray[np.float64]):
        info("Calculating fitness scores")

        fitness = np.empty(self.pop_size)
        for idx, tree in enumerate(self.population):

            y_pred = tree.genotype() * np.ones(len(y_true))
            fitness[idx] = self.metric_pool[self.specified_metric](y_true, y_pred)

            info(f"Fitness for tree {idx}: {fitness[idx]}")

        self.fitness = fitness.copy()
        self.__update()

    def __termination_check(self):
        if self.termination:
            if self.__no_update_counter >= self.termination:
                return True
        return False

    def get_new_individ(
            self,
            uniset: UniversalSet,
            specified_selection: str,
            specified_crossover: str,
            specified_mutation: str
    ) -> Tree:
        selection_func, tour_size = self.selection_pool[specified_selection]
        crossover_func, quantity, crossover_proba = self.crossover_pool[specified_crossover]
        mutation_func, mutation_proba, is_const_mut_rate = self.mutation_pool[specified_mutation]

        while True:
            selected_id = selection_func(self.fitness, np.int64(tour_size), np.int64(quantity))

            offspring = self.population[selected_id]
            offspring_no_mutated = crossover_func(offspring, crossover_proba)

            if len(offspring_no_mutated) < self.limit_depth:
                break

        if is_const_mut_rate:
            mutation_proba = mutation_proba
        else:
            mutation_proba = mutation_proba / len(offspring_no_mutated)

        offspring = mutation_func(offspring_no_mutated, uniset, mutation_proba)
        return offspring

    def get_elite(self):
        elite_pop_size = int(self.pop_size * self.elitism)
        sorted_indices = np.argsort(self.fitness)
        elite_indices = sorted_indices[:elite_pop_size]

        return [self.population[i].copy() for i in elite_indices]

    def get_new_population(self, uniset) -> None:
        get_new_individ = partial(
            self.get_new_individ,
            uniset, self.specified_selection, self.specified_crossover, self.specified_mutation,
        )

        if self.elitism:
            elite_population = self.get_elite()
            new_population = [get_new_individ() for _ in range(self.pop_size - int(self.pop_size * self.elitism))]
            self.population = np.array(elite_population + new_population, dtype=object)

            for idx, tree in enumerate(elite_population):
                info(f'Elite tree {idx}: {tree}')
        else:
            new_population = [get_new_individ() for _ in range(self.pop_size)]
            self.population = np.array(new_population, dtype=object)

        for idx, tree in enumerate(self.population):
            info(f'Get new tree {idx}: {tree}')

    def fit(self, x, y):
        uniset = UniversalSet(x)
        self.get_init_population(uniset)

        for i in range(1, self.n_iters + 1):
            self.__show_progress(i)
            if self.__termination_check():
                break
            else:
                self.from_population_to_fitness(y)
                self.get_new_population(uniset)

        print(f"Best fitness: {self.best_fitness}")
        info(f"Best fitness: {self.best_fitness}")

        print(f"Best solution: {self.best_tree}")
        info(f"Best solution: {self.best_tree}")

        return self.best_tree

    def __show_progress(self, current_iter: int) -> None:
        if self.show_progress_each and current_iter % self.show_progress_each == 0:
            print(f"{current_iter}-th iteration with the best fitness = {self.best_fitness}")
            info(f"{current_iter}-th iteration with the best fitness = {self.best_fitness}")
            info(f"{current_iter}-th iteration with the best solution = {self.best_tree}")

    def __update(self) -> None:
        best_fit_changed = False
        best_idx = None

        for idx, fit in enumerate(self.fitness):
            if fit < self.best_fitness:
                best_fit_changed = True
                best_idx = idx

                self.best_tree = self.population[best_idx].copy()
                self.best_fitness = fit

        if best_fit_changed:
            self.__no_update_counter = 0
            self.temp_tree = self.population[best_idx].copy()
            self.temp_fitness = self.best_fitness
        else:
            self.__no_update_counter += 1

