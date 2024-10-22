from src.utils.randoms import random_weighted_sample
from src.utils.logs import info
from numpy.typing import NDArray
import numpy as np


def proportional_selection(fitness: NDArray[np.float64], tour_size: np.int64, quantity: np.int64) -> NDArray:
    info(f"Calculating proportional selection probabilities")
    inverse_fitness = 1 / (fitness + 1e-10)

    return random_weighted_sample(inverse_fitness, quantity)


def rank_selection(fitness: NDArray[np.float64], tour_size: np.int64, quantity: np.int64) -> NDArray:
    info(f"Calculating rank selection probabilities")
    ranks = np.argsort(np.argsort(-fitness))

    return random_weighted_sample(ranks, quantity)


def tournament_selection(fitness: NDArray[np.float64], tour_size: np.int64, quantity: np.int64) -> NDArray:
    to_return = np.zeros(quantity, dtype=np.int64)
    indexes = np.arange(len(fitness))

    for i in range(quantity):

        tournament = np.random.choice(indexes, tour_size, replace=False)
        winner_index = tournament[np.argmin(fitness[tournament])]
        to_return[i] = winner_index

        winner_pos = np.where(indexes == winner_index)[0][0]
        indexes = np.delete(indexes, winner_pos)

    return to_return
