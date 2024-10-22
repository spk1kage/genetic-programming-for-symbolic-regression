from src.base.gp import GeneticProgrammingSymbolRegression as GP
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    def fitness_function(x):
        return np.sin(3 * x[:, 0]) * x[:, 0] * 0.5
        # return np.sin(x[:, 0])


    n_dimension = 1
    left_border = -4.5
    right_border = 4.5
    sample_size = 100

    x = np.array([np.linspace(left_border, right_border, sample_size)
                  for _ in range(n_dimension)]).T
    y = fitness_function(x)

    gp = GP(
        n_iters=500,
        pop_size=100,
        elitism=None,
        max_depth=5,
        limit_depth=8,
        metric='RMSE',
        selection='tournament_k',
        tour_size=15,
        crossover='standard',
        crossover_rate=1.0,
        mutation='grow',
        mutation_rate=0.45,
        is_const_mut_rate=True,
        is_logging=False,
        show_progress_each=10,
        termination=150,
        random_state=None
    )

    ind = gp.fit(x[:, 0], y)
    y_pred = ind.genotype() * np.ones(sample_size)

    fig, ax = plt.subplots(figsize=(15, 8), ncols=2, nrows=1)
    ax[0].plot(x[:, 0], y, label="True y")
    ax[0].plot(x[:, 0], y_pred, label="Predict y")
    ax[0].legend()

    ind.plot(ax=ax[1])

    plt.tight_layout()
    plt.show()
