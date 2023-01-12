import numpy as np
import matplotlib.pyplot as plt
from baseline import build_benchmarks, collate_benchmark


def plot_benchmark_examples(benchmark, tasks=None, figsize=(8,4)):
    x_min = benchmark.x_bounds[0, 0]
    x_max = benchmark.x_bounds[0, 1]
    x_plt_min = x_min - 0.1 * (x_max - x_min)
    x_plt_max = x_max + 0.1 * (x_max - x_min)
    x_plt = np.linspace(x_plt_min, x_plt_max, 128)
    x_test, y_test = collate_benchmark(benchmark)

    tasks = list(range(6)) if tasks is None else tasks
    nrows = 2
    ncols = len(tasks) // nrows
    figsize = (4 * ncols, 2.5 * nrows) if figsize is None else figsize
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    for r in range(nrows):
        for c in range(ncols):
            i = r * ncols + c
            ax = axs[r, c]
            task = tasks[i]
            std = benchmark.output_noise
            x, y = [i[task, ...].flatten() for i in [x_test, y_test]]
            y_true = benchmark(x_plt, benchmark.params[task])
            ax.plot(x_plt, y_true, label='Function')
            ax.fill_between(x_plt, y_true - std, y_true + std, alpha=0.2, label='Standard deviation')
            ax.plot(x, y, 'o', markersize=3, label='Dataset points')
    axs[nrows - 1, 0].legend( 
        bbox_to_anchor=(0.2, 0., 0.6, 0.),
        bbox_transform=fig.transFigure,
        loc='upper left',
        mode='expand', 
        ncol=ncols,
        borderaxespad=0.,
    )
    return fig, axs