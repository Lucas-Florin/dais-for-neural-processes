import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections.abc import Sequence
import copy


from baseline import build_benchmarks, collate_benchmark, build_model
from metalearning_benchmarks import MetaLearningBenchmark
from neural_process.neural_process import NeuralProcess
from cw2.cw_config import cw_config



def plot_benchmark_examples(benchmark, tasks=None, figsize=(8,4), legend_width=0.8):
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
            ax.plot(x_plt, y_true, label='Ground truth function')
            ax.fill_between(x_plt, y_true - std, y_true + std, alpha=0.2, label='Noise standard deviation')
            ax.plot(x, y, 'o', markersize=3, label='Sampled dataset points')
    axs[nrows - 1, 0].legend( 
        bbox_to_anchor=((1 - legend_width) / 2, 0., legend_width, 0.),
        bbox_transform=fig.transFigure,
        loc='upper left',
        mode='expand', 
        ncol=ncols,
        borderaxespad=0.,
    )
    fig.tight_layout() 
    return fig, axs



def plot_examples(
    np_model: NeuralProcess,
    benchmark: MetaLearningBenchmark,
    n_task_plot: int,
    context_set_sizes: Sequence[int],
    mc_tasks_list: None,
    ais_tasks_list: None,
    dais_tasks_list: None,
    dais_new_tasks_list: None,
    n_samples: int = 3,
    sample_alpha=1.0,
    include_evaluation_method_names=False,
    figsize=(8,4),
    plot_output_std=False,
    tasks=None,
    device=None,
):
    
    # determine n_task
    n_task_plot = min(n_task_plot, benchmark.n_task)
    if tasks is None:
        tasks = list(range(n_task_plot))
    else: 
        assert len(tasks) == n_task_plot
    for t in tasks:
        assert t < benchmark.n_task
    
    subplot_titles = dict()
    for cs in context_set_sizes:
        for t in tasks:
            title = ''
            for task_list, name in zip(
                (mc_tasks_list, ais_tasks_list, dais_tasks_list, dais_new_tasks_list), 
                ('LW', 'AIS', 'DAIS', 'DAIS_new')):
                if task_list is not None:
                    assert len(task_list) > 0
                    ml = task_list[cs][t]
                    if include_evaluation_method_names:
                        title += f'{name}='
                    num_decimals = min(2, max(0, 3 - np.trunc(np.log10(np.abs(ml))).astype(int)))
                    title += f'{ml:.{num_decimals}f}; '
            subplot_titles[(cs, t)] = title
    fig, axs = plt.subplots(
        n_task_plot,
        len(context_set_sizes),
        sharex=True, 
        sharey=True,
        figsize=figsize,
    )
    # fig = make_subplots(
    #     rows=len(context_set_sizes), 
    #     cols=n_task_plot,
    #     subplot_titles=subplot_titles
    # )

    # evaluate predictions
    x_min = benchmark.x_bounds[0, 0]
    x_max = benchmark.x_bounds[0, 1]
    x_plt_min = x_min - 0.1 * (x_max - x_min)
    x_plt_max = x_max + 0.1 * (x_max - x_min)
    x_plt = np.linspace(x_plt_min, x_plt_max, 128)
    x_plt = np.reshape(x_plt, (-1, 1))

    x_test, y_test = collate_benchmark(benchmark)   
    y_min, y_max = 0, 0
    # plot predictions
    for i in range(len(context_set_sizes)):
        cs = context_set_sizes[i]
        np_model.adapt(x=x_test[:, :cs, :], y=y_test[:, :cs, :]) # adapt model on context set of size cs
        mu_y, var_y = np_model.predict(x=x_plt, n_samples=n_samples)
        std_y = np.sqrt(var_y)
        assert mu_y.shape == (benchmark.n_task, n_samples, x_plt.shape[0], benchmark.d_y)
        assert std_y.shape == (benchmark.n_task, n_samples, x_plt.shape[0], benchmark.d_y)
        for l in range(n_task_plot):
            task = tasks[l]
            if len(axs.shape) == 2:
                ax = axs[l, i]
            else:
                assert len(axs.shape) == 1
                ax = axs[i]
            ax.set(
                title=subplot_titles[(cs, task)], 
                # aspect=0.3, 
                # xticks=[], 
                # yticks=[],
            )
            y_true = benchmark(x_plt, benchmark.params[task])
            y_min = min(y_min, y_true.min())
            y_max = max(y_max, y_true.max())
            label_prediction = 'Prediction'
            label_std = 'Standard deviation'
            for s in range(n_samples):
                x_sample = x_plt.flatten()
                y_sample = mu_y[task, s, :, :].flatten()
                ax.plot(x_sample, y_sample, label=label_prediction, color='blue', linewidth=1.0, alpha=sample_alpha)
                if plot_output_std:
                    std_y_sample = std_y[task, s, :, :].flatten()
                    ax.fill_between(
                        x_sample, 
                        y_sample - std_y_sample, 
                        y_sample + std_y_sample, 
                        color='blue',
                        alpha=0.2, 
                        label=label_std)
                label_prediction, label_std = None, None
            ax.plot(x_plt, y_true, label='Ground truth function', color='xkcd:orange', linewidth=3.0)
            ax.plot(
                x_test[task, cs:, :].flatten(), 
                y_test[task, cs:, :].flatten(), 
                'o', 
                color='green',
                markersize=3, 
                label='Test set')
            ax.plot(
                x_test[task, :cs, :].flatten(), 
                y_test[task, :cs, :].flatten(), 
                'o', 
                color='red',
                markersize=3, 
                label='Context set')            
            
    fig.tight_layout() 
    legend_cols = 5 if plot_output_std else 4
    legend_width = 0.8
    if len(axs.shape) == 2:
        ax = axs[len(context_set_sizes) - 1, 0]
    else:
        assert len(axs.shape) == 1
        ax = axs[len(context_set_sizes) - 1]
    # ax = axs[len(context_set_sizes) - 1, 0]
    ax.legend( 
        bbox_to_anchor=((1 - legend_width) / 2, 0., legend_width, 0.),
        bbox_transform=fig.transFigure,
        loc='upper left',
        mode='expand', 
        ncol=legend_cols,
        borderaxespad=0.,
    )
    ax.set_ylim((y_min - 2, y_max + 2))
    return fig, axs


def load_cw_config(config_file, experiment_name):
    config = cw_config.Config(config_file, experiment_name)
    config_dict = config.exp_configs[0]
    return config_dict

def prepare_model_for_plotting(config_file, experiment_name):
    config_dict = load_cw_config(config_file, experiment_name)
    params = config_dict["params"]

    benchmark_params = params['benchmark_params']
    benchmark_meta, benchmark_val, benchmark_test = build_benchmarks(benchmark_params)
    benchmark = benchmark_test
    model_params = copy.deepcopy(params["model_params"])
    model_params["d_x"] = benchmark_meta.d_x
    model_params["d_y"] = benchmark_meta.d_y
    model_params["batch_size"] = benchmark_meta.n_task

    model = build_model(model_params, config_dict['_rep_log_path'])
    train_params = copy.deepcopy(params["train_params"])
    train_params['load_model_path'] = '.' + train_params['load_model_path']
    model.load_model(eval(train_params["n_tasks_train"]), train_params['load_model_path'])
    return model, benchmark, params


def plot_ml_over_css(runs, fig=None, ax=None, legend_prefix='', x_axis_offset=0.0, color='blue', 
                     use_mc=True, use_ais=True, figsize=(7, 4), legend_width=0.8, 
                     symlog=None, additional_ticks=None, final=False):
    n_runs = len(runs)
    context_set_sizes = np.array([1, 3, 9], dtype=float) + x_axis_offset
    n_css = len(context_set_sizes)
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)

    metrics = dict()
    if use_mc:
        metrics['mc'] = 'likelihood weighting'
    if use_ais:
        metrics['ais'] = 'AIS'
    if len(runs) > 1:
        assert len(runs) == 5
        for metric in metrics.keys():
            css = context_set_sizes
            objective_list = np.zeros((n_runs, n_css))

            for i, run in enumerate(runs):
                summary = run.summary
                objective_list[i, :] = summary[f'{metric}_objective_list']

            median_i = np.argsort(objective_list.mean(1))[n_runs // 2]
            median = objective_list[median_i, :]
            objective_max = objective_list.max(0)
            objective_min = objective_list.min(0)
            
            error = np.stack([
                -(objective_min - median),
                objective_max - median,
            ])
            if symlog is not None:
                css, median, _, error = add_symlog_border(ax, css, median, symlog, final, 
                                                                    error=error)
            
            linestyle = 'dashed' if metric == 'ais' else None
            i = 1 if metric == 'ais' else 0
            eb = ax.errorbar(css + 0.05 * i, median, error, 
                        label=f'{legend_prefix}{metrics[metric]}', linestyle=linestyle, color=color)
            if linestyle is not None:
                eb[-1][0].set_linestyle(linestyle)

    elif len(runs) == 1:
        run = runs[0]
        summary = run.summary
        for metric in metrics.keys():
            css = context_set_sizes
            objective_list = np.array(summary[f'{metric}_objective_list'])
            markevery = None
            if symlog is not None:
                css, objective_list, markevery = add_symlog_border(ax, css, objective_list, symlog, final)
            linestyle = 'dashed' if metric == 'ais' else None
            i = 1 if metric == 'ais' else 0
            ax.plot(css + 0.05 * i, objective_list, 
                    label=f'{legend_prefix}{metrics[metric]}', linestyle=linestyle, 
                    marker='o', markevery=markevery, color=color)


    else:
        raise ValueError

    ax.set_ylabel('Log predictive likelihood')
    ax.set_xlabel('Context set size')
    
    ax.legend( 
        bbox_to_anchor=((1 - legend_width) / 2, 0., legend_width, 0.),
        bbox_transform=fig.transFigure,
        loc='upper left',
        mode='expand', 
        ncol=2,
        borderaxespad=0.,
    )
    if symlog is not None:
        set_symlog_scale(ax, symlog, additional_ticks)
    
    return fig, ax
    
    
def plot_ml_over_compute(runs, metric, fig=None, ax=None, color='blue', true_ml=None, 
                         figsize=(7, 4), ylim=None, symlog=None, additional_ticks=None, final=False,
                         max_ml=False, label=None, legend_width=0.8, marker='o'):
    n_runs = len(runs)
    
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    metrics_labels = {
        'mc': 'Likelihood weighting',
        'ais': 'AIS',
        'dais': 'DAIS',
    }
    metrics_list = list(metrics_labels.keys()) if max_ml else [metric]
    objectives = dict()
    computes = dict()
    for metric in metrics_list:
        objective, compute = list(), list()
        objective_name = f'{metric}_objective'
        for i, run in enumerate(runs):
            summary = run.summary
            run_eval_config = run.config
            if objective_name in summary:
                objective.append(summary[objective_name])
                if metric == 'mc':
                    c = run_eval_config['eval_params']['mc_n_samples']
                elif metric == 'ais': 
                    c = 1
                    c *= run_eval_config['eval_params']['ais_n_samples']
                    c *= run_eval_config['eval_params']['ais_chain_length']
                    c *= run_eval_config['eval_params']['ais_n_hmc_steps']
                elif metric == 'dais':
                    c = 1
                    c *= run_eval_config['eval_params']['dais_n_samples']
                    c *= run_eval_config['eval_params']['dais_chain_length']
                else:
                    raise ValueError  
                compute.append(c)
        objective, compute = np.array(objective), np.array(compute)            
        new_order = np.argsort(compute)
        compute = compute[new_order]
        objective = objective[new_order]
        if true_ml is not None:
            objective -= true_ml
        objectives[metric] = objective
        computes[metric] = compute
    if max_ml:
        lens = [len(computes[experiment_name]) for experiment_name in computes.keys()]
        objective = np.zeros((max(*lens), ))
        compute = np.zeros_like(objective)
        for i in range(len(objective)):
            computes_i = get_list_in_dict_value(computes, i, reverse=True)
            assert np.all(np.array(computes_i) == computes_i[0])
            compute[-1-i] = computes_i[0]
            objective[-1-i] = max(get_list_in_dict_value(objectives, i, reverse=True))
        
    markevery = None
    if symlog is not None:
        compute, objective, markevery = add_symlog_border(ax, compute, objective, symlog, final)
    label = label if max_ml else metrics_labels[metric]
    ax.plot(compute, objective, label=label, 
            marker=marker, markevery=markevery, color=color)

    ax.set_xscale('log', base=10)
    ax.set_ylabel('Log predictive likelihood estimate')
    ax.set_xlabel('Number of decoder evaluations')
    ax.legend( 
        bbox_to_anchor=((1 - legend_width) / 2, 0., legend_width, 0.),
        bbox_transform=fig.transFigure,
        loc='upper left',
        mode='expand', 
        ncol=3,
        borderaxespad=0.,
    )
    
    assert ylim is None or symlog is None
    if ylim is not None:
        ax.set_ylim(ylim)
    if symlog is not None and final:
        set_symlog_scale(ax, symlog, additional_ticks)
    
    return fig, ax
    

def get_list_in_dict_value(dict_with_lists, index, reverse=False):
    assert index >= 0
    min_len = index + 1
    i = -1 - index if reverse else index
    values_at_index = [
        dict_with_lists[experiment_name][i] 
        for experiment_name in dict_with_lists.keys() 
        if len(dict_with_lists[experiment_name]) >= min_len
    ]
    return values_at_index


def set_symlog_scale(ax, linthresh, additional_ticks=None):
    assert linthresh > 0
    subs = np.arange(2, 10)
    ax.set_yscale('symlog', base=10, linthresh=linthresh, subs=subs)
    if additional_ticks is not None:
        additional_ticks = np.array(additional_ticks)
        additional_ticks = np.concatenate((-additional_ticks[::-1], [0.], additional_ticks))
        ticks = ax.get_yticks()
        ticks = ticks[ticks < min(additional_ticks)]
        ticks = np.concatenate([ticks, additional_ticks])
        labels = ax.get_yticklabels()
        labels = [str(int(t)) for t in ticks]
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    

def add_symlog_border(ax, x_values, y_values, linthresh, final, error=None):
    markevery = None
    if np.any(y_values < -linthresh) and np.any(y_values > -linthresh):
        i = np.where(y_values > -linthresh)[0][0]
        c = x_values[i-1] + (x_values[i] - x_values[i-1]) * (-linthresh - y_values[i-1]) / (y_values[i] - y_values[i-1])
        y_values = np.concatenate([y_values[:i], [-linthresh], y_values[i:]])
        x_values = np.concatenate([x_values[:i], [c], x_values[i:]])
        if error is not None:
            error = np.concatenate([error[:, :i], [[0.], [0.]], error[:, i:]], 1)
        markevery = np.ones(x_values.shape, dtype=bool)
        markevery[i] = False
    if final:
        l = mlines.Line2D([x_values.min(), x_values.max()], [-linthresh, -linthresh], 
                            linestyle=':', color='gray', linewidth=1.0)
        ax.add_line(l)
    if error is not None:
        return x_values, y_values, markevery, error
    return x_values, y_values, markevery
    
    
def generate_resutlts_table_row(runs):
    
    metric_names = ['mc', 'ais', 'dais']
    latex_table_row = ''
    
    assert len(runs) > 1
    for metric in metric_names:
        objective_list = list()

        for run in runs:
            summary = run.summary
            metric_key = f'{metric}_objective_list'
            if metric_key in summary:
                objective_list.append(summary[metric_key])
        if len(objective_list) == 0:
            continue
        assert len(objective_list) == 5
        objective_list = np.array(objective_list)
        assert len(objective_list.shape) == 2
        assert objective_list.shape == (5, 3)
        objective_mean_over_css = objective_list.mean(1)
        median = np.median(objective_mean_over_css)
        objective_max = objective_mean_over_css.max()
        objective_min = objective_mean_over_css.min()
        diff_max = objective_max - median
        diff_min = median - objective_min
        latex_cell = f'${median:.2f}^{{+{diff_max:.2f}}}_{{-{diff_min:.2f}}}$'

        latex_table_row += ' & ' + latex_cell

    return latex_table_row

