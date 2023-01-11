import os
import copy
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sweep_work.create_sweep import create_sweep
from sweep_work.sweep_logger import SweepLogger
from sweep_work.experiment_wrappers import wrap_experiment

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging
from cw2.cw_data.cw_wandb_logger import WandBLogger

from bayesian_meta_learning.lmlhd_estimators import lmlhd_mc, lmlhd_ais, lmlhd_dais, lmlhd_dais_new
from metalearning_benchmarks import MetaLearningBenchmark
from metalearning_benchmarks import benchmark_dict as BM_DICT
from neural_process.neural_process import NeuralProcess


def build_benchmarks(params):
    benchmark_meta = BM_DICT[params["benchmark"]](
        n_task=params["n_task_meta"],
        n_datapoints_per_task=params["n_datapoints_per_task_meta"],
        output_noise=params["data_noise_std"],
        seed_task=params["seed_task_meta"],
        seed_x=params["seed_x_meta"],
        seed_noise=params["seed_noise_meta"],
    )
    benchmark_val = BM_DICT[params["benchmark"]](
        n_task=params["n_task_val"],
        n_datapoints_per_task=params["n_datapoints_per_task_val"],
        output_noise=params["data_noise_std"],
        seed_task=params["seed_task_val"],
        seed_x=params["seed_x_val"],
        seed_noise=params["seed_noise_val"],
    )
    benchmark_test = BM_DICT[params["benchmark"]](
        n_task=params["n_task_test"],
        n_datapoints_per_task=params["n_datapoints_per_task_test"],
        output_noise=params["data_noise_std"],
        seed_task=params["seed_task_test"],
        seed_x=params["seed_x_test"],
        seed_noise=params["seed_noise_test"],
    )
    return benchmark_meta, benchmark_val, benchmark_test

def collate_benchmark(benchmark: MetaLearningBenchmark):
    # collate test data
    x = np.zeros((benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_x))
    y = np.zeros((benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_y))
    for l, task in enumerate(benchmark):
        x[l] = task.x
        y[l] = task.y

    return x, y

def np_decode(np_model, x, z):
    assert x.ndim == 3  # (n_tsk, n_tst, d_x)
    assert z.ndim == 4  # (n_tsk, n_ls, n_marg, d_z)

    mu_y, std_y = np_model.decoder.decode(x, z) # decoder returns std!!

    return mu_y, std_y

def build_model(config, logpath):
    os.makedirs(logpath, exist_ok=True)
    model = NeuralProcess(
        logpath=logpath,
        seed=config["seed"],
        d_x=config["d_x"],
        d_y=config["d_y"],
        d_z=config["d_z"],
        n_context=config["n_context"],
        aggregator_type=config["aggregator_type"],
        loss_type=config["loss_type"],
        input_mlp_std_y=config["input_mlp_std_y"],
        latent_prior_scale=config["latent_prior_scale"],
        f_act=config["f_act"],
        n_hidden_layers=config["n_hidden_layers"],
        n_hidden_units=config["n_hidden_units"],
        decoder_output_scale=config["decoder_output_scale"],
        device=config["device"],
        adam_lr=config["adam_lr"],
        batch_size=config["batch_size"],
        n_samples=config["n_samples"],
        n_annealing_steps=config["dais_n_annealing_steps"],
        dais_step_size=config["dais_step_size"],
        dais_adapt_step_size_to_std_z=config["dais_adapt_step_size_to_std_z"],
        dais_scalar_step_size=config["dais_scalar_step_size"],
        dais_partial=config["dais_partial"],
        dais_partial_gamma=config["dais_partial_gamma"],
        dais_schedule=config["dais_schedule"],
    )

    return model


def plot_examples(
    np_model: NeuralProcess,
    benchmark: MetaLearningBenchmark,
    n_task_plot: int,
    context_set_sizes: Sequence[int],
    mc_task_list: None,
    ais_task_list: None,
    dais_task_list: None,
    dais_new_task_list: None,
    n_samples: int = 3,
    device=None,
):
    
    # determine n_task
    n_task_plot = min(n_task_plot, benchmark.n_task)
    subplot_titles = list()
    for cs in context_set_sizes:
        for t in range(n_task_plot):
            title = ''
            for task_list, name in zip((mc_task_list, ais_task_list, dais_task_list, dais_new_task_list), ('MC', 'AIS', 'DAIS', 'DAIS_new')):
                if task_list is not None:
                    assert len(task_list) > 0
                    ml = task_list[cs][t]
                    title += f'{name}: {ml:.2f}; '
            subplot_titles.append(title)
    
    fig = make_subplots(
        rows=len(context_set_sizes), 
        cols=n_task_plot,
        subplot_titles=subplot_titles
    )

    # evaluate predictions
    x_min = benchmark.x_bounds[0, 0]
    x_max = benchmark.x_bounds[0, 1]
    x_plt_min = x_min - 0.1 * (x_max - x_min)
    x_plt_max = x_max + 0.1 * (x_max - x_min)
    x_plt = np.linspace(x_plt_min, x_plt_max, 128)
    x_plt = np.reshape(x_plt, (-1, 1))

    x_test = np.zeros((n_task_plot, benchmark.n_datapoints_per_task, benchmark.d_x))
    y_test = np.zeros((n_task_plot, benchmark.n_datapoints_per_task, benchmark.d_y))
    for k in range(0, n_task_plot):
        task = benchmark.get_task_by_index(k)
        x_test[k] = task.x
        y_test[k] = task.y

    # plot predictions
    for i in range(len(context_set_sizes)):
        cs = context_set_sizes[i]
        np_model.adapt(x=x_test[:, :cs, :], y=y_test[:, :cs, :]) # adapt model on context set of size cs

        mu_y, var_y = np_model.predict(x=x_plt, n_samples=n_samples)
        std_y = np.sqrt(var_y)
        assert mu_y.shape == (n_task_plot, n_samples, x_plt.shape[0], benchmark.d_y)
        assert std_y.shape == (n_task_plot, n_samples, x_plt.shape[0], benchmark.d_y)

        for l in range(n_task_plot):
            for s in range(n_samples):
                x_sample = x_plt.flatten()
                y_sample = mu_y[l, s, :, :].flatten()
                fig.add_trace(go.Scatter(x=x_sample, 
                                        y=y_sample, 
                                        mode='lines', name='prediction',
                                        legendgroup='prediction', 
                                        showlegend=(s==0 and l==0 and i==len(context_set_sizes)-1),
                                        marker={'color': 'blue', 'opacity': 1.0}), 
                            row=i+1, col=l+1,)
                x_std_sample = np.concatenate((x_sample, x_sample[::-1]))
                y_std_sample = np.concatenate((y_sample, y_sample[::-1])) + np.concatenate((std_y[l, s, :, :].flatten(), 
                                                              - std_y[l, s, :, :].flatten()[::-1]))
                fig.add_trace(
                    go.Scatter(
                        x = x_std_sample,
                        y = y_std_sample,
                        mode='lines',
                        fill='toself',
                        showlegend=False,
                        line_color='rgba(255,255,255,0)',
                        fillcolor='rgba(0,100,80,0.4)',
                    ),
                    row=i+1, col=l+1,
                )
                
            fig.add_trace(go.Scatter(x=x_test[l, cs:, :].flatten(), 
                                     y=y_test[l, cs:, :].flatten(), 
                                     mode='markers', name='test set',
                                     legendgroup='test set', showlegend=(l==0 and i==len(context_set_sizes)-1),
                                     marker={'color': 'green'}), 
                          row=i+1, col=l+1,)
            fig.add_trace(go.Scatter(x=x_test[l, :cs, :].flatten(), 
                                     y=y_test[l, :cs, :].flatten(), 
                                     mode='markers', name='context set',
                                     legendgroup='context set', showlegend=(l==0 and i==len(context_set_sizes)-1),
                                     marker={'color': 'red'}), 
                          row=i+1, col=l+1,)
            
    return fig
    

def copy_sweep_params(source, destination, key_list):
    for key in key_list:
        destination[key] = source[key]
        
        
def complete_config_with_given_total(source, keys, total, mode='product', return_int=True):
    if mode != 'product':
        raise NotImplementedError
    if total is None:
        return
    none_keys = [k for k in keys if source[k] is None]
    if len(none_keys) == 0:
        return
    assert len(none_keys) == 1
    new_value = total
    for k in keys:
        if k not in none_keys:
            new_value /= source[k]

    source[none_keys[0]] = int(new_value) if return_int else new_value
    

class BaselineExperiment(experiment.AbstractExperiment):
    # ...

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Skip for Quickguide
        benchmark_params = config["params"]["benchmark_params"]
        benchmark_meta, benchmark_val, benchmark_test = build_benchmarks(benchmark_params)
        self.benchmark_meta = benchmark_meta
        self.benchmark_val = benchmark_val
        self.benchmark_test = benchmark_test

    def run(self, cw_config: dict, rep: int, logger: cw_logging.AbstractLogger) -> dict:
        params = cw_config["params"]
        model_params = copy.deepcopy(params["model_params"])
        model_params["d_x"] = self.benchmark_meta.d_x
        model_params["d_y"] = self.benchmark_meta.d_y
        model_params["batch_size"] = self.benchmark_meta.n_task
        copy_sweep_params(params, model_params, params['copy_sweep_model_params'])

        model = build_model(model_params, cw_config['_rep_log_path'])
        train_params = copy.deepcopy(params["train_params"])
        
        if train_params['load_model']:
            model.load_model(eval(train_params["n_tasks_train"]), train_params['load_model_path'])
        else:
            def callback(n_meta_tasks_seen, np_model, metrics): 
                if metrics is not None:
                    # logger.process(metrics | {'iter': n_meta_tasks_seen})
                    logger.process(metrics)
            
            model.meta_train(
            benchmark_meta=self.benchmark_meta,
            benchmark_val=self.benchmark_val,
            n_tasks_train=eval(train_params["n_tasks_train"]),
            validation_interval=eval(train_params["validation_interval"]),
            callback=callback,
            )

        # Evaluate
        eval_params = copy.deepcopy(params["eval_params"])
        copy_sweep_params(params, eval_params, params['copy_sweep_eval_params'])
        # assert eval_params['use_mc'] or eval_params['use_ais']
        x_test, y_test = collate_benchmark(self.benchmark_test)
        context_size_list = eval_params["context_sizes"]
        mc_list = list()
        mc_tasks_list = dict()
        ais_list = list()
        ais_tasks_list = dict()
        dais_list = list()
        dais_tasks_list = dict()        
        dais_new_list = list()
        dais_new_tasks_list = dict()     
        for cs in tqdm(context_size_list):
            model.adapt(x = x_test[:, :cs, :], y = y_test[:, :cs, :])
            mu_z, var_z = model.aggregator.last_latent_state
            mu_z, var_z = mu_z.detach(), var_z.detach()
            if eval_params['use_mc']:
                lmlhd_estimate_mc, _ = lmlhd_mc(
                    lambda x,z: np_decode(model, x, z), 
                    (mu_z, var_z), 
                    (x_test, y_test), 
                    eval_params["mc_n_samples"],
                    batch_size=eval_params["mc_batch_size"],
                    seed=eval_params['mc_seed'],
                    subbatch_size=eval_params['mc_subbatch_size']
                )
                mc_tasks_list[cs] = lmlhd_estimate_mc
                mc_list.append(np.median(lmlhd_estimate_mc))
            if eval_params['use_ais']:
                complete_config_with_given_total(eval_params, 
                                                 ['ais_n_samples', 'ais_chain_length', 'ais_n_hmc_steps'], 
                                                 eval_params['ais_total_compute'])
                lmlhd_estimate_ais = lmlhd_ais(
                    lambda x,z: np_decode(model, x, z), 
                    (mu_z, var_z), 
                    (x_test, y_test), 
                    n_samples=eval_params['ais_n_samples'],
                    chain_length=eval_params['ais_chain_length'],
                    device=model_params['device'],
                    num_leapfrog_steps=eval_params['ais_n_hmc_steps'],
                    step_size=eval_params['ais_step_size'],
                    adapt_step_size_to_std_z=eval_params['ais_adapt_step_size_to_std_z'],
                    scalar_step_size=eval_params['ais_scalar_step_size'],
                    seed=eval_params['ais_seed'],
                )
                ais_tasks_list[cs] = (lmlhd_estimate_ais.detach().numpy())
                ais_list.append(np.median(lmlhd_estimate_ais.detach().numpy()))
            if eval_params['use_dais']:
                lmlhd_estimate_dais = lmlhd_dais(
                    lambda x,z: np_decode(model, x, z), 
                    (mu_z, var_z), 
                    (x_test, y_test), 
                    n_samples=eval_params["dais_n_samples"],
                    chain_length=eval_params['dais_chain_length'],
                    device=model_params['device'],
                    # num_leapfrog_steps=eval_params['dais_n_hmc_steps'],
                    step_size=eval_params['dais_step_size'],
                    adapt_step_size_to_std_z=eval_params['dais_adapt_step_size_to_std_z'],
                    scalar_step_size=eval_params['dais_scalar_step_size'],
                    partial=eval_params['dais_partial'],
                    schedule=eval_params['dais_schedule'],
                    batch_size=eval_params['dais_batch_size'],
                    clip_grad=eval_params['dais_clip_grad'],
                    seed=eval_params['dais_seed'],
                )
                dais_tasks_list[cs] = (lmlhd_estimate_dais.detach().numpy())
                dais_list.append(np.median(lmlhd_estimate_dais.detach().numpy()))
            if eval_params['use_dais_new']:
                # complete_config_with_given_total(eval_params, 
                #                                  ['ais_n_samples', 'ais_chain_length', 'ais_n_hmc_steps'], 
                #                                  eval_params['ais_total_compute'])
                lmlhd_estimate_dais_new = lmlhd_dais_new(
                    lambda x,z: np_decode(model, x, z), 
                    (mu_z, var_z), 
                    (x_test, y_test), 
                    n_samples=eval_params['dais_new_n_samples'],
                    chain_length=eval_params['dais_new_chain_length'],
                    device=model_params['device'],
                    num_leapfrog_steps=eval_params['dais_new_n_hmc_steps'],
                    step_size=eval_params['dais_new_step_size'],
                    step_size_update_factor=eval_params['dais_new_step_size_update_factor'],
                    target_accept_rate=eval_params['dais_new_target_accept_rate'],
                    clip_grad=eval_params['dais_new_clip_grad'],
                    adapt_step_size=eval_params['dais_new_adapt_step_size'],
                    adapt_step_size_to_std_z=eval_params['dais_new_adapt_step_size_to_std_z'],
                    scalar_step_size=eval_params['dais_new_scalar_step_size'],
                    do_accept_reject_step=eval_params['dais_new_do_accept_reject_step'],
                    use_accept_hist=eval_params['dais_new_use_accept_hist'],
                    seed=eval_params['dais_new_seed'],
                )
                dais_new_tasks_list[cs] = (lmlhd_estimate_dais_new.detach().numpy())
                dais_new_list.append(np.median(lmlhd_estimate_dais_new.detach().numpy()))
        result = dict()
        if eval_params['use_mc']:
            mc_objective = np.mean(mc_list)
            print(f'Objective list: ')
            print(mc_list)
            print("Objective: " + str(mc_objective))
            result.update({
                "mc_objective": mc_objective,
                "mc_objective_list": mc_list,
            })
            for l in logger:
                if type(l) is WandBLogger:
                    l.log_plot(context_size_list, mc_list, ["context_size", "MC objective"], 'mc_plot', 'MC estimator for LL')


        if eval_params['use_ais']:
            ais_objective = np.mean(ais_list)
            print(f'AIS Objective list: ')
            print(ais_list)
            print("AIS Objective: " + str(ais_objective))
            result.update({
                "ais_objective": ais_objective,
                "ais_objective_list": ais_list,
            })
            for l in logger:
                if type(l) is WandBLogger:
                    l.log_plot(context_size_list, ais_list, ["context_size", "AIS objective"], 'ais_plot', 'AIS estimator for LL')
        
        if eval_params['use_dais']:
            dais_objective = np.mean(dais_list)
            print(f'DAIS Objective list: ')
            print(dais_list)
            print("DAIS Objective: " + str(dais_objective))
            result.update({
                "dais_objective": dais_objective,
                "dais_objective_list": dais_list,
            })
            for l in logger:
                if type(l) is WandBLogger:
                    l.log_plot(context_size_list, dais_list, ["context_size", "DAIS objective"], 'dais_plot', 'DAIS estimator for LL')
                    
        if eval_params['use_dais_new']:
            dais_new_objective = np.mean(dais_new_list)
            print(f'DAIS_new Objective list: ')
            print(dais_new_list)
            print("DAIS_new Objective: " + str(dais_new_objective))
            result.update({
                "dais_new_objective": dais_new_objective,
                "dais_new_objective_list": dais_new_list,
            })
            for l in logger:
                if type(l) is WandBLogger:
                    l.log_plot(context_size_list, dais_list, ["context_size", "DAIS_new objective"], 
                               'dais_new_plot', 'DAIS_new estimator for LL')
                    
        if eval_params['show_examples']:
            fig = plot_examples(
                model,
                self.benchmark_test,
                eval_params['example_num_tasks'],
                eval_params['example_context_set_sizes'],
                mc_task_list=None if len(mc_tasks_list) == 0 else mc_tasks_list,
                ais_task_list=None if len(ais_tasks_list) == 0 else ais_tasks_list,
                dais_task_list=None if len(dais_tasks_list) == 0 else dais_tasks_list,
                dais_new_task_list=None if len(dais_new_tasks_list) == 0 else dais_new_tasks_list,
                n_samples=eval_params['example_n_samples'],
                device=model_params['device'],
            )
            result.update({
                'examples': fig
            })
        
        logger.process(result)
        if train_params['save_model']:
            model.save_model()
        
    
    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


def main_sweepwork():
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(wrap_experiment(BaselineExperiment))
    #cw = cluster_work.ClusterWork(MyExperiment)
    # this next line is important in order to create a new sweep!
    create_sweep(cw)
    # Sweepwork expects that 1 SweepLogger is present. Additional other loggers should not be a problem
    
    cw.add_logger(SweepLogger())

    # RUN!
    cw.run()   


def main():
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(BaselineExperiment)

    
    cw.add_logger(WandBLogger())

    # RUN!
    cw.run()

if __name__ == "__main__":
    # main_sweepwork()
    main()
