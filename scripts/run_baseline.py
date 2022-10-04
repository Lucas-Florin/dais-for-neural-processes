import numpy as np
import os
import copy
from pathlib import Path
from tqdm import tqdm

from sweep_work.create_sweep import create_sweep
from sweep_work.sweep_logger import SweepLogger
from sweep_work.experiment_wrappers import wrap_experiment

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging
from cw2.cw_data.cw_wandb_logger import WandBLogger

from bayesian_meta_learning.lmlhd_estimators import lmlhd_mc, lmlhd_ais, lmlhd_dais
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

    x = np_model._normalize_x(x)
    mu_y, std_y = np_model.decoder.decode(x, z) # decoder returns std!!
    mu_y = np_model._denormalize_mu_y(mu_y)

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
    )

    return model

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

        model = build_model(model_params, cw_config['_rep_log_path'])
        train_params = params["train_params"]
        
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
        eval_params = params["eval_params"]
        assert eval_params['use_mc'] or eval_params['use_ais']
        x_test, y_test = collate_benchmark(self.benchmark_test)
        context_size_list = eval_params["context_sizes"]
        mc_list = list()
        ais_list = list()
        dais_list = list()
        for cs in tqdm(context_size_list):
            model.adapt(x = x_test[:, :cs, :], y = y_test[:, :cs, :])
            mu_z, var_z = model.aggregator.last_agg_state
            if eval_params['use_mc']:
                lmlhd_estimate_mc, _ = lmlhd_mc(
                    lambda x,z: np_decode(model, x, z), 
                    (mu_z, var_z), 
                    (x_test, y_test), 
                    eval_params["mc_n_samples"],
                    batch_size=eval_params["mc_batch_size"],
                )
                mc_list.append(np.median(lmlhd_estimate_mc))
            if eval_params['use_ais']:
                lmlhd_estimate_ais = lmlhd_ais(
                    lambda x,z: np_decode(model, x, z), 
                    (mu_z, var_z), 
                    (x_test, y_test), 
                    n_samples=eval_params["ais_n_samples"],
                    chain_length=eval_params['ais_chain_length'],
                    device=model_params['device'],
                    num_leapfrog_steps=eval_params['ais_n_hmc_steps'],
                    step_size=eval_params['ais_step_size']
                )
                ais_list.append(np.median(lmlhd_estimate_ais))
            if eval_params['use_dais']:
                lmlhd_estimate_dais = lmlhd_dais(
                    lambda x,z: np_decode(model, x, z), 
                    (mu_z, var_z), 
                    (x_test, y_test), 
                    n_samples=eval_params["dais_n_samples"],
                    chain_length=eval_params['dais_chain_length'],
                    device=model_params['device'],
                    # num_leapfrog_steps=eval_params['dais_n_hmc_steps'],
                    step_size=eval_params['dais_step_size']
                )
                dais_list.append(np.median(lmlhd_estimate_dais))
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
            print("AIS Objective: " + str(dais_objective))
            result.update({
                "ais_objective": dais_objective,
                "ais_objective_list": dais_list,
            })
            for l in logger:
                if type(l) is WandBLogger:
                    l.log_plot(context_size_list, dais_list, ["context_size", "DAIS objective"], 'dais_plot', 'DAIS estimator for LL')
        
        logger.process(result)
        
    
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