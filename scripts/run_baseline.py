import numpy as np
import os
import copy
from pathlib import Path

from sweep_work.create_sweep import create_sweep
from sweep_work.sweep_logger import SweepLogger
from sweep_work.experiment_wrappers import wrap_experiment

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging
from cw2.cw_data.cw_wandb_logger import WandBLogger

from bayesian_meta_learning.lmlhd_estimators import lmlhd_mc
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
    )

    return model

class MyExperiment(experiment.AbstractExperiment):
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
        model.meta_train(
        benchmark_meta=self.benchmark_meta,
        benchmark_val=self.benchmark_val,
        n_tasks_train=eval(train_params["n_tasks_train"]),
        validation_interval=eval(train_params["validation_interval"]),
        callback=None,
        )

        # Evaluate
        eval_params = params["eval_params"]
        x_test, y_test = collate_benchmark(self.benchmark_test)
        context_size_list = eval_params["context_sizes"]
        objective_list = list()
        for cs in context_size_list:
            model.adapt(x = x_test[:, :cs, :], y = y_test[:, :cs, :])
            mu_z, var_z = model.aggregator.last_agg_state
            lmlhd_estimate_mc, _ = lmlhd_mc(
                lambda x,z: np_decode(model, x, z), 
                (mu_z, var_z), 
                (x_test, y_test), 
                eval_params["n_mc_samples"],
                batch_size=eval_params["batch_size"],
            )
            objective_list.append(np.median(lmlhd_estimate_mc))
        objective = np.mean(objective_list)
        print(f'Objective list: ')
        print(objective_list)
        print("Objective: " + str(objective))
        result =  {
            "objective": objective,
            "objective_list": objective_list,
        }
        logger.process(result)
        for l in logger:
            if type(l) is WandBLogger:
                l.log_plot(context_size_list, objective_list, ["context_size", "objective"])
    
    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


def main_sweepwork():
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(wrap_experiment(MyExperiment))
    #cw = cluster_work.ClusterWork(MyExperiment)
    # this next line is important in order to create a new sweep!
    create_sweep(cw)
    # Sweepwork expects that 1 SweepLogger is present. Additional other loggers should not be a problem
    
    cw.add_logger(SweepLogger())

    # RUN!
    cw.run()   


def main():
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MyExperiment)

    
    cw.add_logger(WandBLogger())

    # RUN!
    cw.run()

if __name__ == "__main__":
    # main_sweepwork()
    main()