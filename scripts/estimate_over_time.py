from sweep_work.create_sweep import create_sweep
from sweep_work.sweep_logger import SweepLogger
from sweep_work.experiment_wrappers import wrap_experiment

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging
import numpy as np
import os

from bayesian_meta_learning.lmlhd_estimators import lmlhd_mc
from metalearning_benchmarks import MetaLearningBenchmark
from metalearning_benchmarks import benchmark_dict as BM_DICT
from neural_process.neural_process import NeuralProcess
import eval_neural_process

class MyExperiment(experiment.AbstractExperiment):

    def initialize(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        logpath = os.path.dirname(os.path.abspath(__file__))
        logpath = os.path.join(logpath, os.path.join("..", "log"))
        os.makedirs(logpath, exist_ok=True)

        ## config
        config = dict()
        # model and benchmark
        config["model"] = "StandardNP"
        config["benchmark"] = "Quadratic1D"
        # logging
        config["logpath"] = logpath
        # seed
        config["seed"] = 1234
        # meta data
        config["data_noise_std"] = 0.1
        config["n_task_meta"] = 256
        config["n_datapoints_per_task_meta"] = 64
        config["seed_task_meta"] = 1234
        config["seed_x_meta"] = 2234
        config["seed_noise_meta"] = 3234
        # validation data
        config["n_task_val"] = 16 
        config["n_datapoints_per_task_val"] = 64
        config["seed_task_val"] = 1236
        config["seed_x_val"] = 2236
        config["seed_noise_val"] = 3236
        # test data
        config["n_task_test"] = 256 
        config["n_datapoints_per_task_test"] = 64
        config["seed_task_test"] = 1235
        config["seed_x_test"] = 2235
        config["seed_noise_test"] = 3235

        # generate benchmarks
        benchmark_meta = BM_DICT[config["benchmark"]](
            n_task=config["n_task_meta"],
            n_datapoints_per_task=config["n_datapoints_per_task_meta"],
            output_noise=config["data_noise_std"],
            seed_task=config["seed_task_meta"],
            seed_x=config["seed_x_meta"],
            seed_noise=config["seed_noise_meta"],
        )
        benchmark_val = BM_DICT[config["benchmark"]](
            n_task=config["n_task_val"],
            n_datapoints_per_task=config["n_datapoints_per_task_val"],
            output_noise=config["data_noise_std"],
            seed_task=config["seed_task_val"],
            seed_x=config["seed_x_val"],
            seed_noise=config["seed_noise_val"],
        )
        benchmark_test = BM_DICT[config["benchmark"]](
            n_task=config["n_task_test"],
            n_datapoints_per_task=config["n_datapoints_per_task_test"],
            output_noise=config["data_noise_std"],
            seed_task=config["seed_task_test"],
            seed_x=config["seed_x_test"],
            seed_noise=config["seed_noise_test"],
        )

        # architecture
        config["d_x"] = benchmark_meta.d_x
        config["d_y"] = benchmark_meta.d_y
        config["d_z"] = 16
        config["aggregator_type"] = "BA"
        config["loss_type"] = "MC"
        config["input_mlp_std_y"] = ""
        config["f_act"] = "relu"
        config["n_hidden_layers"] = 2
        config["n_hidden_units"] = 64
        config["latent_prior_scale"] = 1.0
        config["decoder_output_scale"] = config["data_noise_std"]

        # training
        config["n_tasks_train"] = int(2**19)
        config["validation_interval"] = config["n_tasks_train"] // 4
        config["device"] = "cpu"
        config["adam_lr"] = 1e-4
        config["batch_size"] = config["n_task_meta"]
        config["n_samples"] = 16
        config["n_context"] = [1, config["n_datapoints_per_task_meta"] // 2,]

        # generate NP model
        model = NeuralProcess(
            logpath=config["logpath"],
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

        model.load_model(config["n_tasks_train"])

        x_test, y_test = eval_neural_process.collate_benchmark(benchmark_test)
        context_distributions = []
        #mu_z, var_z = model.aggregator.last_agg_state
        #context_distributions.append((mu_z, var_z))
        for i in range(6):
            model.adapt(x = x_test[:, :(2 ** i), :], y = y_test[:, :(2 ** i), :])
            mu_z, var_z = model.aggregator.last_agg_state
            context_distributions.append((mu_z, var_z))
        mu_z_target, var_z_target = model.aggregator.last_agg_state
        
        eval_neural_process.estimates_over_time(lambda x,z: eval_neural_process.np_decode(model, x, z), (x_test, y_test), context_distributions, (mu_z_target, var_z_target), n_samples = 23000)
    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass

if __name__ == "__main__":
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MyExperiment)
    #cw = cluster_work.ClusterWork(MyExperiment)
    # this next line is important in order to create a new sweep!
    # RUN!
    cw.run()