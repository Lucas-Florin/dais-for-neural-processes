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
        config = cw_config["params"]

        logpath = os.path.dirname(os.path.abspath(__file__))
        logpath = os.path.join(logpath, os.path.join("..", "log"))
        os.makedirs(logpath, exist_ok=True)
        config["logpath"] = logpath

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
            adam_lr=eval(config["adam_lr"]),
            batch_size=config["batch_size"],
            n_samples=config["n_samples"],
        )
        eval_neural_process.train(model, benchmark_meta, benchmark_val, benchmark_test, config)
        #model.load_model(eval(config["n_tasks_train"]))

        x_test, y_test = eval_neural_process.collate_benchmark(benchmark_test)
        x_test = x_test[:config["n_task_meta"], :, :]
        y_test = y_test[:config["n_task_meta"], :, :]

        print(x_test.shape)
        context_distributions = []

        context_sizes = [0, 1, 2, 4, 5, 6, 8, 12, 16]
        #mu_z, var_z = model.aggregator.last_agg_state
        #context_distributions.append((mu_z, var_z))
        for context_size in context_sizes:
            model.adapt(x = x_test[:, :context_size, :], y = y_test[:, :context_size, :])
            mu_z, var_z = model.aggregator.last_agg_state
            context_distributions.append((mu_z, var_z))
        mu_z_target, var_z_target = model.aggregator.last_agg_state
        
        log_likelihood_probs_mc_matrix = eval_neural_process.estimates_over_time(lambda x,z: eval_neural_process.np_decode(model, x, z), (x_test, y_test), context_distributions, (mu_z_target, var_z_target), n_samples = 100)

        print(log_likelihood_probs_mc_matrix.shape)
        with open('log_likelihood_probs_mc_matrix.npy', 'wb') as f:
            np.save(f, log_likelihood_probs_mc_matrix)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass

if __name__ == "__main__":
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MyExperiment)
    #cw = cluster_work.ClusterWork(MyExperiment)
    # this next line is important in order to create a new sweep!
    # RUN!
    cw.run()