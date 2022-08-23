import numpy as np
import os
import torch
import wandb
import yaml

from bayesian_meta_learning.lmlhd_estimators import lmlhd_mc
from metalearning_benchmarks import MetaLearningBenchmark
from metalearning_benchmarks import benchmark_dict as BM_DICT
from neural_process.neural_process import NeuralProcess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    benchmark_meta, benchmark_val, benchmark_test = build_benchmarks('scripts/benchmark_config.yaml')

    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        config["d_x"] = benchmark_meta.d_x
        config["d_y"] = benchmark_meta.d_y
        config["batch_size"] = benchmark_meta.n_task

        model = build_model(config)

        train_model(model=model, benchmark_meta=benchmark_meta, benchmark_val=benchmark_val, benchmark_test=benchmark_test, config=config)

        x_test, y_test = collate_benchmark(benchmark_test)

        context_sizes = [0, 1, 2, 5]
        median_list=[]
        for context_size in context_sizes:
            model.adapt(x = x_test[:, :context_size, :], y = y_test[:, :context_size, :])
            mu_z, var_z = model.aggregator.last_agg_state
            lmlhd_estimate_mc, _ = lmlhd_mc(lambda x,z: np_decode(model, x, z), (mu_z, var_z), (x_test, y_test), 1000)
            median_list.append(np.median(lmlhd_estimate_mc))
        
        objective = np.mean(median_list)

        print("median_list: " + str(median_list))
        print("objective: " + str(objective))

        wandb.log({"Average_ctx_size over {median_tasks over {LMLHD per task} per ctx size} for context sizes (0, 1, 2, 5)": objective})

        #optimizer = build_optimizer(network, config.optimizer, config.learning_rate)
        '''
        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})  
        '''

def build_benchmarks(config_path):
    with open(config_path, "r") as stream:
        try:
            benchmark_config = yaml.safe_load(stream)
            benchmark_meta = BM_DICT[benchmark_config["benchmark"]](
                n_task=benchmark_config["n_task_meta"],
                n_datapoints_per_task=benchmark_config["n_datapoints_per_task_meta"],
                output_noise=benchmark_config["data_noise_std"],
                seed_task=benchmark_config["seed_task_meta"],
                seed_x=benchmark_config["seed_x_meta"],
                seed_noise=benchmark_config["seed_noise_meta"],
            )
            benchmark_val = BM_DICT[benchmark_config["benchmark"]](
                n_task=benchmark_config["n_task_val"],
                n_datapoints_per_task=benchmark_config["n_datapoints_per_task_val"],
                output_noise=benchmark_config["data_noise_std"],
                seed_task=benchmark_config["seed_task_val"],
                seed_x=benchmark_config["seed_x_val"],
                seed_noise=benchmark_config["seed_noise_val"],
            )
            benchmark_test = BM_DICT[benchmark_config["benchmark"]](
                n_task=benchmark_config["n_task_test"],
                n_datapoints_per_task=benchmark_config["n_datapoints_per_task_test"],
                output_noise=benchmark_config["data_noise_std"],
                seed_task=benchmark_config["seed_task_test"],
                seed_x=benchmark_config["seed_x_test"],
                seed_noise=benchmark_config["seed_noise_test"],
            )
            return benchmark_meta, benchmark_val, benchmark_test

        except yaml.YAMLError as exc:
            print(exc)

def build_model(config):
    # logpath
    logpath = os.path.dirname(os.path.abspath(__file__))
    #logpath = os.path.join(logpath, os.path.join("..", "log"))
    logpath = os.path.join(logpath, "log")
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
        self_attention_type=None,
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

def train_model(model, benchmark_meta, benchmark_val, benchmark_test, config):

    log_loss = lambda n_meta_tasks_seen, np_model, metrics: wandb.log(metrics) if metrics is not None else None

    # callback switched to None for much faster meta-training during debugging
    model.meta_train(
        benchmark_meta=benchmark_meta,
        benchmark_val=benchmark_val,
        n_tasks_train=eval(config["n_tasks_train"]),
        validation_interval=eval(config["validation_interval"]),
        callback=log_loss,
    )

    model.save_model()

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

if __name__ == "__main__":
    with open("scripts/config.yaml", "r") as stream:
        sweep_config = yaml.safe_load(stream)

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
    
    wandb.agent(sweep_id, train, count=2)