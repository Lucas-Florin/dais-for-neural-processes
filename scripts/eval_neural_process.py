import logging
import os
import numpy as np
import torch
torch.manual_seed(0)
import wandb


from matplotlib import pyplot as plt
from metalearning_benchmarks import MetaLearningBenchmark
from metalearning_benchmarks import benchmark_dict as BM_DICT
from neural_process.neural_process import NeuralProcess

from bayesian_meta_learning.lmlhd_estimators import lmlhd_mc, lmlhd_iwmc, lmlhd_ais
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

def np_decode(np_model, x, z):
    assert x.ndim == 3  # (n_tsk, n_tst, d_x)
    assert z.ndim == 4  # (n_tsk, n_ls, n_marg, d_z)

    x = np_model._normalize_x(x)
    mu_y, std_y = np_model.decoder.decode(x, z) # decoder returns std!!
    mu_y = np_model._denormalize_mu_y(mu_y)

    return mu_y, std_y

def estimates_over_time(decode, task, context_distributions, target_distribution, n_samples = 10):
    task_x, task_y = task
    n_tsk = task_x.shape[0]
    log_likelihood_probs_mc_matrix = np.zeros((len(context_distributions), n_samples, n_tsk))
    #log_likelihood_probs_iwmc_matrix = np.zeros((len(context_distributions), n_samples, n_tsk))
    for i, context_distribution in enumerate(context_distributions):
        _, log_likelihood_probs_mc = lmlhd_mc(decode, context_distribution, task, n_samples = n_samples)
        #_, log_likelihood_probs_iwmc, _, _ = lmlhd_iwmc(decode, context_distribution, target_distribution, task, n_samples = n_samples)
        log_likelihood_probs_mc_matrix[i, :, :] = log_likelihood_probs_mc
        #log_likelihood_probs_iwmc_matrix[i, :, :] = log_likelihood_probs_iwmc

    log_likelihoods_mc = [[] for x in range(len(context_distributions))]
    log_likelihoods_iwmc = [[] for x in range(len(context_distributions))]

    for j in range(4, int(np.log(n_samples)) + 1):
        # logsumexp over the samples
        log_likelihoods_mc_per_task = logsumexp(log_likelihood_probs_mc_matrix[:, :int(np.exp(j)), :], axis=1) - j
        assert log_likelihoods_mc_per_task.shape == (len(context_distributions), n_tsk)

        # logsumexp over the samples
        #log_likelihoods_iwmc_per_task = logsumexp(log_likelihood_probs_iwmc_matrix[:, :int(np.exp(j)), :], axis=1) - j
        #assert log_likelihoods_iwmc_per_task.shape == (len(context_distributions), n_tsk)

        for k in range(0, len(context_distributions)):
            # take median across tasks
            median_log_likelihoods_mc = np.median(log_likelihoods_mc_per_task, axis=1)
            assert median_log_likelihoods_mc.shape == (len(context_distributions),)
            log_likelihoods_mc[k].append(median_log_likelihoods_mc[k].item())
            #log_likelihoods_iwmc[k].append((logsumexp(log_likelihoods_iwmc_per_task, axis=1) - np.log(n_tsk))[k].item())

    logger.warning("log_likelihoods_mc: " + str(log_likelihoods_mc))
    logger.warning("log_likelihoods_iwmc: " + str(log_likelihoods_iwmc))
    '''
    x = np.arange(4, int(np.log(n_samples)) + 1)
    line_symbols = ['o-', 'o--', 'o-.', 'o:', 'v-', '+-']
    for k in range(0, len(context_distributions)):
        plt.plot(x, log_likelihoods_mc[k], f'r{line_symbols[k]}', label=f'MC, {2 ** k} context points')
    
    #for k in range(0, len(context_distributions)):
    #    plt.plot(x, log_likelihoods_iwmc[k], f'g{line_symbols[k]}', label=f'IWMC, {2 ** k} context points')
    
    #plt.plot(x, log_likelihoods_iwmc[0], 'go-', label='IWMC')
    plt.xlabel("Log num samples")
    plt.ylabel("log predictive likelihood estimate")
    plt.legend()
    plt.title('Estimates over time, n_context_points = 4')
    plt.savefig("plot.png")
    '''
    return log_likelihood_probs_mc_matrix

def plot_likelihoods_box(decode, task, distribution1, distribution2, num_samples):
    mu1, var1 = distribution1
    mu2, var2 = distribution2

    assert mu1.shape == var1.shape
    assert mu2.shape == var2.shape
    assert mu1.shape == mu2.shape

    _, log_lhds_samples_context = lmlhd_mc(decode, distribution1, task, num_samples)
    _, log_lhds_samples_target = lmlhd_mc(decode, distribution2, task, num_samples)

    logger.warning(log_lhds_samples_context.shape)
    logger.warning(log_lhds_samples_context.max())

    logger.warning(log_lhds_samples_target.shape)
    logger.warning(log_lhds_samples_target.max())
 
    plt.boxplot([log_lhds_samples_context[:, 0], log_lhds_samples_target[:, 0]])
    plt.title("Likelihoods of latent samples from different distributions")
    plt.xticks([1, 2], ["context (0 datapoints)", "target (32 datapoints)"])
    plt.show()

def plot_weights_likelihoods(decode, task, context_distribution, target_distribution, num_samples):
    _, _, log_lmlhd_samples, log_importance_weights = lmlhd_iwmc(decode, context_distribution, target_distribution, task, num_samples)
    x = np.arange(0, num_samples)
    plt.plot(x, log_lmlhd_samples[:, 0], 'ro-', label='Log Likelihoods')
    plt.plot(x, log_importance_weights[:, 0], 'go-', label='Log Importance Weights')
    plt.xlabel("Samples")
    plt.legend()
    plt.show()

def collate_benchmark(benchmark: MetaLearningBenchmark):
    # collate test data
    x = np.zeros((benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_x))
    y = np.zeros((benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_y))
    for l, task in enumerate(benchmark):
        x[l] = task.x
        y[l] = task.y

    return x, y

def plot(
    np_model: NeuralProcess,
    benchmark: MetaLearningBenchmark,
    n_task_max: int,
    n_context_points: int,
    fig,
    axes,
    device=None,
):
    # determine n_task
    n_task_plot = min(n_task_max, benchmark.n_task)

    # evaluate predictions
    n_samples = 500
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
    for i in range(0, n_context_points + 1):
        np_model.adapt(x=x_test[:, :i, :], y=y_test[:, :i, :]) # adapt model on context set of size i
        mu_z, var_z = np_model.aggregator.last_agg_state

        mu_y, _ = np_model.predict(x=x_plt, n_samples=n_samples)
        assert mu_y.shape == (n_task_plot, n_samples, x_plt.shape[0], benchmark.d_y)

        np_model.adapt(x = x_test[:,:32,:], y = y_test[:,:32,:]) # adapt model on max. context set size seen during training for importance sampling
        mu_z_target, var_z_target = np_model.aggregator.last_agg_state
        
        lmlhd_mc_estimates, _ = lmlhd_mc(lambda x,z: np_decode(np_model, x, z), (mu_z, var_z), (x_test, y_test), 1000)
        lmlhd_iwmc_estimates, _, _, _ = lmlhd_iwmc(lambda x,z: np_decode(np_model, x, z), (mu_z, var_z), (mu_z_target, var_z_target),(x_test, y_test), 1000)
        lmlhd_ais_estimates = lmlhd_ais(lambda x,z: np_decode(np_model, x, z), (mu_z, var_z), (x_test, y_test), n_samples = 10, chain_length=100, device=device)

        assert lmlhd_ais_estimates.shape == (n_task_plot,)
        assert lmlhd_mc_estimates.shape == (n_task_plot,)
        assert lmlhd_iwmc_estimates.shape == (n_task_plot,)

        for l in range(n_task_plot):
            ax = axes[i, l]
            ax.clear()
            ax.scatter(x_test[l, i:, :], y_test[l, i:, :], s=15, color="g", alpha=1.0, zorder=3)
            ax.scatter(x_test[l, :i, :], y_test[l, :i, :], marker="x", s=30, color="r", alpha=1.0, zorder=3)
            
            for s in range(n_samples):
                ax.plot(x_plt, mu_y[l, s, :, :], color="b", alpha=0.3, label="posterior", zorder=2)
            
            print("number of context points: " + str(i) + ", task: " + str(l))
            print("MC estimate: " + str(round(lmlhd_mc_estimates[l].item(), 3)))
            print("AIS estimate: " + str(round(lmlhd_ais_estimates[l].item(), 3)))
            print("IWMC estimate: " + str(round(lmlhd_iwmc_estimates[l].item(), 3)))

            ax.grid(zorder=1)
            if(i == 0):
                ax.set_title(f"Predictions (Task {l:d})")
            ax.text(0, 0, "MC: " + str(round(lmlhd_mc_estimates[l].item(), 3)) + ", AIS: " + str(round(lmlhd_ais_estimates[l].item(), 3)) + ", IWMC: " + str(round(lmlhd_iwmc_estimates[l].item(), 3)), horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

def train(model, benchmark_meta, benchmark_val, benchmark_test, config):
    # Log in to your W&B account
    wandb.login()
    wandb.init(
      # Set the project where this run will be logged
      project="Eval Neural Process", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"experiment_3", 
      config={"n_tasks_train": eval(config["n_tasks_train"]), "n_hidden_units": config["n_hidden_units"]}
    )

    log_loss = lambda n_meta_tasks_seen, np_model, metrics: wandb.log(metrics) if metrics is not None else None

    # callback switched to None for much faster meta-training during debugging
    model.meta_train(
        benchmark_meta=benchmark_meta,
        benchmark_val=benchmark_val,
        n_tasks_train=eval(config["n_tasks_train"]),
        validation_interval=eval(config["validation_interval"]),
        callback=log_loss,
    )

    # Mark the run as finished
    wandb.finish()
    model.save_model()

def main():
    # logpath
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
    # If there is no trained model to load, the model can be trained with the following line
    #train(model, benchmark_meta, benchmark_val, benchmark_test, config)
    
    model.load_model(config["n_tasks_train"])
    
    #task = benchmark_test.get_task_by_index(1)
    #x_test = np.expand_dims(task.x, axis=0)
    #y_test = np.expand_dims(task.y, axis=0)
    #x_test, y_test = collate_benchmark(benchmark_test)

    '''
    context_distributions = []
    for i in range(6):
        model.adapt(x = x_test[:, :(2 ** i), :], y = y_test[:, :(2 ** i), :])
        mu_z, var_z = model.aggregator.last_agg_state
        context_distributions.append((mu_z, var_z))
    '''
    
    #torch.manual_seed(0)
    #lmlhd_estimate_mc, _ = lmlhd_mc(lambda x,z: np_decode(model, x, z), (mu_z, var_z), (x_test, y_test), 100)
    #print(lmlhd_estimate_mc)
    #model.adapt(x = x_test[:, :4, :], y = y_test[:, :4, :])
    #mu_z, var_z = model.aggregator.last_agg_state
    #model.adapt(x = x_test[:,:32,:], y = y_test[:,:32,:])
    #mu_z_target, var_z_target = model.aggregator.last_agg_state
    #torch.manual_seed(0)
    #lmlhd_estimate_iwmc, _, _, _ = lmlhd_iwmc(lambda x,z: np_decode(model, x, z), (mu_z, var_z), (mu_z_target, var_z_target),(x_test, y_test), 100)
    #print(lmlhd_estimate_iwmc)

    #plot_weights_likelihoods(lambda x,z: np_decode(model, x, z), (x_test, y_test), (mu_z, var_z), (mu_z_target, var_z_target), 100)
    #plot_likelihoods_box(lambda x,z: np_decode(model, x, z), (x_test, y_test), (mu_z, var_z), (mu_z_target, var_z_target), 10000)
    #estimates_over_time(lambda x,z: np_decode(model, x, z), (x_test, y_test), context_distributions, (mu_z_target, var_z_target), n_samples = 100)
    
    n_task_plot = 4
    n_context_points = 4
    
    fig, axes = plt.subplots(
        nrows=n_context_points + 1,
        ncols=n_task_plot,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(5 * n_task_plot, 11),
    )

    fig.subplots_adjust(wspace=1, hspace=-100)

    plot(
        np_model=model,
        n_task_max=n_task_plot,
        benchmark=benchmark_test,
        n_context_points = n_context_points,
        fig=fig,
        axes=axes,
        device=config['device']
    )
    fig.savefig('temp.png', dpi=fig.dpi)
    fig.savefig('temp.pdf')
    plt.show()


if __name__ == "__main__":
    main()
