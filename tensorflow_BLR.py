from itertools import chain
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

class BayesianLinearRegression():

    def __init__(self, feature_dim):
        self.posterior_mean = None
        self.posterior_cov = None
        self.feature_dim = feature_dim

        self.precision = 0.2
        self.prior_mean = np.zeros(feature_dim)
        self.prior_cov = np.identity(feature_dim) * (1.0/self.precision)

        self.noise_var = 0.25


    def train(self, dataset):
        train_points, noisy_labels, _ = dataset

        n_datapoints = train_points.shape[0]
        feature_matrix = np.stack((np.ones(n_datapoints),train_points), axis=1)

        inverse_mat = np.linalg.inv(np.dot(feature_matrix.T, feature_matrix) + self.noise_var * self.precision * np.identity(self.feature_dim))

        print("shape of inverse_mat: " + str(inverse_mat.shape))

        # Compute the posterior distribution
        self.posterior_mean = np.dot(inverse_mat, np.dot(feature_matrix.T, noisy_labels))
        self.posterior_cov = self.noise_var * inverse_mat

        print("shape of posterior_mean: " + str(self.posterior_mean.shape))
        print("shape of posterior_cov: " + str(self.posterior_cov.shape))
        
    def predict(self, x):
        ones = np.ones(x.shape[0])
        train_features = np.stack([ones, x], axis=1)
        print("shape of train features: " + str(train_features.shape))

        mu_y = np.dot(train_features, self.posterior_mean)
        print("shape of mu_y: " + str(mu_y.shape))

        # cov_y should have shape n_points x d_y x d_y
        first_dot_product = np.dot(self.posterior_cov, train_features.T)
        intermediate_result = train_features * first_dot_product.T
        print("shape of intermediate_result: " + str(intermediate_result.shape))

        cov_y = np.sum(intermediate_result, axis=1) + self.noise_var

        return mu_y, cov_y

    def generate_dataset(n_datapoints, slope, intercept, noise_std_dev, seed = 42):
        np.random.seed(seed)
        lower_bound = -1.5
        upper_bound = 1.5

        train_points = np.random.uniform(lower_bound, upper_bound, n_datapoints)
        noisy_labels = slope * train_points + intercept + np.random.normal(0, noise_std_dev, n_datapoints)
        true_labels = slope * train_points + intercept

        return train_points, noisy_labels, true_labels

    def get_log_dataset_likelihood(self, test_values, test_labels):
        mu_y, cov_y = self.predict(test_values)
        print("shape of mu_y: " + str(mu_y.shape) + ", shape of cov_y: " + str(cov_y.shape))
        predictive = tfd.Normal(mu_y, np.sqrt(cov_y))
        
        log_lhds = predictive.log_prob(test_labels)
        return tf.math.reduce_sum(log_lhds, axis=-1)

    def get_log_dataset_likelihood_param(self, test_values, test_labels, parameter_val):
        ones = np.ones(test_values.shape[0])
        train_features = np.stack([ones, test_values], axis=1)
        
        pred_means = np.dot(train_features, parameter_val.T).T
        pred_cov = self.noise_var
        predictive = tfd.Normal(pred_means, np.sqrt(pred_cov))
        
        log_lhds = predictive.log_prob(test_labels)
        return tf.math.reduce_sum(log_lhds, axis=-1)

    def get_prior_distribution(self):
        return tfd.MultivariateNormalDiag(loc = self.prior_mean, scale_diag = np.ones(self.prior_mean.shape[0]) * (1.0/self.precision))

if __name__ == "__main__":
    blr = BayesianLinearRegression(feature_dim = 2)
    dataset = BayesianLinearRegression.generate_dataset(n_datapoints = 1000, slope = 0.9, intercept = -0.7, noise_std_dev = 0.5, seed = 42)

    train_points, noisy_labels, _ = dataset
    blr.train(dataset)

    log_dataset_likelihood = blr.get_log_dataset_likelihood(train_points, noisy_labels)
    print("True log dataset likelihood: " + str(log_dataset_likelihood))

    n_samples = 100
    chain_length = 10000
    initial_state = blr.get_prior_distribution().sample((n_samples,))

    weight_samples, ais_weights, kernel_results = (
        tfp.mcmc.sample_annealed_importance_chain(
            num_steps = chain_length,
            proposal_log_prob_fn = blr.get_prior_distribution().log_prob,
            target_log_prob_fn = lambda z: blr.get_log_dataset_likelihood_param(train_points, noisy_labels, z),
            current_state = initial_state,
            make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=tlp_fn,
                step_size=0.1,
                num_leapfrog_steps=2)

        )
    )

    log_normalizer_estimate = (tf.reduce_logsumexp(ais_weights)
                           - np.log(n_samples))
    print(log_normalizer_estimate)
