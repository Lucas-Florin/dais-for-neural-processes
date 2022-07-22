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
        self.prior_mean = tf.zeros(feature_dim)
        self.prior_cov = tf.eye(feature_dim) * (1.0/self.precision)

        self.noise_var = 0.25


    def train(self, dataset):
        train_points, noisy_labels, _ = dataset
        train_points = tf.convert_to_tensor(train_points, dtype=tf.float32)
        noisy_labels = tf.convert_to_tensor(noisy_labels, dtype=tf.float32)

        n_datapoints = train_points.shape[0]
        feature_matrix = tf.stack((tf.ones(n_datapoints),train_points), axis=1)

        inverse_mat = tf.linalg.inv(tf.matmul(feature_matrix.T, feature_matrix) + self.noise_var * self.precision * tf.eye(self.feature_dim))

        # Compute the posterior distribution
        self.posterior_mean = tf.linalg.matvec(inverse_mat, tf.linalg.matvec(feature_matrix.T, noisy_labels))
        self.posterior_cov = self.noise_var * inverse_mat
        
    def predict(self, x):
        ones = tf.ones(x.shape[0])
        train_features = tf.stack([ones, x], axis=1)

        mu_y = tf.linalg.matvec(train_features, self.posterior_mean)

        # cov_y should have shape n_points x d_y x d_y
        first_dot_product = tf.matmul(self.posterior_cov, train_features.T)
        intermediate_result = tf.matmul(train_features, first_dot_product)

        cov_y = intermediate_result + self.noise_var * tf.eye(intermediate_result.shape[0])

        return mu_y, cov_y

    def generate_dataset(n_datapoints, slope, intercept, noise_std_dev, seed = 42):
        np.random.seed(seed)
        lower_bound = -1.5
        upper_bound = 1.5

        train_points = np.random.uniform(lower_bound, upper_bound, n_datapoints)
        noisy_labels = slope * train_points + intercept + np.random.normal(0, noise_std_dev, n_datapoints)
        true_labels = slope * train_points + intercept

        return train_points, noisy_labels, true_labels

    def get_log_predictive_likelihood(self, test_values, test_labels):
        mu_y, cov_y = self.predict(test_values)
        predictive = tfd.MultivariateNormalTriL(loc=mu_y, scale_tril=tf.linalg.cholesky(cov_y))
        
        log_lhd = predictive.log_prob(test_labels)
        return log_lhd

    def get_log_marg_likelihood(self, test_values, test_labels):

        ones = tf.ones(test_values.shape[0])
        train_features = tf.stack([ones, test_values], axis=1)

        marg_means = tf.linalg.matvec(train_features, self.prior_mean)

        # cov_y should have shape n_points x n_points
        first_dot_product = tf.matmul(self.prior_cov, train_features.T)
        intermediate_result = tf.matmul(train_features, first_dot_product)

        marg_cov = intermediate_result + self.noise_var * tf.eye(intermediate_result.shape[0])

        # What if y is itself multidimensional?
        marg = tfd.MultivariateNormalTriL(loc=marg_means, scale_tril=tf.linalg.cholesky(marg_cov))

        log_marg_lhd = marg.log_prob(test_labels)
        return log_marg_lhd

    def get_log_dataset_likelihood_param(self, test_values, test_labels, parameter_val):
        ones = tf.ones(test_values.shape[0])
        train_features = tf.stack([ones, test_values], axis=1)
        
        pred_means = tf.matmul(train_features, parameter_val.T).T
        pred_cov = self.noise_var
        predictive = tfd.Normal(pred_means, tf.sqrt(pred_cov))
        
        log_lhds = predictive.log_prob(test_labels)
        return tf.math.reduce_sum(log_lhds, axis=-1)

    def get_prior_distribution(self):
        return tfd.MultivariateNormalDiag(loc = self.prior_mean, scale_diag = tf.ones(self.prior_mean.shape[0]) * (1.0/self.precision))

    def get_posterior_distribution(self):
        return tfd.MultivariateNormalTriL(loc = self.posterior_mean, scale_tril=tf.linalg.cholesky(self.posterior_cov))

if __name__ == "__main__":
    blr = BayesianLinearRegression(feature_dim = 2)
    dataset = BayesianLinearRegression.generate_dataset(n_datapoints = 1000, slope = 0.9, intercept = -0.7, noise_std_dev = 0.5, seed = 42)
    testset = BayesianLinearRegression.generate_dataset(n_datapoints = 1000, slope = 0.9, intercept = -0.7, noise_std_dev = 0.5, seed = 21)

    test_points, noisy_labels, true_labels = testset

    test_points = tf.convert_to_tensor(test_points, dtype=tf.float32)
    noisy_labels = tf.convert_to_tensor(noisy_labels, dtype=tf.float32)
    true_labels = tf.convert_to_tensor(true_labels, dtype=tf.float32)


    blr.train(dataset)

    log_predictive_likelihood = blr.get_log_predictive_likelihood(test_points, noisy_labels)
    log_marginal_likelihood = blr.get_log_marg_likelihood(test_points, noisy_labels)
    print("True log predictive likelihood: " + str(log_predictive_likelihood))
    print("True log marginal likelihood: " + str(log_marginal_likelihood))

    n_samples = 1000
    chain_length = 100
    initial_state = blr.get_prior_distribution().sample((n_samples,))

    weight_samples, ais_weights, kernel_results = (
        tfp.mcmc.sample_annealed_importance_chain(
            num_steps = chain_length,
            proposal_log_prob_fn = blr.get_prior_distribution().log_prob,
            target_log_prob_fn = lambda z: blr.get_prior_distribution().log_prob(z) + blr.get_log_dataset_likelihood_param(test_points, noisy_labels, z),
            current_state = initial_state,
            make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=tlp_fn,
                step_size=0.01,
                num_leapfrog_steps=30)

        )
    )

    log_normalizer_estimate = (tf.reduce_logsumexp(ais_weights)
                           - np.log(n_samples))
    print("AIS estimate of log marginal likelihood: " + str(log_normalizer_estimate))
    

    initial_state = blr.get_posterior_distribution().sample((n_samples,))

    weight_samples, ais_weights, kernel_results = (
        tfp.mcmc.sample_annealed_importance_chain(
            num_steps = chain_length,
            proposal_log_prob_fn = blr.get_posterior_distribution().log_prob,
            target_log_prob_fn = lambda z: blr.get_posterior_distribution().log_prob(z) + blr.get_log_dataset_likelihood_param(test_points, noisy_labels, z),
            current_state = initial_state,
            make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=tlp_fn,
                step_size=0.01,
                num_leapfrog_steps=30)

        )
    )
    log_normalizer_estimate = (tf.reduce_logsumexp(ais_weights)
                           - np.log(n_samples))
    print("AIS estimate of log predictive likelihood: " + str(log_normalizer_estimate))
