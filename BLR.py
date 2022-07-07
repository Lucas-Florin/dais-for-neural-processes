import math
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm as univariate_normal

import torch
import random

from ais import ais_trajectory

class BayesianLinearRegression():

    def __init__(self, feature_dim):
        self.posterior_mean = None
        self.posterior_cov = None

        self.feature_dim = feature_dim

        self.precision = 0.2
        prior_mean = np.zeros(feature_dim)
        prior_cov = np.identity(feature_dim) * (1.0/self.precision)

        self.prior = multivariate_normal(prior_mean, prior_cov)

        self.noise_var = 0.25


    def train(self, dataset):
        train_points, noisy_labels, true_labels = dataset
        print("shape of train_points: " + str(train_points.shape))
        n_datapoints = train_points.shape[0]
        feature_matrix = np.stack((np.ones(n_datapoints),train_points), axis=1)

        print("shape of feature_matrix: " + str(feature_matrix.shape))

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

        mu_y = train_features.dot(self.posterior_mean)
        print("shape of mu_y: " + str(mu_y.shape))

        # cov_y should have shape n_points x d_y x d_y
        first_dot_product = self.posterior_cov.dot(train_features.T)
        intermediate_result = np.multiply(train_features, first_dot_product.T)
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
        predictive = univariate_normal(mu_y, np.sqrt(cov_y))
        
        log_lhds = np.log(predictive.pdf(test_labels))
        return np.sum(log_lhds)

    def get_log_dataset_likelihood_param(self, test_values, test_labels, parameter_val):
        ones = np.ones(test_values.shape[0])
        train_features = np.stack([ones, test_values], axis=1)
        
        pred_means = train_features.dot(parameter_val)
        pred_cov = self.noise_var
        predictive = univariate_normal(pred_means, math.sqrt(pred_cov))
        
        log_lhds = np.log(predictive.pdf(test_labels))
        return np.sum(log_lhds)


if __name__ == "__main__":
    blr = BayesianLinearRegression(feature_dim = 2)
    dataset = BayesianLinearRegression.generate_dataset(n_datapoints = 1000, slope = 0.9, intercept = -0.7, noise_std_dev = 0.5, seed = 42)

    train_points, noisy_labels, true_labels = dataset

    blr.train(dataset)

    log_dataset_likelihood = blr.get_log_dataset_likelihood(train_points, noisy_labels)
    print("True log dataset likelihood: " + str(log_dataset_likelihood))

    initial_state = torch.from_numpy(blr.prior.rvs(size=3)).float()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chain_length = 500

    forward_schedule = torch.linspace(0, 1, chain_length, device=device)
    
    ais_trajectory(
    lambda z: torch.from_numpy(blr.prior.logpdf(z)),
    lambda z: torch.from_numpy(blr.get_log_dataset_likelihood_param(train_points, noisy_labels, z)),
    initial_state,
    forward=True,
    schedule=forward_schedule)
    

