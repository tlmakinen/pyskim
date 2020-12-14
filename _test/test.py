# test functions for skim module
# by TLM
import argparse
import itertools
import os
import time
import sys

import numpy as onp

import jax
from jax import vmap
import jax.numpy as np
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import corner


from SKIM import skim

################################################################################################

# Create artificial regression dataset where only S out of P feature
# dimensions contain signal and where there is a single pairwise interaction
# between the first and second dimensions.
def get_toy_data(N=20, S=2, P=10, sigma_obs=0.05, active_pairs=[0,1]):
    assert S < P and P > 1 and S > 0
    onp.random.seed(0)

    X = onp.random.randn(N, P)
    # generate S coefficients with non-negligible magnitude
    W = 0.5 + 2.5 * onp.random.rand(S)
    # generate data using the S coefficients and a single pairwise interaction
    Y = onp.sum(X[:, 0:S] * W, axis=-1) + X[:, 0] * X[:, 1] + sigma_obs * onp.random.randn(N)
    Y -= np.mean(Y)
    Y_std = np.std(Y)

    assert X.shape == (N, P)
    assert Y.shape == (N,)

    return X, Y / Y_std, W / Y_std, 1.0 / Y_std

################################################################################################

# FIRST TEST INFERENCE
def test_one():

	print('running default test inference with 3 active dims and 1 pairwise interaction')

	# define the parameters for the initialization
	num_dimensions=20; active_dimensions=3; num_data=100; labels=None; N_samps = 1000; num_chains=1


	X, Y, expected_thetas, expected_pairwise = get_toy_data(N=num_data, P=num_dimensions,
	                                                        S=active_dimensions)

	hypers = {'expected_sparsity': max(1.0, num_dimensions / 10),
	              'sigma' : 3.0,
	              'alpha1': 3.0, 'beta1': 1.0,
	              'alpha2': 3.0, 'beta2': 1.0,
	              'alpha3': 1.0, 'c': 1.0,
	              'alpha_obs': 3.0, 'beta_obs': 1.0}

	skim_model = skim.SKIM(X, Y, hypers, seed=0)

	all_active_dimensions, thetas, labels, pair_labs = skim_model.generate_posterior(
														known_active_dimensions=3, labels=None)
	fig = corner.corner(thetas, labels = labels)

	plt.show()


################################################################################################

# SECOND SET OF TEST INFERENCES -- vary toy model active dimensions
def test_two():

	print('running test varying number of active dimensions')

	num_dimensions=20;
	active_dimensions=[1, 2, 4, 5];
	num_data=100;
	labels=None;
	N_samps = 1000;
	num_chains=1


	for i in range(len(active_dimensions)):

		n_dim = active_dimensions[i]

		X, Y, expected_thetas, expected_pairwise = get_toy_data(N=num_data, P=num_dimensions,
	                                                        S=n_dim)

		hypers = {'expected_sparsity': max(1.0, num_dimensions / 10),
	              'sigma' : 3.0,
	              'alpha1': 3.0, 'beta1': 1.0,
	              'alpha2': 3.0, 'beta2': 1.0,
	              'alpha3': 1.0, 'c': 1.0,
	              'alpha_obs': 3.0, 'beta_obs': 1.0}

		print('now testing inference on toy data with %d active dimensions'%(n_dim))

		skim_model = skim.SKIM(X, Y, hypers, seed=0)

		all_active_dimensions, thetas, labels, pair_labs = skim_model.generate_posterior(
													known_active_dimensions=n_dim, labels=None)

		# plot the last inference
		if i == len(active_dimensions) -1:
			fig = corner.corner(thetas, labels = labels)
			plt.show()


################################################################################################

# THIRD SET OF TEST INFERENCES -- vary toy model pairwise interactions

def get_toy_pairs(N=20, S=2, P=10, sigma_obs=0.05, active_pairs=[(0,1), (1,2)]):
    assert S < P and P > 1 and S > 0


    onp.random.seed(0)

    X = onp.random.randn(N, P)
    # generate S coefficients with non-negligible magnitude
    W = 0.5 + 2.5 * onp.random.rand(S)
    # generate data using the S coefficients and however many pairwise interactions


    Y = onp.sum(X[:, 0:S] * W, axis=-1)

    # now add in all pairwise interactions
    for pair in active_pairs:
    	Y += X[:, pair[0]] * X[:, pair[1]]

    Y += sigma_obs * onp.random.randn(N)
    Y -= np.mean(Y)
    Y_std = np.std(Y)

    assert X.shape == (N, P)
    assert Y.shape == (N,)

    return X, Y / Y_std, W / Y_std, 1.0 / Y_std

def test_three():

	print('running test varying active pairs')

	num_dimensions=20;
	active_dimensions=4;
	num_data=100;
	labels=None;
	N_samps = 1000;
	num_chains=1

	active_pairs = [(0,1), (1,2), (0,2)]


	X, Y, expected_thetas, expected_pairwise = get_toy_pairs(N=num_data, P=num_dimensions,
                                                        S=active_dimensions, active_pairs=active_pairs)

	hypers = {'expected_sparsity': max(1.0, num_dimensions / 10),
              'sigma' : 3.0,
              'alpha1': 3.0, 'beta1': 1.0,
              'alpha2': 3.0, 'beta2': 1.0,
              'alpha3': 1.0, 'c': 1.0,
              'alpha_obs': 3.0, 'beta_obs': 1.0}

	print('now testing inference on toy data with active pairwise interactions between :\n')
	for pair in active_pairs:
		print('dim %d and dim %d'%(pair[0], pair[1]))

	skim_model = skim.SKIM(X, Y, hypers, seed=0)

	all_active_dimensions, thetas, labels, pair_labs = skim_model.generate_posterior(
										known_active_dimensions=active_dimensions, labels=None)

	# plot the last inference
	#if i == len(active_dimensions) -1:
	fig = corner.corner(thetas, labels = labels)
	plt.show()


################################################################################################

# FOURTH TEST INFERENCE -- vary sigma
def test_four():

	print('running test varying noise sigma')

	num_dimensions=20;
	active_dimensions=3;
	num_data=100;
	labels=None;
	N_samps = 1000;
	num_chains=1
	sigmas = [0.1, 0.15, 0.5, 0.8]


	for i in range(len(sigmas)):

		sigma_obs = sigmas[i]

		X, Y, expected_thetas, expected_pairwise = get_toy_data(N=num_data, P=num_dimensions,
	                                                      S=active_dimensions, sigma_obs=sigma_obs)

		hypers = {'expected_sparsity': max(1.0, num_dimensions / 10),
	              'sigma' : 3.0,
	              'alpha1': 3.0, 'beta1': 1.0,
	              'alpha2': 3.0, 'beta2': 1.0,
	              'alpha3': 1.0, 'c': 1.0,
	              'alpha_obs': 3.0, 'beta_obs': 1.0}

		print('now testing inference on toy data with %02f observed noise'%(sigma_obs))

		skim_model = skim.SKIM(X, Y, hypers, seed=0)

		all_active_dimensions, thetas, labels, pair_labs = skim_model.generate_posterior(
											known_active_dimensions=active_dimensions, labels=None)

		print('all active dimensions :', all_active_dimensions)

		# plot the last inference
		if i == len(sigmas) - 1:
			fig = corner.corner(thetas, labels = labels)
			plt.show()


################################################################################################

if __name__ == "__main__":

	if int(sys.argv[1]) == 2:
		test_two()

	elif int(sys.argv[1]) == 3:
		test_three()

	elif int(sys.argv[1]) == 4:
		test_four()

	else:
		test_one()




