#!/usr/bin/env python
# coding: utf-8

# # UCI regression with outliers

import sys
import jax
import toml
import optax
import pickle
import datagen
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from tqdm import tqdm
from time import time
from functools import partial
from bayes_opt import BayesianOptimization
from rebayes_mini.methods import replay_sgd
from rebayes_mini.methods import robust_filter as rkf


# Load config
dataset_name = sys.argv[1]
n_runs = int(sys.argv[2])
config_path_search = sys.argv[3]


with open(config_path_search, "r") as f:
    config_search = toml.load(f)

print("*" * 80)
print(f"Dataset: {dataset_name}")

noise_type = "target" # or "covariate"

def lossfn(params, counter, x, y, applyfn):
    yhat = applyfn(params, x)
    return jnp.sum(counter * (y - yhat) ** 2) / counter.sum()

def callback_fn(bel, bel_pred, y, x, applyfn):
    yhat = applyfn(bel_pred.mean, x[None])
    return yhat


Q = 0.0
observation_covariance = jnp.eye(1) * 1.0

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(20)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


model = MLP()


def latent_fn(x): return x
measurement_fn = model.apply


@jax.jit
def filter_kf(
        log_lr, params_init,
        dynamics_covariance, measurements, covariates
):
    lr = jnp.exp(log_lr)
    agent = rkf.ExtendedKalmanFilterIMQ(
        latent_fn, measurement_fn,
        dynamics_covariance=dynamics_covariance,
        observation_covariance=observation_covariance,
        soft_threshold=1e8,
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()


@jax.jit
def filter_kfb(
        log_lr, alpha, beta, n_inner, dynamics_covariance,
        params_init, measurements, covariates
):
    lr = jnp.exp(log_lr)
    n_inner = n_inner.astype(int)
    
    agent = rkf.ExtendedKalmanFilterBernoulli(
        latent_fn, measurement_fn,
        dynamics_covariance=dynamics_covariance,
        observation_covariance=observation_covariance,
        alpha=alpha,
        beta=beta,
        tol_inlier=1e-7,
        n_inner=n_inner
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()


@jax.jit
def filter_kfiw(
        log_lr, noise_scaling, n_inner, dynamics_covariance,
        params_init, measurements, covariates
):
    lr = jnp.exp(log_lr)
    n_inner = n_inner.astype(int)
    
    agent = rkf.ExtendedKalmanFilterInverseWishart(
        latent_fn, measurement_fn,
        dynamics_covariance=dynamics_covariance,
        prior_observation_covariance=observation_covariance,
        n_inner=n_inner,
        noise_scaling=noise_scaling
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()


@jax.jit
def filter_wlfimq(
        log_lr, soft_threshold, dynamics_covariance,
        params_init, measurements, covariates
):
    lr = jnp.exp(log_lr)
    agent = rkf.ExtendedKalmanFilterIMQ(
        latent_fn, measurement_fn,
        dynamics_covariance=dynamics_covariance,
        observation_covariance=observation_covariance,
        soft_threshold=soft_threshold,
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()


@jax.jit
def filter_wlfmd(
        log_lr, threshold, dynamics_covariance,
        params_init, measurements, covariates
):
    lr = jnp.exp(log_lr)
    agent = rkf.ExtendedKalmanFilterMD(
        latent_fn, measurement_fn,
        dynamics_covariance=dynamics_covariance,
        observation_covariance=observation_covariance,
        threshold=threshold,
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()

@jax.jit
def filter_ogd(
        log_lr, n_inner, params_init, 
        measurements, covariates
):
    lr = jnp.exp(log_lr)
    n_inner = n_inner.astype(int)
    
    agent = replay_sgd.FifoSGD(
        measurement_fn,
        lossfn,
        optax.adam(lr),
        buffer_size=1,
        dim_features=covariates.shape[-1],
        dim_output=1,
        n_inner=n_inner,
    )
    
    callback = partial(callback_fn, applyfn=measurement_fn)
    init_bel = agent.init_bel(params_init)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()


def build_bopt_step(filterfn, hparams, hparams_static, params_init, random_state, y, X):
    @jax.jit
    def opt_step(**hparams):
        # TODO: consider adding the measurement_fn and latent_fn as arguments
        yhat_pp = filterfn(
            **hparams, **hparams_static, params_init=params_init, measurements=y, covariates=X
        )
        err = jnp.power(yhat_pp - y, 2)
        err = jnp.median(err)
        err = jax.lax.cond(jnp.isnan(err), lambda: 1e6, lambda: err)
        return -err
    
    bo = BayesianOptimization(
        opt_step, hparams, random_state=random_state,
    )

    return bo
    

fileter_fns = {
    "KF": filter_kf,
    "KF-B": filter_kfb,
    "KF-IW": filter_kfiw,
    "WLF-IMQ": filter_wlfimq,
    "WLF-MD": filter_wlfmd,
    "OGD": filter_ogd,
}


X_collection, y_collection, ixs_collection = datagen.create_uci_collection(
    dataset_name, noise_type=noise_type, p_error=0.1, n_runs=n_runs,
    v_error=50, seed_init=314, path="./data"
)

key = jax.random.PRNGKey(314)
X, y = X_collection[0], y_collection[0]
random_state = config_search["shared"]["random_state"]
params_init = model.init(key, X[:1])

method = "KF-B"
filterfn = fileter_fns[method]
hparams = config_search[method]["learn"]
hparams_static = config_search[method]["static"]

bo = build_bopt_step(filterfn, hparams, hparams_static, params_init, random_state, y, X)
bo.maximize()
