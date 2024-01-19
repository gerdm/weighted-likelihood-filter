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
from rebayes_mini.methods import replay_sgd
from rebayes_mini.methods import robust_filter as rkf


# Load config
config_path = sys.argv[1]
n_runs = int(sys.argv[2])

with open(config_path, "r") as f:
    config = toml.load(f)

uci = datagen.UCIDatasets("./data")

dataset_name = config["metadata"]["dataset-name"]
print("*" * 80)
print(f"Dataset: {dataset_name}")

noise_type = "target" # or "covariate"


def lossfn(params, counter, x, y, applyfn):
    yhat = applyfn(params, x)
    return jnp.sum(counter * (y - yhat) ** 2) / counter.sum()


def callback_fn(bel, bel_pred, y, x, applyfn):
    yhat = applyfn(bel_pred.mean, x[None])
    return yhat


def create_collection_datsets(p_error, n_runs, v_error=50, seed_init=314):
    X_collection= []
    y_collection = []
    ix_clean_collection = []

    for i in range(n_runs):
        if noise_type == "target":
            data = uci.sample_one_sided_noisy_dataset(dataset_name, p_error=p_error, seed=seed_init + i, v_error=v_error)
            ix_clean = ~data["err_where"].astype(bool)
        elif noise_type == "covariate":
            data = uci.sample_noisy_covariates(dataset_name, p_error=p_error, seed=seed_init + i, v_error=v_error)
            ix_clean = ~data["err_where"].any(axis=1).astype(bool)
        else:
            raise KeyError(f"Noise {noise_type} not available")
            
        X = data["X"]
        y = data["y"]
        
        X_collection.append(X)
        y_collection.append(y)
        ix_clean_collection.append(ix_clean)

    X_collection = jnp.array(X_collection)
    y_collection = jnp.array(y_collection)
    ix_clean_collection = np.array(ix_clean_collection).T

    return X_collection, y_collection, ix_clean_collection




@jax.jit
def filter_kf(log_lr, params_init, measurements, covariates):
    lr = jnp.exp(log_lr)
    agent = rkf.ExtendedKalmanFilterIMQ(
        latent_fn, measurement_fn,
        dynamics_covariance=Q,
        observation_covariance=observation_covariance,
        soft_threshold=1e8,
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    # out = (agent, bel)
    return yhat_pp.squeeze()


@jax.jit
def filter_kfb(log_lr, alpha, beta, n_inner, params_init, measurements, covariates):
    lr = jnp.exp(log_lr)
    n_inner = n_inner.astype(int)
    
    agent = rkf.ExtendedKalmanFilterBernoulli(
        latent_fn, measurement_fn,
        dynamics_covariance=Q,
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
def filter_kfiw(log_lr, noise_scaling, n_inner, params_init, measurements, covariates):
    lr = jnp.exp(log_lr)
    n_inner = n_inner.astype(int)
    
    agent = rkf.ExtendedKalmanFilterInverseWishart(
        latent_fn, measurement_fn,
        dynamics_covariance=Q,
        prior_observation_covariance=observation_covariance,
        n_inner=n_inner,
        noise_scaling=noise_scaling
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()


@jax.jit
def filter_wlfimq(log_lr, soft_threshold, params_init, measurements, covariates):
    lr = jnp.exp(log_lr)
    nsteps = len(measurements)
    agent = rkf.ExtendedKalmanFilterIMQ(
        latent_fn, measurement_fn,
        dynamics_covariance=Q,
        observation_covariance=observation_covariance,
        soft_threshold=soft_threshold,
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()


@jax.jit
def filter_wlfmd(log_lr, threshold, params_init, measurements, covariates):
    lr = jnp.exp(log_lr)
    agent = rkf.ExtendedKalmanFilterMD(
        latent_fn, measurement_fn,
        dynamics_covariance=Q,
        observation_covariance=observation_covariance,
        threshold=threshold,
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    # out = (agent, bel)
    return yhat_pp.squeeze()




@jax.jit
def filter_ogd(log_lr, n_inner, params_init, measurements, covariates):
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


fileter_fns = {
    "KF": filter_kf,
    "KF-B": filter_kfb,
    "KF-IW": filter_kfiw,
    "WLF-IMQ": filter_wlfimq,
    "WLF-MD": filter_wlfmd,
    "OGD": filter_ogd,
}

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(20)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


key = jax.random.PRNGKey(314)
model = MLP()
hyperparams = config["hyperparams"]

Q = 0.0
observation_covariance = jnp.eye(1) * 1.0

def latent_fn(x): return x
measurement_fn = model.apply

p_errors_collection = {}
time_collection_all = {}

p_errors = np.arange(0, 0.5, 0.05)

@partial(jax.jit, static_argnames=("model", "filterfn"))
def load_and_run(key, y, X, model, filterfn):
    params_init = model.init(key, X[:1])
    yhat_pp = filterfn(**hparams, params_init=params_init, measurements=y, covariates=X)
    errs = (y - yhat_pp) ** 2
    return errs


for p_error in tqdm(p_errors):
    p100 = int(100 * p_error)
    X_collection, y_collection, ix_clean_collection = create_collection_datsets(p_error, n_runs, v_error=50, seed_init=314)

    p_errors_collection[p100] = {}
    time_collection_all[p100] = {}
    for method in (pbar := tqdm(hyperparams, leave=False)):
        pbar.set_description(f"Method: {method}")
        filterfn = fileter_fns[method]
        hparams = hyperparams[method]

        errs_collection = []
        time_collection = []
        for X, y in tqdm(zip(X_collection, y_collection), total=n_runs, leave=False):
            time_init = time()
            errs = load_and_run(key, y, X, model, filterfn)
            errs = jax.block_until_ready(errs)
            time_end = time()
            
            time_collection.append(time_end - time_init)
            errs_collection.append(errs)

        errs_collection = np.array(errs_collection)
        time_collection = np.array(time_collection)

        p_errors_collection[p100][method] = errs_collection
        time_collection_all[p100][method] = time_collection

res = {
    "p_errors_collection": p_errors_collection,
    "time_collection": time_collection_all,
    "config": config,
}


filename = f"./results/regression-stress-{dataset_name}.pkl"
with open(filename, "wb") as f:
    pickle.dump(res, f)
