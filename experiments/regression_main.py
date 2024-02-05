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


def filter_kf(
        log_lr, params_init,
        dynamics_covariance, observation_covariance, measurements, covariates,
        measurement_fn, state_fn
):
    lr = jnp.exp(log_lr)
    agent = rkf.ExtendedKalmanFilterIMQ(
        state_fn, measurement_fn,
        dynamics_covariance=dynamics_covariance,
        observation_covariance=observation_covariance,
        soft_threshold=1e8,
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()



def filter_kfb(
        log_lr, alpha, beta, n_inner, dynamics_covariance, observation_covariance,
        params_init, measurements, covariates,
        measurement_fn, state_fn
):
    lr = jnp.exp(log_lr)
    n_inner = n_inner.astype(int)
    
    agent = rkf.ExtendedKalmanFilterBernoulli(
        state_fn, measurement_fn,
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



def filter_kfiw(
        log_lr, noise_scaling, n_inner, dynamics_covariance, observation_covariance,
        params_init, measurements, covariates,
        measurement_fn, state_fn
):
    lr = jnp.exp(log_lr)
    n_inner = n_inner.astype(int)
    
    agent = rkf.ExtendedKalmanFilterInverseWishart(
        state_fn, measurement_fn,
        dynamics_covariance=dynamics_covariance,
        prior_observation_covariance=observation_covariance,
        n_inner=n_inner,
        noise_scaling=noise_scaling
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()



def filter_wlfimq(
        log_lr, soft_threshold, dynamics_covariance, observation_covariance,
        params_init, measurements, covariates,
        measurement_fn, state_fn
):
    lr = jnp.exp(log_lr)
    agent = rkf.ExtendedKalmanFilterIMQ(
        state_fn, measurement_fn,
        dynamics_covariance=dynamics_covariance,
        observation_covariance=observation_covariance,
        soft_threshold=soft_threshold,
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()



def filter_wlfmd(
        log_lr, threshold, dynamics_covariance, observation_covariance,
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


def filter_ogd(
        log_lr, n_inner, params_init, 
        measurements, covariates,
        measurement_fn, state_fn=None
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


def build_bopt_step(
        filterfn, hparams, hparams_static, params_init,
        random_state, y, X, measurement_fn, state_fn,
):
    @jax.jit
    def filterfn_jit(measurements, covariates, **hparams):
        yhat_pp = filterfn(
            **hparams, **hparams_static,
            params_init=params_init, measurements=measurements, covariates=covariates,
            measurement_fn=measurement_fn, state_fn=state_fn
        )
        err = jnp.power(yhat_pp - measurements, 2)
        err = jnp.median(err)
        err = jax.lax.cond(jnp.isnan(err), lambda: 1e6, lambda: err)
        return -err
    
    partial_filter = partial(filterfn_jit, measurements=y, covariates=X)
    bo = BayesianOptimization(
        partial_filter, hparams, random_state=random_state,
    )

    return bo, filterfn_jit


def eval_filterfn_collection(filterfn, hparams, X_collection, y_collection):
    hist_time = []
    hist_metric = []
    n_runs = len(X_collection)
    for yc, Xc in tqdm(zip(y_collection, X_collection), total=n_runs):
        tinit = time()
        run = filterfn(**hparams, measurements=yc, covariates=Xc)
        run = jax.block_until_ready(run)
        tend = time()

        hist_time.append(tend - tinit)
        hist_metric.append(run)
    
    hist_metric = np.stack(hist_metric)
    return hist_time, hist_metric
    

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
filterfn_name = fileter_fns[method]
hparams = config_search[method]["learn"]
hparams_static = config_search[method]["static"]

# There must be a better way to do this
hparams_static["observation_covariance"] = jnp.eye(1) * hparams_static["observation_covariance"]
bo, filterfn = build_bopt_step(
        filterfn_name, hparams, hparams_static, params_init,
        random_state, y, X, measurement_fn, latent_fn,
)
bo.maximize()

hparams = bo.max["params"]
hist_times, hist_metrics = eval_filterfn_collection(
    filterfn, hparams, X_collection, y_collection
)
