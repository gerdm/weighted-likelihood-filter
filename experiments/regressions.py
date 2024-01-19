#!/usr/bin/env python
# coding: utf-8

# # UCI regression with outliers

import os
import sys
import jax
import optax
import pickle
import datagen
import pandas as pd
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
p_error = float(sys.argv[2]) / 100
n_runs = int(sys.argv[3])

uci = datagen.UCIDatasets("./data")

init_points = 10
n_iter = 15

print("*" * 80)
print(f"Dataset: {dataset_name}")
# ## Load dataset

noise_type = "target" # or "covariate"

X_collection= []
y_collection = []
ix_clean_collection = []

v_error = 50
seed_init = 314
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


# In[5]:

n_samples = y.shape[0]
X_collection = jnp.array(X_collection)
y_collection = jnp.array(y_collection)
mask_clean = np.array(ix_clean_collection).T
X_collection.shape


# ## Setup

# In[6]:


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(20)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


# In[7]:


def callback_fn(bel, bel_pred, y, x, applyfn):
    yhat = applyfn(bel_pred.mean, x[None])
    return yhat


# In[8]:


y, X = y_collection[0], X_collection[0]
ix_clean = ix_clean_collection[0]


# In[9]:


Q = 0.0
observation_covariance = 1.0


# In[10]:


key = jax.random.PRNGKey(314)
model = MLP()
def latent_fn(x): return x
measurement_fn = model.apply
params_init = model.init(key, X[:1])


# # Run experiments

# In[11]:


time_methods = {}
hist_methods = {}
configs = {}


# In[12]:


observation_covariance = jnp.eye(1) * 1.0


# *********************************** KF  ******************************************
print("-" * 20, "KF", "-" * 20)

# In[13]:


@jax.jit
def filter_kf(log_lr, measurements, covariates):
    lr = jnp.exp(log_lr)
    nsteps = len(measurements)
    agent = rkf.ExtendedKalmanFilterIMQ(
        latent_fn, measurement_fn,
        dynamics_covariance=Q,
        observation_covariance=observation_covariance,
        soft_threshold=1e8,
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    bel, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    # out = (agent, bel)
    return yhat_pp.squeeze()


@jax.jit
def opt_step(log_lr):
    yhat_pp = filter_kf(log_lr, y, X)
    err = jnp.power(yhat_pp - y, 2)
    err = jnp.median(err)
    err = jax.lax.cond(jnp.isnan(err), lambda: 1e6, lambda: err)
    return -err


# In[14]:


bo = BayesianOptimization(
    opt_step,
    pbounds={
        "log_lr": (-5, 0),
    },
    random_state=314,
    allow_duplicate_points=True
)

bo.maximize(init_points=init_points, n_iter=n_iter)


# In[15]:


method = "KF"
log_lr = bo.max["params"]["log_lr"]
configs[method] = bo.max["params"]

hist_bel = []
times = []

for yc, Xc in tqdm(zip(y_collection, X_collection), total=n_runs): 
    tinit = time()
    run = filter_kf(log_lr, y, X)
    run = jax.block_until_ready(run)
    tend = time()
    
    hist_bel.append(run)
    times.append(tend - tinit)

hist_bel = np.stack(hist_bel)

hist_methods[method] = hist_bel
time_methods[method] = times


# *********************************** KF-B  ******************************************
print("-" * 20, "KF-B", "-" * 20)

# In[16]:


@jax.jit
def filter_kfb(log_lr, alpha, beta, n_inner, measurements, covariates):
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
    bel, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()


@jax.jit
def opt_step(log_lr, alpha, beta, n_inner):
    yhat_pp = filter_kfb(log_lr, alpha, beta, n_inner, y, X)
    err = jnp.power(yhat_pp - y, 2)
    err = jnp.median(err)
    err = jax.lax.cond(jnp.isnan(err), lambda: 1e6, lambda: err)
    return -err


# In[17]:


bo = BayesianOptimization(
    opt_step,
    pbounds={
        "log_lr": (-5, 0),
        "alpha": (0.0, 5.0),
        "beta": (0.0, 5.0),

        "n_inner":  (1, 10),
    },
    random_state=314,
    
)
bo.maximize(init_points=init_points, n_iter=n_iter)


# In[18]:


method = "KF-B"
configs[method] = bo.max["params"]

hist_bel = []
times = []

for yc, Xc in tqdm(zip(y_collection, X_collection), total=n_runs): 
    tinit = time()
    run = filter_kfb(**bo.max["params"], measurements=y, covariates=X)
    run = jax.block_until_ready(run)
    tend = time()
    
    hist_bel.append(run)
    times.append(tend - tinit)

hist_bel = np.stack(hist_bel)

hist_methods[method] = hist_bel
time_methods[method] = times


# *********************************** KF-IW  ******************************************
print("-" * 20, "KF-IW", "-" * 20)


@jax.jit
def filter_kfiw(log_lr, noise_scaling, n_inner, measurements, covariates):
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
    bel, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()


@jax.jit
def opt_step(log_lr, noise_scaling, n_inner):
    yhat_pp = filter_kfiw(log_lr, noise_scaling, n_inner, y, X)
    err = jnp.power(yhat_pp - y, 2)
    err = jnp.median(err)
    err = jax.lax.cond(jnp.isnan(err), lambda: 1e6, lambda: err)
    return -err


# In[20]:


bo = BayesianOptimization(
    opt_step,
    pbounds={
        "log_lr": (-5, 0),
        "noise_scaling": (1e-6, 20),
        "n_inner":  (1, 10),
    },
    random_state=314,
)
bo.maximize(init_points=init_points, n_iter=n_iter)


# In[21]:


method = "KF-IW"
configs[method] = bo.max["params"]

hist_bel = []
times = []

for yc, Xc in tqdm(zip(y_collection, X_collection), total=n_runs): 
    tinit = time()
    run = filter_kfiw(**bo.max["params"], measurements=y, covariates=X)
    run = jax.block_until_ready(run)
    tend = time()
    
    hist_bel.append(run)
    times.append(tend - tinit)

hist_bel = np.stack(hist_bel)

hist_methods[method] = hist_bel
time_methods[method] = times


# *********************************** WLF-IMQ  ******************************************
print("-" * 20, "WLF-IMQ", "-" * 20)


@jax.jit
def filter_wlfimq(log_lr, soft_threshold, measurements, covariates):
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
    bel, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    # out = (agent, bel)
    return yhat_pp.squeeze()


@jax.jit
def opt_step(log_lr, soft_threshold):
    yhat_pp = filter_wlfimq(log_lr, soft_threshold, y, X)
    err = jnp.power(yhat_pp - y, 2)
    err = jnp.median(err)
    err = jax.lax.cond(jnp.isnan(err), lambda: 1e6, lambda: err)
    return -err


# In[23]:


bo = BayesianOptimization(
    opt_step,
    pbounds={
        "log_lr": (-5, 0),
        "soft_threshold": (1e-6, 20)
    },
    random_state=314,
    allow_duplicate_points=True
)

bo.maximize(init_points=init_points, n_iter=n_iter)


# In[24]:


method = "WLF-IMQ"
log_lr = bo.max["params"]["log_lr"]
configs[method] = bo.max["params"]

hist_bel = []
times = []

for yc, Xc in tqdm(zip(y_collection, X_collection), total=n_runs): 
    tinit = time()
    run = filter_wlfimq(**bo.max["params"], measurements=y, covariates=X)
    run = jax.block_until_ready(run)
    tend = time()
    
    hist_bel.append(run)
    times.append(tend - tinit)

hist_bel = np.stack(hist_bel)

hist_methods[method] = hist_bel
time_methods[method] = times


# *********************************** WLF-MD  ******************************************
print("-" * 20, "WLF-MD", "-" * 20)

# In[162]:


@jax.jit
def filter_wlfmd(log_lr, threshold, measurements, covariates):
    lr = jnp.exp(log_lr)
    nsteps = len(measurements)
    agent = rkf.ExtendedKalmanFilterMD(
        latent_fn, measurement_fn,
        dynamics_covariance=Q,
        observation_covariance=observation_covariance,
        threshold=threshold,
    )
    
    init_bel = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    bel, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    # out = (agent, bel)
    return yhat_pp.squeeze()


@jax.jit
def opt_step(log_lr, threshold):
    yhat_pp = filter_wlfmd(log_lr, threshold, y, X)
    err = jnp.power(yhat_pp - y, 2)
    err = jnp.median(err)
    err = jax.lax.cond(jnp.isnan(err), lambda: 1e6, lambda: err)
    return -err


# In[163]:


bo = BayesianOptimization(
    opt_step,
    pbounds={
        "log_lr": (-5, 0),
        "threshold": (1e-6, 20)
    },
    random_state=314,
    allow_duplicate_points=True
)

bo.maximize(init_points=init_points, n_iter=n_iter)


# In[164]:


method = "WLF-MD"
configs[method] = bo.max["params"]

hist_bel = []
times = []

for yc, Xc in tqdm(zip(y_collection, X_collection), total=n_runs): 
    tinit = time()
    run = filter_wlfmd(**bo.max["params"], measurements=y, covariates=X)
    run = jax.block_until_ready(run)
    tend = time()
    
    hist_bel.append(run)
    times.append(tend - tinit)

hist_bel = np.stack(hist_bel)

hist_methods[method] = hist_bel
time_methods[method] = times


# *********************************** OGD  ******************************************
print("-" * 20, "OGD", "-" * 20)


def lossfn(params, counter, x, y, applyfn):
    yhat = applyfn(params, x)
    return jnp.sum(counter * (y - yhat) ** 2) / counter.sum()


@jax.jit
def filter_ogd(log_lr, n_inner, measurements, covariates):
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


@jax.jit
def opt_step(log_lr, n_inner):
    yhat_pp = filter_ogd(log_lr, n_inner, y, X)
    err = jnp.power(yhat_pp - y, 2)
    err = jnp.median(err)
    err = jax.lax.cond(jnp.isnan(err), lambda: 1e6, lambda: err)
    return -err


# In[146]:


bo = BayesianOptimization(
    opt_step,
    pbounds={
        "log_lr": (-5, 0),
        "n_inner": (1, 10),
    },
    random_state=314,
    allow_duplicate_points=True
)

bo.maximize(init_points=init_points, n_iter=n_iter)


# In[147]:


method = "OGD"
configs[method] = bo.max["params"]

hist_bel = []
times = []

for yc, Xc in tqdm(zip(y_collection, X_collection), total=n_runs): 
    tinit = time()
    run = filter_ogd(**bo.max["params"], measurements=y, covariates=X)
    run = jax.block_until_ready(run)
    tend = time()
    
    hist_bel.append(run)
    times.append(tend - tinit)

hist_bel = np.stack(hist_bel)

hist_methods[method] = hist_bel
time_methods[method] = times

# *********************************** WLF-OGD  ******************************************
print("-" * 20, "WLF-OGD", "-" * 20)
@jax.jit
def filter_wlf_ogd(log_lr, soft_threshold, n_inner, measurements, covariates):
    lr = jnp.exp(log_lr)
    n_inner = n_inner.astype(int)
    
    agent = rkf.FifoSGDIMQ(
        measurement_fn,
        optax.adam(lr),
        buffer_size=1,
        dim_features=covariates.shape[-1],
        dim_output=1,
        soft_threshold=soft_threshold,
        n_inner=n_inner,
    )
    
    callback = partial(callback_fn, applyfn=measurement_fn)
    
    init_bel = agent.init_bel(params_init)
    _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
    
    return yhat_pp.squeeze()


@jax.jit
def opt_step(log_lr, soft_threshold, n_inner):
    yhat_pp = filter_wlf_ogd(log_lr, soft_threshold, n_inner, y, X)
    err = jnp.power(yhat_pp - y, 2)
    err = jnp.median(err)
    err = jax.lax.cond(jnp.isnan(err), lambda: 1e6, lambda: err)
    return -err


bo = BayesianOptimization(
    opt_step,
    pbounds={
        "log_lr": (-10, 0),
        "soft_threshold": (1e-6, 20),
        "n_inner": (1, 10),
    },
    random_state=314,
    allow_duplicate_points=True
)

bo.maximize(init_points=init_points, n_iter=n_iter)


method = "WLF-OGD"
configs[method] = bo.max["params"]

hist_bel = []
times = []

for yc, Xc in tqdm(zip(y_collection, X_collection), total=n_runs): 
    tinit = time()
    run = filter_wlf_ogd(**bo.max["params"], measurements=y, covariates=X)
    run = jax.block_until_ready(run)
    tend = time()
    
    hist_bel.append(run)
    times.append(tend - tinit)

hist_bel = np.stack(hist_bel)

hist_methods[method] = hist_bel
time_methods[method] = times


# ***************************************************************************************
# *********************************** SUMMARY  ******************************************
# ***************************************************************************************


rmedse_df = pd.DataFrame(jax.tree_map(
    lambda x: np.sqrt(np.median(np.power(x - y_collection, 2), 1)),
    hist_methods
))
rmedse_df = rmedse_df.reset_index().melt("index")
rmedse_df = rmedse_df.rename({
    "index": "run",
    "variable": "method",
    "value": "err"
}, axis=1)
rmedse_df.head()


# In[169]:


time_df = pd.DataFrame(time_methods).reset_index().melt("index")
time_df = time_df.rename({
    "index": "run",
    "variable": "method",
    "value": "time"
}, axis=1)

time_df.head()


# In[170]:


df = rmedse_df.merge(time_df, on=["method", "run"]).query("run > 0")


# ## Store data

# In[193]:


print(df.groupby("method").median().drop("run", axis=1))


# In[194]:


data = {
    "datasets": {
        "X": np.array(X_collection),
        "y": np.array(y_collection),
    },
    "time": {k: np.array(v) for k, v in time_methods.items()},
    "posterior-states": hist_methods,
    "config": configs,
    "dataset-name": dataset_name,
    "p-error": p_error,
}


# In[195]:


p_error_str = format(p_error * 100, "0.0f")
filename = f"{dataset_name}-{noise_type}-p-error{p_error_str}.pkl"
path = os.path.join("results", filename)
print(f"Storing in {path}")
with open(path, "wb") as f:
    pickle.dump(data, f)
