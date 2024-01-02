#!/usr/bin/env python
# coding: utf-8

# # UCI regression 

# In[479]:

import sys

import jax
import optax
import pickle
import numpy as np
import pandas as pd
import flax.linen as nn
import jax.numpy as jnp

import datagen
from functools import partial
from bayes_opt import BayesianOptimization
from jax.sharding import PositionalSharding
from rebayes_mini.methods import replay_sgd
from rebayes_mini.methods import gauss_filter as gfilter
from rebayes_mini.methods import robust_filter as rfilter
from rebayes_mini.methods import generalised_bayes_filter as gbfilter


dataset_name = sys.argv[1]
p_error = float(sys.argv[2]) / 100
n_runs = int(sys.argv[3])


devices = jax.devices()
sharding = PositionalSharding(devices)



uci = datagen.UCIDatasets("./data")
print("-" * 80)
print(f"Dataset: {dataset_name}")
print(f"p_error: {p_error}")
print(f"n_runs: {n_runs}")


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


# In[928]:


X_collection = jnp.array(X_collection)
y_collection = jnp.array(y_collection)
mask_clean = np.array(ix_clean_collection).T
X_collection.shape

# In[929]:


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(20)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


# # Setup

# In[930]:


def callback_fn(bel, bel_pred, y, x, applyfn):
    yhat = applyfn(bel_pred.mean, x[None])
    return yhat


# In[931]:


y, X = y_collection[0], X_collection[0]
ix_clean = ix_clean_collection[0]


# In[932]:


Q = 0.0
observation_covariance = 1.0


# In[933]:


key = jax.random.PRNGKey(314)
model = MLP()
params_init = model.init(key, X[:1])


# In[934]:


X_collection.shape


# ## EKF

# In[935]:


def filter_ekf(log_lr):
    lr = np.exp(log_lr)
    agent = gfilter.ExtendedKalmanFilter(
        lambda x: x,
        model.apply,
        dynamics_covariance=Q,
        observation_covariance=observation_covariance  * jnp.eye(1),
    )

    bel_init = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.vobs_fn)
    bel_imq, yhat_pp = agent.scan(bel_init, y, X, callback_fn=callback)
    out = (agent, bel_imq)
    return yhat_pp.squeeze(), out

def opt_step(log_lr):
    res = -jnp.power(filter_ekf(log_lr)[0] - y, 2)
    res = np.median(res)
    
    if np.isnan(res) or np.isinf(res):
        res = -1e+6
    
    return res


# In[936]:


get_ipython().run_cell_magic('time', '', 'bo = BayesianOptimization(\n    opt_step,\n    pbounds={\n        "log_lr": (-5, 0),\n    },\n    verbose=1,\n    random_state=314,\n    allow_duplicate_points=True\n)\n\nbo.maximize(init_points=5, n_iter=5)\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nlr = np.exp(bo.max["params"]["log_lr"])\nagent = gfilter.ExtendedKalmanFilter(\n    lambda x: x,\n    model.apply,\n    dynamics_covariance=Q,\n    observation_covariance=1.0 * jnp.eye(1),\n)\n\nbel_init = agent.init_bel(params_init, cov=lr)\n\ncallback = partial(callback_fn, applyfn=agent.vobs_fn)\nscanfn = jax.vmap(agent.scan, in_axes=(None, 0, 0, None))\nres = scanfn(bel_init, y_collection, X_collection, callback)\n\nres = jax.block_until_ready(res)\nstate_final_collection, yhat_collection_ekf = res\nyhat_collection_ekf = yhat_collection_ekf.squeeze()\n')


# In[ ]:


err_collection_ekf = pd.DataFrame(np.power(y_collection - yhat_collection_ekf, 2).T)


# ## WLF-IMQ

# ### Hparam choice

# In[ ]:


def filter_imqf(soft_threshold, log_lr):
    lr = np.exp(log_lr)
    agent = gbfilter.IMQFilter(
        model.apply, dynamics_covariance=Q,
        observation_covariance=observation_covariance,
        soft_threshold=soft_threshold
    )

    bel_init = agent.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent.link_fn)
    bel_imq, yhat_pp = agent.scan(bel_init, y, X, callback_fn=callback)
    out = (agent, bel_imq)
    return yhat_pp.squeeze(), out

def opt_step(soft_threshold, log_lr):
    # res = -jnp.power(filter_imqf(soft_threshold, log_lr)[0] - y, 2)[ix_clean].mean()
    res = -jnp.power(filter_imqf(soft_threshold, log_lr)[0] - y, 2)
    res = jnp.median(res)
    
    if np.isnan(res) or np.isinf(res):
        res = -1e+6
    
    return res


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bo = BayesianOptimization(\n    opt_step,\n    pbounds={\n        "soft_threshold": (1e-6, 15),\n        "log_lr": (-5, 0),\n    },\n    verbose=1,\n    random_state=314,\n    allow_duplicate_points=True\n)\n\nbo.maximize(init_points=5, n_iter=5)\n')


# ### Eval

# In[ ]:


soft_threshold = bo.max["params"]["soft_threshold"]
lr = np.exp(bo.max["params"]["log_lr"])

agent = gbfilter.IMQFilter(
    model.apply,
    dynamics_covariance=0.0,
    observation_covariance=1.0,
    soft_threshold=soft_threshold
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bel_init = agent.init_bel(params_init, cov=lr)\ncallback = partial(callback_fn, applyfn=agent.link_fn)\n_, yhat_collection_wlf = jax.vmap(agent.scan, in_axes=(None, 0, 0, None))(bel_init, y_collection, X_collection, callback)\nyhat_collection_wlf = jax.block_until_ready(yhat_collection_wlf.squeeze())\n')


# In[ ]:


err_collection_wlf = pd.DataFrame(np.power(y_collection - yhat_collection_wlf, 2).T)


# ## IW-based EKF
# (Agamenoni 2012)

# ### Hparam choice

# In[ ]:


def filter_rkf(noise_scaling, log_lr):
    lr = np.exp(log_lr)
    agent_rekf = rfilter.ExtendedRobustKalmanFilter(
        lambda x: x, model.apply, dynamics_covariance=Q,
        prior_observation_covariance=observation_covariance * jnp.eye(1),
        noise_scaling=noise_scaling,
        n_inner=1
    )
    
    bel_init = agent_rekf.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent_rekf.vobs_fn)
    bel_rekf, yhat_pp = agent_rekf.scan(bel_init, y, X, callback_fn=callback)
    out = (agent_rekf, bel_rekf)
    
    return yhat_pp.squeeze(), out


# In[ ]:


def opt_step(noise_scaling, log_lr):
    # res = -jnp.power(filter_rkf(noise_scaling, log_lr)[0] - y, 2)[ix_clean].mean()
    res = -jnp.power(filter_rkf(noise_scaling, log_lr)[0] - y, 2)
    res = jnp.median(res)
    if np.isnan(res) or np.isinf(res):
        res = -1e+6
    
    return res


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bo = BayesianOptimization(\n    opt_step,\n    pbounds={\n        "noise_scaling": (1e-6, 15),\n        "log_lr": (-5, 0)\n    },\n    verbose=1,\n    random_state=314,\n    allow_duplicate_points=True\n)\n\nbo.maximize(init_points=5, n_iter=5)\n')


# ### Eval

# In[ ]:


noise_scaling = bo.max["params"]["noise_scaling"]
lr = np.exp(bo.max["params"]["log_lr"])

agent = rfilter.ExtendedRobustKalmanFilter(
    lambda x: x,
    model.apply,
    dynamics_covariance=0.0,
    prior_observation_covariance=1.0 * jnp.eye(1),
    n_inner=1,
    noise_scaling=noise_scaling,
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bel_init = agent.init_bel(params_init, cov=lr)\ncallback = partial(callback_fn, applyfn=agent.vobs_fn)\n_, yhat_collection_ann1 = jax.vmap(agent.scan, in_axes=(None, 0, 0, None))(bel_init, y_collection, X_collection, callback)\nyhat_collection_ann1 = yhat_collection_ann1.squeeze()\n')


# In[ ]:


err_collection_ann1 = pd.DataFrame(np.power(y_collection - yhat_collection_ann1, 2).T)


# ## WLF-MD
# Weighted likelihood filter with Mahalanobis distance thresholding weighting function

# ### Hparam section

# In[ ]:


def filter_mah_ekf(log_lr, threshold):
    lr = np.exp(log_lr)
    agent_mekf = rfilter.ExtendedThresholdedKalmanFilter(
        lambda x: x, model.apply,
        dynamics_covariance=Q,
        observation_covariance=observation_covariance * jnp.eye(1),
        threshold=threshold
    )
    
    bel_init = agent_mekf.init_bel(params_init, cov=lr)
    callback = partial(callback_fn, applyfn=agent_mekf.vobs_fn)
    
    bel_mekf, yhat_pp = agent_mekf.scan(bel_init, y, X, callback_fn=callback)
    out = (agent_mekf, bel_mekf)
    return yhat_pp.squeeze(), out

def opt_step(log_lr, threshold):
    # res = -jnp.power(filter_rkf(noise_scaling, log_lr)[0] - y, 2)[ix_clean].mean()
    res = -jnp.power(filter_mah_ekf(noise_scaling, log_lr)[0] - y, 2)
    res = jnp.median(res)
    if np.isnan(res) or np.isinf(res):
        res = -1e+6
    
    return res


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bo = BayesianOptimization(\n    opt_step,\n    pbounds={\n        "threshold": (1e-6, 15),\n        "log_lr": (-5, 0)\n    },\n    verbose=1,\n    random_state=314,\n    allow_duplicate_points=True\n)\n\nbo.maximize(init_points=5, n_iter=5)\n')


# ### Eval

# In[ ]:


threshold = bo.max["params"]["threshold"]
lr = np.exp(bo.max["params"]["log_lr"])

agent = rfilter.ExtendedThresholdedKalmanFilter(
    lambda x: x,
    model.apply,
    dynamics_covariance=0.0,
    observation_covariance=1.0 * jnp.eye(1),
    threshold=threshold,
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bel_init = agent.init_bel(params_init, cov=lr)\nbel_init = jax.device_put(bel_init, sharding.replicate(0))\ncallback = partial(callback_fn, applyfn=agent.vobs_fn)\nscanfn = jax.jit(jax.vmap(agent.scan, in_axes=(None, 0, 0, None)), static_argnames=("callback_fn",))\n\n_, yhat_collection_mekf = scanfn(bel_init, y_collection, X_collection, callback)\nyhat_collection_mekf = jax.block_until_ready(yhat_collection_mekf)\nyhat_collection_mekf = yhat_collection_mekf.squeeze()\n')


# In[ ]:


err_collection_mekf = pd.DataFrame(np.power(y_collection - yhat_collection_mekf, 2).T)


# ## Online SGD

# In[ ]:


def lossfn(params, counter, x, y, applyfn):
    yhat = applyfn(params, x)
    return jnp.sum(counter * (y - yhat) ** 2) / counter.sum()

def filter_ogd(log_lr, n_inner):
    lr = np.exp(log_lr)
    n_inner = int(n_inner)
    
    agent = replay_sgd.FifoSGD(
        model.apply,
        lossfn,
        optax.adam(lr),
        buffer_size=1,
        dim_features=X.shape[-1],
        dim_output=1,
        n_inner=n_inner,
    )

    callback = partial(callback_fn, applyfn=model.apply)

    bel_init = agent.init_bel(params_init)
    bel_final, yhat_pp = agent.scan(bel_init, y, X, callback)
    out = (agent, bel_final)
    yhat_pp = yhat_pp.squeeze()

    return yhat_pp.squeeze(), out

def opt_step(log_lr, n_inner):
    res = -jnp.power(filter_ogd(log_lr, n_inner)[0] - y, 2)
    res = jnp.median(res)
    if np.isnan(res) or np.isinf(res):
        res = -1e+6
    
    return res


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bo = BayesianOptimization(\n    opt_step,\n    pbounds={\n        "log_lr": (-5, 0),\n        "n_inner": (1, 10),\n    },\n    verbose=1,\n    random_state=314,\n    allow_duplicate_points=True\n)\n\nbo.maximize(init_points=5, n_iter=5)\n')


# In[ ]:


lr = jnp.exp(bo.max["params"]["log_lr"])
n_inner = int(bo.max["params"]["n_inner"])

agent = replay_sgd.FifoSGD(
    model.apply,
    lossfn,
    optax.adam(lr),
    buffer_size=1,
    dim_features=X.shape[-1],
    dim_output=1,
    n_inner=n_inner
)

callback = partial(callback_fn, applyfn=model.apply)

bel_init = agent.init_bel(params_init)
state_final, yhat = agent.scan(bel_init, y, X, callback)
yhat = yhat.squeeze()

errs = (y - yhat)
jnp.sqrt(jnp.power(errs, 2).mean())


# In[ ]:


get_ipython().run_cell_magic('time', '', '_, yhat_collection_ogd = jax.vmap(agent.scan, in_axes=(None, 0, 0, None))(bel_init, y_collection, X_collection, callback)\nyhat_collection_ogd = jax.block_until_ready(yhat_collection_ogd)\nyhat_collection_ogd = yhat_collection_ogd.squeeze()\n')


# In[ ]:


err_collection_ogd  = pd.DataFrame(np.power(y_collection - yhat_collection_ogd, 2).T)


# # Results

# In[ ]:


import seaborn as sns


# In[ ]:


pd.set_option("display.float_format", lambda x: format(x, "0.4f"))



df_results = pd.DataFrame({
    "WLF-IMQ": err_collection_wlf.median(axis=0),
    "EKF": err_collection_ekf.median(axis=0),
    "OGD": err_collection_ogd.median(axis=0),
    "WLF-MD": err_collection_mekf.median(axis=0),
    "E-ANN-1": err_collection_ann1.median(axis=0),
})

print(df_results.describe())


# In[ ]:



# In[ ]:


err_collection = {
    "methods": {
        "WLF-IMQ": err_collection_wlf,
        "EKF": err_collection_ekf,
        "OGD": err_collection_ogd,
        "WLF-MD": err_collection_mekf,
        "E-ANN-1": err_collection_ann1
    },
    "config": {
        "mask-clean": mask_clean,
        "p_error": p_error,
    }
}



with open(f"./results/{dataset_name}-{noise_type}-p-error{p_error * 100:02.0f}.pkl", "wb") as f:
    pickle.dump(err_collection, f)


p_error