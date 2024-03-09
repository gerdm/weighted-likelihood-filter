import jax
import sys
import toml
import pickle
import numpy as np
import flax.linen as nn
from tqdm import tqdm
import jax.numpy as jnp

# local imports
import datagen
import experiments_main as experiment


config_path, n_runs = sys.argv[1:]
n_runs = int(n_runs)

config_path_search = "./configs/uci-hparam-search.toml"


with open(config_path_search, "r") as f:
    config_search = toml.load(f)


with open(config_path) as f:
    config = toml.load(f)


p_errors_collection = {}
time_collection_all = {}


noise_type = "target" # or "covariate"

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(20)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

model = MLP()
def state_fn(x): return x
measurement_fn = model.apply

hyperparams = config["hyperparams"]

key = jax.random.PRNGKey(314)
p_errors = np.arange(0, 0.5, 0.05)
dataset_name = config["metadata"]["dataset-name"]

p_errors_collection = {}
time_collection_all = {}
for p_error in tqdm(p_errors):
    p100 = int(p_error * 100)
    X_collection, y_collection, ixs_collection = datagen.create_uci_collection(
        dataset_name, noise_type=noise_type, p_error=p_error, n_runs=n_runs,
        v_error=50, seed_init=314, path="./data"
    )

    X0, y0 = X_collection[0], y_collection[0]
    params_init = model.init(key, X0[:1])

    errs_methods = {} 
    time_methods = {}
    for method in (pbar := tqdm(hyperparams, leave=False)):

        config_method = config_search[method]
        hparams_static = config_method.get("static", {})

        if "observation_covariance" in hparams_static:
            hparams_static["observation_covariance"] = jnp.eye(1) * hparams_static["observation_covariance"]

        @jax.jit
        def filterfn_jit(measurements, covariates, **hparams):
            filterfn = experiment.filter_fns[method]
            yhat_pp = filterfn(
                **hparams, **hparams_static,
                params_init=params_init, measurements=measurements, covariates=covariates,
                measurement_fn=measurement_fn, state_fn=state_fn
            )
            return yhat_pp

        pbar.set_description(f"p_error: {p100}, method: {method}")
        hparams = hyperparams[method]
        time_collection, pp_est = experiment.eval_filterfn_collection(
            filterfn_jit, hparams, X_collection, y_collection
        )
        errs = pp_est - y_collection
        errs_methods[method] = errs
        time_methods[method] = time_collection


    p_errors_collection[p100] = errs_methods
    time_collection_all[p100] = time_methods


res = {
    "p_errors_collection": p_errors_collection,
    "time_collection": time_collection_all,
    "config": config,
}

filename = f"./results/regression-stress-{dataset_name}.pkl"
with open(filename, "wb") as f:
    pickle.dump(res, f)