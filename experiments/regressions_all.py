import os
import jax
import toml
import pickle
import numpy as np
import pandas as pd
import jax.numpy as jnp
import flax.linen as nn

# local imports 
import datagen
import regression_main as regressions

uci = datagen.UCIDatasets("./data")

n_runs = 100
noise_type = "target"

# Load config
config_path_search = "./configs/uci-hparam-search.toml"
p_error = 0.1


with open(config_path_search, "r") as f:
    config_search = toml.load(f)

noise_type = "target" # or "covariate"

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

key = jax.random.PRNGKey(314)
for dataset_name in uci.datasets:
    hist_methods = {}
    time_methods = {}
    configs = {}
    print("*" * 80)
    print(f"Dataset: {dataset_name}")
    X_collection, y_collection, ixs_collection = datagen.create_uci_collection(
        dataset_name, noise_type=noise_type, p_error=p_error, n_runs=n_runs,
        v_error=50, seed_init=314, path="./data"
    )

    X, y = X_collection[0], y_collection[0]
    random_state = config_search["shared"]["random_state"]
    params_init = model.init(key, X[:1])

    for method in regressions.filter_fns:
        print(f"Method: {method}")
        config_method = config_search[method]
        filterfn_name = regressions.filter_fns[method]
        hparams = config_method["learn"]
        hparams_static = config_method.get("static", {})

        # There must be a better way to do this
        if "observation_covariance" in hparams_static:
            hparams_static["observation_covariance"] = jnp.eye(1) * hparams_static["observation_covariance"]
        bo, filterfn = regressions.build_bopt_step(
                filterfn_name, hparams, hparams_static, params_init,
                random_state, y, X, measurement_fn, latent_fn,
        )
        bo.maximize()

        hparams = bo.max["params"]
        hist_times, hist_metrics = regressions.eval_filterfn_collection(
            filterfn, hparams, X_collection, y_collection
        )

        hist_methods[method] = hist_metrics
        time_methods[method] = hist_times
        configs[method] = hparams


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
    print(rmedse_df.groupby("method").median()["err"])
    
    data = {
        "datasets": {
            "X": np.array(X_collection),
            "y": np.array(y_collection),
        },
        "time": {k: np.array(v) for k, v in time_methods.items()},
        "pp_estimates": hist_methods,
        "configs": configs,
        "dataset-name": dataset_name,
        "p-error": p_error,
    }


    p_error_str = format(p_error * 100, "0.0f")
    filename = f"{dataset_name}-{noise_type}-p-error{p_error_str}.pkl"
    path = os.path.join("results", filename)
    print(f"Storing in {path}")
    with open(path, "wb") as f:
        pickle.dump(data, f)
