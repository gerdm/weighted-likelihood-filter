import jax
import sys
import toml
import pickle
import numpy as np
import flax.linen as nn
from tqdm import tqdm

# local imports
import datagen
import experiments_main as experiment


if __name__ == "__main__":
    config_path, n_runs = sys.argv[1:]

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
def latent_fn(x): return x
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

    for method in (pbar := tqdm(hyperparams, leave=False)):
        pbar.set_description(f"p_error: {p100}, method: {method}")
        hparams = hyperparams[method]
        time_collection, pp_est = experiment.eval_filterfn_collection(
            experiment.filter_fns[method], hparams, X_collection, y_collection
        )
        errs = np.power(pp_est - y_collection, 2)
        p_errors_collection[method] = errs
        time_collection_all[method] = time_collection


res = {
    "p_errors_collection": p_errors_collection,
    "time_collection": time_collection_all,
    "config": config,
}

filename = f"./results/regression-stress-{dataset_name}.pkl"
with open(filename, "wb") as f:
    pickle.dump(res, f)