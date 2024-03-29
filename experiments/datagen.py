import os
import jax
import numpy as np
import pandas as pd
import jax.numpy as jnp


class MovingObject2D:
    def __init__(
        self, sampling_period, dynamics_covariance, observation_covariance,
    ):
        self.sampling_period = sampling_period
        self.dynamics_covariance = dynamics_covariance * jnp.eye(4)
        self.observation_covariance = observation_covariance * jnp.eye(2)
        self.transition_matrix, self.projection_matrix = self._init_dynamics(sampling_period)
        self.dim_obs, self.dim_latent = self.projection_matrix.shape

    def _init_dynamics(self, sampling_period):
        transition_matrix = jnp.array([
            [1, 0, sampling_period, 0],
            [0, 1, 0, sampling_period],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        projection_matrix = jnp.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        return transition_matrix, projection_matrix


    def step(self, z_prev, key):
        key_latent, key_obs = jax.random.split(key)
        z_next = jax.random.multivariate_normal(
            key_latent,
            mean=self.transition_matrix @ z_prev,
            cov=self.dynamics_covariance,
        )
        x_next = jax.random.multivariate_normal(
            key_obs,
            mean=self.projection_matrix @ z_next,
            cov=self.observation_covariance,
        )

        output = {
            "observed": x_next,
            "latent": z_next,
        }

        return z_next, output


    def sample(self, key, z0, n_steps):
        keys = jax.random.split(key, n_steps)
        _, output = jax.lax.scan(self.step, z0, keys)
        return output


class GaussStMovingObject2D(MovingObject2D):
    def __init__(
        self, sampling_period, dynamics_covariance, observation_covariance,
        dof_observed,
    ):
        super().__init__(sampling_period, dynamics_covariance, observation_covariance)
        self.dof_observed = dof_observed

    def sample_multivariate_t(self, key, mean, covariance, df):
        key_gamma, key_norm = jax.random.split(key)
        dim = len(mean)
        zeros = jnp.zeros(dim)
        err = jax.random.multivariate_normal(key_norm, mean=zeros, cov=covariance)
        shape, rate = df / 2, df / 2
        w  = jax.random.gamma(key_gamma, shape=(1,), a=shape) / rate

        x = mean + err / jnp.sqrt(w)
        return x

    def step(self, z_prev, key):
        key_latent, key_obs = jax.random.split(key, 2)
        z_next = jax.random.multivariate_normal(
            key_latent,
            mean=self.transition_matrix @ z_prev,
            cov=self.dynamics_covariance,
        )

        x_next = self.sample_multivariate_t(
            key_obs,
            mean=self.projection_matrix @ z_next,
            covariance=self.observation_covariance,
            df=self.dof_observed,
        )

        output = {
            "observed": x_next,
            "latent": z_next,
        }

        return z_next, output


class GaussOutlierMovingObject2D(MovingObject2D):
    """
    Tracking object moving in 2D space with constant velocity.
    The latent space is taken to be Gaussian and
    the observed space is taken to be Gaussian with
    covariance matrix scaled by `outlier_scale` with probability
    `outlier_proba`
    """
    def __init__(
        self, sampling_period, dynamics_covariance, observation_covariance, outlier_proba, outlier_scale
    ):
        super().__init__(sampling_period, dynamics_covariance, observation_covariance)
        self.outlier_proba = outlier_proba
        self.outlier_scale = outlier_scale

    def step(self, z_prev, key):
        key_latent, key_obs, key_corruped = jax.random.split(key, 3)
        z_next = jax.random.multivariate_normal(
            key_latent,
            mean=self.transition_matrix @ z_prev,
            cov=self.dynamics_covariance,
        )

        # Corrupt with outliers
        corrupted = (jax.random.uniform(key_corruped) < self.outlier_proba).astype(jnp.float32)
        mean_next = self.projection_matrix @ z_next
        cov_next = (
            self.observation_covariance * self.outlier_scale * corrupted +
            self.observation_covariance * (1 - corrupted)
        )

        x_next = jax.random.multivariate_normal(
            key_obs,
            mean=mean_next,
            cov=cov_next,
        )

        output = {
            "observed": x_next,
            "latent": z_next,
        }

        return z_next, output


class GaussMeanOutlierMovingObject2D(MovingObject2D):
    """
    Tracking object moving in 2D space with constant velocity.
    The latent space is taken to be Gaussian and
    the observed space is taken to be Gaussian with
    mean scaled by `outlier_scale` with probability
    `outlier_proba`
    """
    def __init__(
        self, sampling_period, dynamics_covariance, observation_covariance, outlier_proba, outlier_scale
    ):
        super().__init__(sampling_period, dynamics_covariance, observation_covariance)
        self.outlier_proba = outlier_proba
        self.outlier_scale = outlier_scale

    def step(self, z_prev, key):
        key_latent, key_obs, key_corruped = jax.random.split(key, 3)
        z_next = jax.random.multivariate_normal(
            key_latent,
            mean=self.transition_matrix @ z_prev,
            cov=self.dynamics_covariance,
        )

        # Corrupt with outliers
        corrupted = (jax.random.uniform(key_corruped) < self.outlier_proba).astype(jnp.float32)
        mean_next = (
            self.projection_matrix @ z_next * (1 - corrupted) +
            self.projection_matrix @ z_next * corrupted * self.outlier_scale
        )
        cov_next = self.observation_covariance

        x_next = jax.random.multivariate_normal(
            key_obs,
            mean=mean_next,
            cov=cov_next,
        )

        # x_next = corrupted * mean_next * self.outlier_scale + (1 - corrupted) * x_next

        output = {
            "observed": x_next,
            "latent": z_next,
            "is_outlier": corrupted,
        }

        return z_next, output


class GaussOneSideOutlierMovingObject2D(MovingObject2D):
    def __init__(
        self, sampling_period, dynamics_covariance, observation_covariance,
        outlier_proba, outlier_minval, outlier_maxval
    ):
        super().__init__(sampling_period, dynamics_covariance, observation_covariance)
        self.outlier_proba = outlier_proba
        self.outlier_minval = outlier_minval
        self.outlier_maxval = outlier_maxval


    def step(self, z_pred, key):
        key_latent, key_obs, key_corruped, key_value_corrupted = jax.random.split(key, 4)
        z_next = jax.random.multivariate_normal(
            key_latent,
            mean=self.transition_matrix @ z_pred,
            cov=self.dynamics_covariance,
        )

        # Corrupt with outliers
        corrupted = (jax.random.uniform(key_corruped) < self.outlier_proba).astype(jnp.float32)
        mean_next = self.projection_matrix @ z_next
        cov_next = self.observation_covariance

        x_next = jax.random.multivariate_normal(
            key_obs,
            mean=mean_next,
            cov=cov_next,
        )

        corrupted_value = jax.random.uniform(
            key_value_corrupted, minval=self.outlier_minval, maxval=self.outlier_maxval
        )
        x_next = x_next + corrupted * corrupted_value

        output = {
            "observed": x_next,
            "latent": z_next,
            "is_outlier": corrupted,
        }

        return z_next, output


class UCIDatasets:
    """
    https://github.com/yaringal/DropoutUncertaintyExps/tree/master
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.base_url = (
            "https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps"
            "/master/UCI_Datasets/{dataset}/data/data.txt"
        )

    @property
    def datasets(self):
        return [
            "bostonHousing",
            "concrete",
            "energy",
            "kin8nm",
            "naval-propulsion-plant",
            "power-plant",
            "protein-tertiary-structure",
            "wine-quality-red",
            "yacht",
        ]

    def load_dataset(self, dataset):
        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not found")

        file_name = f"{dataset}.csv"
        downloaded_datasets = os.listdir(self.root_dir)
        if any([dataset in d for d in downloaded_datasets]):
            # Dataset already downloaded
            path = os.path.join(self.root_dir, file_name)
            df = pd.read_csv(path, sep=r"\s+", header=None)
        else:
            # Download dataset
            path = self.base_url.format(dataset=dataset)
            df = pd.read_csv(path, sep=r"\s+", header=None)
            df.to_csv(os.path.join(self.root_dir, file_name), index=False, header=None, sep="\t")

        return df

    def sample_one_sided_noisy_dataset(
            self, dataset_name, p_error, v_error=10, prop_warmup=0.1, seed=314
    ):
        dataset = self.load_dataset(dataset_name)
        n_obs, _ = dataset.shape
        n_warmup = int(n_obs * prop_warmup)

        np.random.seed(seed)
        data_norm = dataset.iloc[n_warmup:].sample(frac=1.0, replace=False)
        minv = data_norm.iloc[:n_warmup].min().values
        maxv = data_norm.iloc[:n_warmup].max().values

        normv = maxv - minv
        if np.any(normv == 0):
            mask = np.where(normv == 0)[0]
            normv[mask] = minv[mask]

        data_norm = (data_norm.values - minv) / normv
        n_obs_eval, _ = data_norm.shape

        err_where = np.random.choice(2, size=n_obs_eval, p=[1 - p_error, p_error])

        ix_where = np.where(err_where)
        # err_vals = np.random.uniform(-v_error, v_error, size=len(ix_where[0]))
        err_vals = np.random.choice([-1, 1], size=len(ix_where[0])) * v_error
        data_norm[ix_where, -1] = err_vals

        res = {
            "X": data_norm[:, :-1],
            "y": data_norm[:, -1],
            "maxv": maxv,
            "minv": minv,
            "err_where": err_where,
        }

        return res

    def sample_noisy_covariates(
            self, dataset_name, p_error, v_error=10, prop_warmup=0.1, seed=314
    ):
        dataset = self.load_dataset(dataset_name)
        n_obs, _ = dataset.shape
        n_warmup = int(n_obs * prop_warmup)

        np.random.seed(seed)
        data_norm = dataset.iloc[n_warmup:].sample(frac=1.0, replace=False)
        minv = data_norm.iloc[:n_warmup].min().values
        maxv = data_norm.iloc[:n_warmup].max().values

        data_norm = (data_norm.values - minv) / (maxv - minv)
        n_obs_eval, n_features = data_norm.shape

        X = data_norm[:, :-1]
        y = data_norm[:, -1]
        n_features = X.shape[1]

        err_where = np.random.choice(2, size=(n_obs_eval, n_features), p=[1 - p_error, p_error])
        ix_where = np.where(err_where)
        # err_vals = np.random.choice([-1, 1], size=len(ix_where[0])) * v_error
        err_vals = np.random.uniform(-v_error, v_error, size=len(ix_where[0]))
        X[ix_where] = err_vals

        res = {
            "X": X,
            "y": y,
            "maxv": maxv,
            "minv": minv,
            "err_where": err_where,
        }

        return res


def create_uci_collection(
        dataset_name, noise_type, p_error, n_runs, v_error=50, seed_init=314, path=None
):
    """
    Create a collection of datasets with different noise realizations
    """
    path = path or "./data"
    uci = UCIDatasets(path)
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
