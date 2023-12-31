import os
import jax
import numpy as np
import pandas as pd
import jax.numpy as jnp
from functools import partial

class MultiSensorBearingsOnly:
    """
    Example taken from:
    Recursive outlier-robust filtering and smoothing for nonlinear systems using the multivariate student-t distribution
    by Piche, Sarkka, and Hartikainen
    """
    def __init__(self, turning_rate, dt, q1, q2, proba_clutter):
        self.turning_rate = turning_rate
        self.dt = dt
        self.q1 = q1
        self.q2 = q2
        self.proba_clutter = proba_clutter
        self.transition_matrix = self._build_transition_matrix()
        self.process_covariance = self._build_process_noise_covariance()
    
    def _build_transition_matrix(self):
        sint = jnp.sin(self.turning_rate * self.dt)
        cost = jnp.cos(self.turning_rate * self.dt)
        A = jnp.array([
            [1, sint / self.turning_rate, 0, (cost - 1) / self.turning_rate, 0],
            [0, cost, 0, -sint, 0],
            [0, (1 - cost) / self.turning_rate, 1, sint / self.turning_rate, 0],
            [0, sint, 0, cost, 0],
            [0, 0, 0, 0, 1]
        ])
        return A
    
    def _build_process_noise_covariance(self):
        M = jnp.array([
            [self.dt ** 3 / 3, self.dt ** 2 / 2],
            [self.dt ** 2 / 2, self.dt]
        ])

        Q = jax.scipy.linalg.block_diag(
            self.q1 * M, self.q2 * self.dt * M, self.q2
        )

        return Q
    
    @partial(jax.jit, static_argnums=(0,))
    def _step_process(self, key, latent):
        latent_next = jax.random.multivariate_normal(
            key=key, mean=self.transition_matrix @ latent, cov=self.process_covariance
        )
        return latent_next
    
    @partial(jax.jit, static_argnums=(0,))
    def _measurement_step(self, key, latent):
        ...
    

class GaussOutlierMovingObject2D:
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
        self.sampling_period = sampling_period
        self.dynamics_covariance = dynamics_covariance * jnp.eye(4)
        self.observation_covariance = observation_covariance * jnp.eye(2)
        self.transition_matrix, self.projection_matrix = self._init_dynamics(sampling_period)
        self.dim_obs, self.dim_latent = self.projection_matrix.shape
        self.outlier_proba = outlier_proba
        self.outlier_scale = outlier_scale
    
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
        
        # x_next = corrupted * mean_next * self.outlier_scale + (1 - corrupted) * x_next

        output = {
            "observed": x_next,
            "latent": z_next,
        }

        return z_next, output
    
    def sample(self, key, z0, n_steps):
        keys = jax.random.split(key, n_steps)
        _, output = jax.lax.scan(self.step, z0, keys)
        return output


class GaussMeanOutlierMovingObject2D:
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
        self.sampling_period = sampling_period
        self.dynamics_covariance = dynamics_covariance * jnp.eye(4)
        self.observation_covariance = observation_covariance * jnp.eye(2)
        self.transition_matrix, self.projection_matrix = self._init_dynamics(sampling_period)
        self.dim_obs, self.dim_latent = self.projection_matrix.shape
        self.outlier_proba = outlier_proba
        self.outlier_scale = outlier_scale
    
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
        }

        return z_next, output
    
    def sample(self, key, z0, n_steps):
        keys = jax.random.split(key, n_steps)
        _, output = jax.lax.scan(self.step, z0, keys)
        return output


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
            # "wine-quality-red", # classification
            "yacht",
        ]
    
    def load_dataset(self, dataset):
        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not found")

        downloaded_datasets = os.listdir(self.root_dir)
        if any([dataset in d for d in downloaded_datasets]):
            path = os.path.join(self.root_dir, dataset)
            df = pd.read_csv(path, sep=r"\s+", header=None)
        else:
            path = self.base_url.format(dataset=dataset)
            df = pd.read_csv(path, sep=r"\s+", header=None)
            df.to_csv(os.path.join(self.root_dir, dataset), index=False, sep="\t")
            
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

        data_norm = (data_norm.values - minv) / (maxv - minv)
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