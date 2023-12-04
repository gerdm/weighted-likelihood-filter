import jax
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
