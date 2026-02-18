import jax
import jax.numpy as jnp
import numpy as np
from core.stochastic_process import JAXFractionalBrownianMotion
from functools import partial

class RoughBergomiModel:
    """
    Rough Bergomi (rBergomi) stochastic volatility model.

    Dynamics  (Bayer, Friz & Gatheral 2016):
        V_t  = xi_0 * exp(eta * Ŵ^H_t  -  0.5 * eta² * Var[Ŵ^H_t])
        dS_t / S_t  = mu dt  +  sqrt(V_t) * (rho dW_t + sqrt(1-rho²) dW⊥_t)

    Ŵ^H is a *Riemann-Liouville* fBM built from the same driving BM  W
    that enters the spot:

        Ŵ^H(t) = sqrt(2H) * ∫₀ᵗ (t-s)^{H-0.5} dW(s)

    Two fBM simulation methods are available (configurable via
    bergomi.fBm_method in config/params.yaml):

    - "volterra" (default, recommended): Discretized Volterra kernel.
      Produces RL-fBM which is the theoretically correct process for
      rBergomi. Exact spot-vol correlation by construction.
      Reference: Bayer, Friz & Gatheral (2016), Bennedsen et al. (2017).

    - "davies_harte": Circulant embedding / FFT method. Produces standard
      fBM (Mandelbrot-Van Ness). Faster (O(N log N) vs O(N²)) but has a
      different covariance structure for H ≠ 0.5, and does not share the
      same driving BM as the spot (no exact spot-vol correlation).
      Reference: Dietrich & Newsam (1997), Dieker (2004).

    The spot Euler scheme uses the **previsible** variance V_{k-1}
    to avoid the adaptedness bias.
    """

    def __init__(self, params: dict):
        self.h = params['hurst']
        self.eta = params['eta']
        self.rho = params['rho']
        self.xi0 = params['xi0']
        self.n_steps = params['n_steps']
        self.T = params['T']
        self.mu = params.get('mu', 0.05)
        self.dt = self.T / self.n_steps
        self.fBm_method = params.get('fBm_method', 'volterra')

        self._A, self._var_wh = self._build_volterra_kernel()
        self.fbm_gen = JAXFractionalBrownianMotion(self.n_steps, self.T, self.h)

    # ------------------------------------------------------------------
    #  Volterra kernel
    # ------------------------------------------------------------------
    def _build_volterra_kernel(self):
        """
        Build the lower-triangular kernel matrix  A  such that
            Ŵ^H(t_j)  =  Σ_{k=0}^{j}  A[j,k] · Z_k

        A[j,k] = C · [(j-k+1)^{H+½} - (j-k)^{H+½}]
        with C = sqrt(2H) · dt^H / (H + ½).

        Returns  (A, var_wh)  where var_wh[j] = Σ_k A[j,k]².
        """
        n, H, dt = self.n_steps, self.h, self.dt
        C = np.sqrt(2 * H) * dt ** H / (H + 0.5)

        A = np.zeros((n, n))
        for j in range(n):
            for k in range(j + 1):
                lag = j - k
                A[j, k] = C * ((lag + 1) ** (H + 0.5) - lag ** (H + 0.5))

        var_wh = np.array([np.sum(A[j, :] ** 2) for j in range(n)])
        return jnp.array(A), jnp.array(var_wh)

    # ------------------------------------------------------------------
    #  Variance-only
    # ------------------------------------------------------------------
    def simulate_variance_paths(self, n_paths: int, key=None):
        """
        Variance paths using the configured fBm method.

        - 'volterra': Uses Volterra RL-fBM (same as spot-vol sim).
          Consistent with simulate_spot_vol_paths.
        - 'davies_harte': Uses standard fBM via circulant embedding.
          Faster but different covariance structure for H ≠ 0.5.
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        if self.fBm_method == 'davies_harte':
            wh = self.fbm_gen.generate_paths(key, n_paths)
            return self._compute_variance_dh(wh)
        else:
            Z = jax.random.normal(key, (n_paths, self.n_steps))
            Wh = Z @ self._A.T
            return self.xi0 * jnp.exp(
                self.eta * Wh - 0.5 * self.eta ** 2 * self._var_wh
            )

    # ------------------------------------------------------------------
    #  Joint Spot + Vol  (Volterra kernel, correct correlation)
    # ------------------------------------------------------------------
    def simulate_spot_vol_paths(self, n_paths: int, s0=100.0, mu=None, key=None):
        """
        Joint simulation of (S_t, V_t) with *exact* spot-vol correlation.

        Steps
        -----
        1.  Z ~ N(0, I_n)
        2.  dW = sqrt(dt) · Z           (driving standard BM)
        3.  Ŵ^H = A · Z                (Volterra fBM, shares the same Z)
        4.  V_t = xi0 · exp(eta·Ŵ^H - ½ eta² · var_wh)
        5.  dW_spot = rho·dW + sqrt(1-rho²)·dW⊥
        6.  Log-Euler with **previsible** V_{k-1}
        """
        if mu is None:
            mu = self.mu
        if key is None:
            key = jax.random.PRNGKey(42)
        key_z, key_perp = jax.random.split(key)

        n = self.n_steps

        # 1–3. Joint sample
        Z = jax.random.normal(key_z, (n_paths, n))
        dW = Z * jnp.sqrt(self.dt)                 # (n_paths, n)
        Wh = Z @ self._A.T                          # (n_paths, n)

        # 4. Variance process
        vt = self.xi0 * jnp.exp(self.eta * Wh
                                - 0.5 * self.eta ** 2 * self._var_wh)

        # 5. Correlated spot noise
        Z_perp = jax.random.normal(key_perp, (n_paths, n))
        dW_perp = Z_perp * jnp.sqrt(self.dt)
        dW_spot = self.rho * dW + jnp.sqrt(1 - self.rho ** 2) * dW_perp

        # 6. Log-Euler with *previsible* variance  V_{k-1}
        #    (V_k depends on Z_k → not F_{k-1}-measurable)
        v0_col = jnp.full((n_paths, 1), self.xi0)
        vt_prev = jnp.concatenate([v0_col, vt[:, :-1]], axis=1)  # (n_paths, n)

        log_ret = (mu - 0.5 * vt_prev) * self.dt + jnp.sqrt(vt_prev) * dW_spot
        log_s = jnp.cumsum(log_ret, axis=1)
        st = s0 * jnp.exp(log_s)

        # Prepend S0
        s0_col = jnp.full((n_paths, 1), s0)
        st = jnp.concatenate([s0_col, st], axis=1)

        return st, vt

    # ------------------------------------------------------------------
    #  Variance helpers
    # ------------------------------------------------------------------
    def _compute_variance_dh(self, wh):
        """Variance from Davies-Harte fBM (exact t^{2H} drift)."""
        tg = jnp.linspace(self.dt, self.T, self.n_steps)
        drift = 0.5 * self.eta ** 2 * tg ** (2 * self.h)
        return self.xi0 * jnp.exp(self.eta * wh - drift)
