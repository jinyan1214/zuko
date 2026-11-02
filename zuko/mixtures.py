r"""Mixture models."""

__all__ = [
    "GMM",
]

import torch
import torch.nn as nn
import math

from collections.abc import Sequence
from math import prod
from torch import Tensor
from torch.distributions import (
    Distribution,
    MultivariateNormal,
)
from typing import Literal

from .distributions import DiagNormal, Mixture, vMFNDistribution
from .lazy import LazyDistribution
from .nn import MLP
from .utils import unpack
import torch.nn.functional as F


class GMM(LazyDistribution):
    r"""Creates a Gaussian mixture model (GMM).

    .. math:: p(X | c) = \sum_{i = 1}^K w_i(c) \, \mathcal{N}(X | \mu_i(c), \Sigma_i(c))

    Wikipedia:
        https://wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model

    Arguments:
        features: The number of features.
        context: The number of context features.
        components: The number of components :math:`K` in the mixture.
        covariance_type: The type of covariance matrix parameterization to use.
        tied: Whether to tie the covariance parameters across components.
        epsilon: A numerical stability term.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        components: int = 2,
        covariance_type: Literal["full", "diagonal", "spherical"] = "full",
        tied: bool = False,
        epsilon: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__()

        self.components = components
        self.covariance_type = covariance_type
        self.tied = tied
        self.epsilon = epsilon

        shapes = _get_gmm_shapes(components, features, covariance_type, tied)

        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

        if context > 0:
            self.hyper = MLP(context, self.total, **kwargs)
        else:
            self.phi = nn.ParameterList(torch.randn(*s) for s in shapes)

        self.component_responsibilities = None

        self.context = context
        self.context_configs = kwargs

    def forward(self, c: Tensor | None = None) -> Distribution:
        if c is None:
            phi = self.phi
        else:
            phi = self.hyper(c)
            phi = unpack(phi, self.shapes)

        if self.covariance_type == "full":
            return self._forward_full(*phi)
        elif self.covariance_type == "diagonal":
            return self._forward_diagonal(*phi)
        elif self.covariance_type == "spherical":
            return self._forward_diagonal(*phi)
        else:
            raise NotImplementedError(f"Unknown covariance type '{self.covariance_type}'.")

    def _forward_full(
        self, logits: Tensor, loc: Tensor, diag: Tensor, tril: Tensor
    ) -> Distribution:
        scale = torch.diag_embed(diag.exp() + self.epsilon)
        mask = torch.tril(torch.ones_like(scale, dtype=bool), diagonal=-1)
        scale = torch.masked_scatter(scale, mask, tril)

        return Mixture(MultivariateNormal(loc=loc, scale_tril=scale), logits)

    def _forward_diagonal(self, logits: Tensor, loc: Tensor, diag: Tensor) -> Distribution:
        scale = diag.exp() + self.epsilon

        return Mixture(DiagNormal(loc=loc, scale=scale), logits)

    @torch.no_grad()
    def initialize(self, x: Tensor, strategy: Literal["random", "kmeans", "kmeans++", "EM"],
                   W: Tensor | None = None, max_iter: int = 500, tol = 1e-5) -> None:
        r"""Initializes the components of the model.

        Note:
            For more information on the clustering strategies see
            https://scikit-learn.org/dev/modules/mixture.html#choice-of-the-initialization-method

        Arguments:
            x: The feature samples, with shape :math:`(N, D)`.
            strategy: The clustering strategy.
            max_iter: The maximum number of iterations for the EM algorithm.
            tol: The tolerance for convergence of the EM algorithm.
            W: The optional weights for the samples.
        """

        N, _ = x.shape

        assert N > self.components, (
            f"The number of samples ({N}) should be larger than the number of components ({self.components})."
        )
        if strategy == "EM":
            if self.tied:
                raise NotImplementedError("Tied covariance matrices is not supported for EM initialization.")
            if W is None:
                W = torch.ones(N, device=x.device, dtype=x.dtype)
            R = _gmm_initialization(x.T, self.components)
            llh = torch.full(
                    (max_iter,),
                    -torch.inf,
                    dtype=x.dtype,
                    device=x.device
                )
            converged = False
            t = 0
            # ------------------------------------------------------------
            # Soft EM
            # ------------------------------------------------------------
            while (not converged) and (t + 1 < max_iter):

                t += 1

                # --------------------------------------------------------
                # Find non-empty components
                # --------------------------------------------------------
                label = torch.argmax(R, dim=1)
                u = torch.unique(label)

                # Remove empty components
                if R.shape[1] != u.numel():
                    R = R[:, u]

                # --------------------------------------------------------
                # M-step
                # --------------------------------------------------------
                mu, si, pi = _gmm_maximization(
                    x.T,
                    W,
                    R
                )

                # --------------------------------------------------------
                # E-step
                # --------------------------------------------------------
                R, llh_t = _gmm_expectation(
                    x.T,
                    W,
                    mu,
                    si,
                    pi
                )

                llh[t] = llh_t

                # --------------------------------------------------------
                # Check convergence
                # --------------------------------------------------------
                if t > 1:
                    diff = llh[t] - llh[t - 1]

                    eps = torch.abs(diff)

                    converged = bool(
                        eps < tol * torch.abs(llh[t])
                    )

            # ------------------------------------------------------------
            # Output
            # ------------------------------------------------------------
            if not converged:
                print(f"GMM EM Not converged in {max_iter} steps.")
        
            si = si.permute(2, 0, 1)
        
            lower = torch.linalg.cholesky(si)
        
            diag = torch.diagonal(lower, dim1=-2, dim2=-1).log() # Take log to be compatable with forward_full
            tril = lower[(..., *torch.tril_indices(x.shape[1], x.shape[1], offset=-1))]

            mu = mu.T
            params = (pi.log(), mu, diag, tril)

            self.component_responsibilities = R.clone()

            if R.shape[1] < self.components:
                # Some components have no assigned data points, indicating components too large
                empty_components = self.components - R.shape[1]
                print(f"Warning: {empty_components} components have no assigned data points in GMM EM fitting.")
                self.components = R.shape[1]
                shapes = _get_gmm_shapes(R.shape[1], x.shape[1], self.covariance_type, self.tied)
                
                self.shapes = shapes
                self.total = sum(prod(s) for s in shapes)
        
                if hasattr(self, "hyper"):
                    self.hyper = MLP(self.context, self.total, **self.context_configs)
                else:
                    self.phi = nn.ParameterList(torch.randn(*s) for s in shapes)
                
        else:
            if strategy == "random":
                centers = _cluster_random(x, self.components)
            elif strategy == "kmeans":
                centers = _cluster_kmeans(x, self.components)
            elif strategy == "kmeans++":
                centers = _cluster_kmeans_pp(x, self.components)
            else:
                raise NotImplementedError(f"Unknown clustering strategy '{strategy}'.")

            match = torch.cdist(x, centers).argmin(dim=-1)
            match = torch.nn.functional.one_hot(match, num_classes=self.components).to(dtype=x.dtype)

            probs = torch.sum(match, dim=0) / torch.sum(match)
            means = torch.sum(match[:, :, None] * x[:, None, :], dim=0) / torch.sum(
                match[:, :, None], dim=0
            )

            if self.covariance_type == "full":
                covs = _estimate_full_cov(x, match, self.tied)
            elif self.covariance_type == "diagonal":
                covs = _estimate_diagonal_cov(x, match, self.tied)
            elif self.covariance_type == "spherical":
                covs = _estimate_spherical_cov(x, match, self.tied)
            else:
                raise NotImplementedError(f"Unknown covariance type '{self.covariance_type}'.")



            if torch.is_tensor(covs):
                params = (probs.log(), means, covs)
            else:
                params = (probs.log(), means, *covs)

        assert all(p.shape == s for p, s in zip(params, self.shapes, strict=True))

        if hasattr(self, "hyper"):
            self.hyper[-1].weight.mul_(1e-2)
            self.hyper[-1].bias.copy_(torch.cat([p.flatten() for p in params]))
        else:
            for p, p_ in zip(self.phi, params, strict=True):
                p.copy_(p_)


def _get_gmm_shapes(
    components: int,
    features: int,
    covariance_type: Literal["full", "diagonal", "spherical"],
    tied: bool,
) -> Sequence[int]:
    leading = 1 if tied else components

    shapes = [
        (components,),  # probabilities
        (components, features),  # mean
    ]

    if covariance_type == "full":
        shapes.extend([
            (leading, features),  # diagonal
            (leading, features * (features - 1) // 2),  # off diagonal
        ])
    elif covariance_type == "diagonal":
        shapes.extend([
            (leading, features),  # diagonal
        ])
    elif covariance_type == "spherical":
        shapes.extend([
            (leading, 1),  # diagonal
        ])
    else:
        raise NotImplementedError(f"Unknown covariance type '{covariance_type}'.")

    return shapes


def _estimate_full_cov(x: Tensor, match: Tensor, tied: bool) -> tuple[Tensor, Tensor]:
    _, D = x.shape
    _, K = match.shape

    covariances = []

    for k in range(K):
        covariances.append(torch.cov(x.T, aweights=match[:, k]))

    covariances = torch.stack(covariances)

    if tied:
        covariances = covariances.mean(dim=0, keepdim=True)

    lower = torch.linalg.cholesky(covariances)

    diag = torch.diagonal(lower, dim1=-2, dim2=-1)
    tril = lower[(..., *torch.tril_indices(D, D, offset=-1))]

    return diag.log(), tril


def _estimate_diagonal_cov(x: Tensor, match: Tensor, tied: bool) -> Tensor:
    _, D = x.shape
    _, K = match.shape

    diag = []

    for k in range(K):
        for d in range(D):
            diag.append(torch.cov(x[:, d], aweights=match[:, k]))

    diag = torch.stack(diag).reshape(K, D)

    if tied:
        diag = diag.mean(dim=0, keepdim=True)

    return diag.log()


def _estimate_spherical_cov(x: Tensor, match: Tensor, tied: bool) -> Tensor:
    diag = _estimate_diagonal_cov(x, match, tied)
    diag = diag.exp().mean(dim=-1, keepdim=True)

    return diag.log()


def _cluster_random(x: Tensor, components: int) -> Tensor:
    N, _ = x.shape

    idx = torch.multinomial(torch.ones(N, device=x.device), components)
    centers = x[idx]

    return centers


def _cluster_kmeans(x: Tensor, components: int, iterations: int = 7) -> Tensor:
    N, _ = x.shape

    centers = _cluster_kmeans_pp(x, components)

    for _ in range(iterations):
        match = torch.cdist(x, centers).argmin(dim=-1)
        match = torch.nn.functional.one_hot(match, num_classes=components).to(dtype=x.dtype)

        idx = torch.multinomial(torch.ones(N, device=x.device), components)
        centers = torch.where(
            torch.sum(match[:, :, None], dim=0) > 0,
            torch.sum(match[:, :, None] * x[:, None, :], dim=0)
            / torch.sum(match[:, :, None], dim=0),
            x[idx],
        )

    return centers


def _cluster_kmeans_pp(x: Tensor, components: int) -> Tensor:
    N, _ = x.shape

    idx = torch.multinomial(torch.ones(N, device=x.device), components)
    centers = x[idx]

    mask = torch.zeros((N, components), dtype=torch.bool, device=x.device)
    mask[idx, 0] = True

    for k in range(1, components):
        dist = torch.cdist(x, centers[:k])
        dist[mask[:, :k]] = 0
        dist = dist.min(dim=-1).values

        idx = torch.multinomial(dist.square(), 1)
        centers[k] = x[idx]
        mask[idx, k] = True

    return centers

class vMFNM(LazyDistribution):
    r"""Creates a von Mises-Fisher-Nakagami mixture model (vMFNM).

    Source:
        Papaioannou, I., Geyer, S., & Straub, D. (2019).
        Improved cross entropy-based importance sampling with a flexible mixture model.
        Reliability Engineering & System Safety, 191, 106564

    Arguments:
        features: The number of features.
        context: The number of context features.
        components: The number of components :math:`K` in the mixture.
        
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.
    """

    def __init__(
        self,
        features: int,
        components: int,
        context: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.components = components

        shapes = [
            (components,),  # probabilities alpha
            (components, features),  # mean mu
            (components, ),  # concentration kappa
            (components, ),  # shape m
            (components, ),  # spread omega
        ]

        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

        if context > 0:
            self.hyper = MLP(context, self.total, **kwargs)
        else:
            self.phi = nn.ParameterList(torch.randn(*s) for s in shapes)

        self.component_responsibilities = None
        self.context = context
        self.context_configs = kwargs

    def forward(self, c: Tensor | None = None) -> Distribution:
        if c is None:
            phi = self.phi
        else:
            phi = self.hyper(c)
            phi = unpack(phi, self.shapes)

        return Mixture(vMFNDistribution(*(phi[1:])), phi[0])

    @torch.no_grad()
    def initialize(self, x: Tensor, strategy: Literal["EM"], W = None,
                   max_iter: int = 500,
                   tol = 1e-5) -> None:
        r"""Initializes the components of the model with the EM algorithm.

        Note:
            https://github.com/ERA-Software/Overview/blob/5733ff88ba5e791e53177aa1fc41410ec49e3bab/Reliability%20Analysis%20Tools/2.%20Cross%20Entropy%20Method%20and%20Improved%20Cross%20Entropy%20Method/CE_iCE_Python/EMvMFNM.py 

        Arguments:
            x: The feature samples, with shape :math:`(N, D)`.
            strategy: The clustering strategy (Only EM is supported).
        """
        if x.ndim != 2:
            raise ValueError(
                f"x must have shape (d, n), got {tuple(x.shape)}."
            )
        if not x.is_floating_point():
            raise TypeError(
                "x must be a floating-point tensor."
            )
        n, d = x.shape

        if d < 2:
            raise ValueError(
                "The vMF model requires dimension d >= 2."
            )
        
        if torch.any(~torch.isfinite(x)):
            raise ValueError("x contains NaN or infinite values.")
        
        # Initial hard responsibilities: (n, k)
        M = _vMFNM_initialization(
            x.T,
            self.components,
        )
        # Radial component: (n, 1)
        R = torch.linalg.vector_norm(
            x,
            dim=1,
        ).reshape(-1, 1)
        # Direction is undefined for a zero vector, and log(R) is undefined in
        # the Nakagami density.
        if torch.any(R <= 0):
            zero_count = int((R <= 0).sum().item())
            raise ValueError(
                f"X contains {zero_count} zero-norm sample(s). "
                "The vMF direction and Nakagami log-density are undefined "
                "for zero-radius observations."
            )
        # Unit direction vectors: (n, d)
        X_norm = x / R

        previous_llh = None
        converged = False
        if W is None:
            W = torch.ones((n, 1), dtype=x.dtype, device=x.device)
            W = W / W.sum()
        for iteration in range(1, max_iter + 1):
            # Remove components that are not the maximum-responsibility
            # component for any observation, matching the original code.
            label = M.argmax(dim=1)
            nonempty_components = torch.unique(
                label,
                sorted=True,
            )

            if M.shape[1] != nonempty_components.numel():
                M = M[:, nonempty_components]

            mu, kappa, m, omega, alpha = _vmfnm_maximization(
                X_norm.T,
                W,
                R,
                M,
            )

            M, llh = _vmfnm_expectation(
                X_norm.T,
                W,
                R,
                mu,
                kappa,
                m,
                omega,
                alpha,
            )

            if previous_llh is not None:
                con1 = (
                    torch.abs(llh[0] - previous_llh[0])
                    < tol * torch.abs(llh[0])
                )

                con2 = (
                    torch.abs(llh[1] - previous_llh[1])
                    < tol * 100.0 * torch.abs(llh[1])
                )

                if bool(con1 & con2):
                    converged = True
                    break

            previous_llh = llh
        if not converged:
            raise Warning(
                f"EM algorithm did not converge after {max_iter} iterations."
            )

        m = m.reshape(-1, )
        omega = omega.reshape(-1, )
        mu = mu.T
        params = (alpha.log(), mu, kappa, m, omega)

        self.component_responsibilities = M.clone()
        
        if M.shape[1] < self.components:
            # Some components have no assigned data points, indicating components too large
            empty_components = self.components - M.shape[1]
            print(f"Warning: {empty_components} components have no assigned data points in GMM EM fitting.")
            self.components = M.shape[1]
            shapes = [
                        (self.components,),  # probabilities alpha
                        (self.components, x.shape[1]),  # mean mu
                        (self.components, ),  # concentration kappa
                        (self.components, ),  # shape m
                        (self.components, ),  # spread omega
                    ]
            
            self.shapes = shapes
            self.total = sum(prod(s) for s in shapes)
    
            if hasattr(self, "hyper"):
                self.hyper = MLP(self.context, self.total, **self.context_configs)
            else:
                self.phi = nn.ParameterList(torch.randn(*s) for s in shapes)

        assert all(p.shape == s for p, s in zip(params, self.shapes, strict=True))

        if hasattr(self, "hyper"):
            self.hyper[-1].weight.mul_(1e-2)
            self.hyper[-1].bias.copy_(torch.cat([p.flatten() for p in params]))
        else:
            for p, p_ in zip(self.phi, params, strict=True):
                p.copy_(p_)

def _gmm_initialization(X, nGM):
    """
    Parameters
    ----------
    X : torch.Tensor
        Shape: (n_dim, n_samples)
    nGM : int
        Number of mixture components.

    Returns
    -------
    R : torch.Tensor
        One-hot assignment matrix with shape (n_samples, nGM).
    """

    n = X.shape[1]
    device = X.device

    # Random initialization
    idx = torch.randint(0, n, (nGM,), device=device)
    m = X[:, idx]  # (n_dim, nGM)

    # score shape: (nGM, n_samples)
    score = m.T @ X - torch.sum(m * m, dim=0).reshape(-1, 1) / 2

    label = torch.argmax(score, dim=0)
    u = torch.unique(label)

    # Ensure every component has at least one assigned sample
    while len(u) != nGM:
        print("while-loop")

        idx = torch.randint(0, n, (nGM,), device=device)
        m = X[:, idx]

        score = (
            m.T @ X
            - torch.sum(m * m, dim=0).reshape(-1, 1) / 2
        )

        label = torch.argmax(score, dim=0)
        u = torch.unique(label)

    # One-hot responsibility matrix
    R = torch.zeros(
        (n, nGM),
        dtype=torch.int64,
        device=device
    )

    R[torch.arange(n, device=device), label] = 1

    return R
# --------------------------------------------------------------------------
# Log PDF of multivariate Gaussian
# X     : (d, n)
# mu    : (d,)
# Sigma : (d, d)
#
# return: (n,)
# --------------------------------------------------------------------------
def loggausspdf(X, mu, Sigma):
    d = X.shape[0]

    Xo = X - mu[:, None]  # (d, n)

    # Cholesky decomposition:
    # Sigma = L @ L.T
    L = torch.linalg.cholesky(Sigma)

    # Solve L y = Xo
    y = torch.linalg.solve_triangular(
        L,
        Xo,
        upper=False
    )

    mahal = torch.sum(y**2, dim=0)

    # log(det(Sigma)) = 2 * sum(log(diag(L)))
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L)))

    logpdf = -0.5 * (
        d * torch.log(
            torch.tensor(
                2.0 * torch.pi,
                dtype=X.dtype,
                device=X.device
            )
        )
        + logdet
        + mahal
    )

    return logpdf


# --------------------------------------------------------------------------
# Expectation
# --------------------------------------------------------------------------
def _gmm_expectation(X, W, mu, si, pi):
    """
    Parameters
    ----------
    X : (d, n)
    W : (n,) or (n, 1)
    mu : (d, k)
    si : (d, d, k)
    pi : (k,)

    Returns
    -------
    R : (n, k)
        Responsibilities.

    llh : scalar tensor
        Weighted average log-likelihood.
    """

    n = X.shape[1]
    k = mu.shape[1]

    # Make W consistently (n, 1)
    W = W.reshape(-1, 1)

    logpdf = torch.empty(
        (n, k),
        dtype=X.dtype,
        device=X.device
    )

    for i in range(k):
        logpdf[:, i] = loggausspdf(
            X,
            mu[:, i],
            si[:, :, i]
        )

    # log p(x, z=i) = log p(x | z=i) + log pi_i
    logpdf = logpdf + torch.log(pi).reshape(1, -1)

    # T_j = log sum_i exp(logpdf_ji)
    T = torch.logsumexp(
        logpdf,
        dim=1,
        keepdim=True
    )  # (n, 1)

    # Weighted average log likelihood
    llh = torch.sum(W * T) / torch.sum(W)

    # Responsibilities
    logR = logpdf - T
    R = torch.exp(logR)

    return R, llh


# --------------------------------------------------------------------------
# Maximization GMM
# --------------------------------------------------------------------------
def _gmm_maximization(X, W, R):
    """
    Parameters
    ----------
    X : (d, n)
    W : (n,) or (n, 1)
    R : (n, k)

    Returns
    -------
    mu : (d, k)
    Sigma : (d, d, k)
    w : (k,)
    """

    d = X.shape[0]
    k = R.shape[1]

    # Ensure W is (n, 1)
    W = W.reshape(-1, 1)

    # Apply importance/sample weights
    R = W * R  # (n, k)

    # Effective number of samples in each component
    nk = torch.sum(R, dim=0)  # (k,)

    # Prevent division by zero
    # More robust than modifying all nk when only one is zero
    nk_safe = torch.clamp(nk, min=1e-6)

    # Mixture weights
    w = nk_safe / torch.sum(W)

    # Means
    # X: (d,n), R: (n,k)
    # -> (d,k)
    mu = (X @ R) / nk_safe.reshape(1, -1)

    # Covariance matrices
    Sigma = torch.empty(
        (d, d, k),
        dtype=X.dtype,
        device=X.device
    )

    sqrtR = torch.sqrt(R)

    eye = torch.eye(
        d,
        dtype=X.dtype,
        device=X.device
    )

    for i in range(k):
        Xo = X - mu[:, i:i+1]  # (d,n)

        # Weight each sample by sqrt(responsibility)
        Xo_weighted = Xo * sqrtR[:, i].reshape(1, -1)

        Sigma[:, :, i] = (
            Xo_weighted @ Xo_weighted.T
        ) / nk_safe[i]

        # Numerical regularization
        Sigma[:, :, i] += eye * 1e-6

    return mu, Sigma, w

def _vMFNM_initialization(X: Tensor, k: int):
    """
    PyTorch equivalent of the NumPy implementation https://github.com/jinyan1214/Overview/blob/5733ff88ba5e791e53177aa1fc41410ec49e3bab/Reliability%20Analysis%20Tools/2.%20Cross%20Entropy%20Method%20and%20Improved%20Cross%20Entropy%20Method/CE_iCE_Python/EMvMFNM.py.

    Parameters
    ----------
    X : Tensor
        Input tensor with shape (d, n), where d is the feature dimension
        and n is the number of samples.
    k : int
        Number of initial centers/clusters.

    Returns
    -------
    M : Tensor
        One-hot cluster-assignment matrix with shape (n, k).
    """
    if X.ndim != 2:
        raise ValueError(f"X must have shape (d, n), but got {X.shape}")

    n = X.shape[1]
    if k <= 0:
        raise ValueError("k must be positive")

    while True:
        # Matches np.random.choice(range(n), k), which samples with replacement.
        idx = torch.randint(
            low=0,
            high=n,
            size=(k,),
            device=X.device,
        )

        # Selected columns of X: shape (d, k)
        m = X[:, idx]

        # scores[j, i] = m_j^T X_i - ||m_j||^2 / 2
        scores = m.T @ X - (m.square().sum(dim=0) / 2).unsqueeze(1)

        # Cluster assignment for each sample: shape (n,)
        label = scores.argmax(dim=0)

        if torch.unique(label).numel() == k:
            break

    # One-hot assignment matrix: shape (n, k)
    M = torch.nn.functional.one_hot(label, num_classes=k)

    return M.to(dtype=X.dtype)

# def _logsumexp(x: Tensor, dim: int = 0) -> Tensor:
#     """
#     Compatibility wrapper around torch.logsumexp.
#     keepdim=True preserves the reduced dimension, matching how the original
#     function was used inside expectation().
#     """
#     return torch.logsumexp(x, dim=dim, keepdim=True)

def _logbesseli(nu: float, x: Tensor) -> Tensor:
    """
    Approximate log(I_nu(x)), where I_nu is the modified Bessel function
    of the first kind.
    For nu == 0, use PyTorch's exponentially scaled i0e:

        log(I_0(x)) = log(i0e(x)) + |x|

    For nu > 0, retain the approximation from the original implementation.

    Parameters
    ----------
    nu
        Bessel-function order.
    x
        Nonnegative tensor.

    Returns
    -------
    Tensor
        log(I_nu(x)), with the same shape as x.
    """
    if torch.any(x < 0):
        raise ValueError("The Bessel-function argument x must be nonnegative.")

    # Numerically stable exact PyTorch implementation for order zero.
    if abs(nu) < 1e-14:
        return torch.log(torch.special.i0e(x)) + x.abs()

    if nu < 0:
        raise ValueError(
            "The retained Bessel approximation requires nu >= 0."
        )

    nu_tensor = x.new_tensor(nu)

    frac = x / nu_tensor
    square = 1.0 + frac.square()
    root = torch.sqrt(square)

    eta = (
        root
        + torch.log(frac)
        - torch.log1p(root)
    )

    return (
        -0.5
        * torch.log(x.new_tensor(2.0 * math.pi) * nu_tensor)
        + nu_tensor * eta
        - 0.25 * torch.log(square)
    )

def _logvMFpdf(
    X: Tensor,
    mu: Tensor,
    kappa: Tensor | float,
    ) -> Tensor:
    """
    Log-density of the von Mises-Fisher distribution.
    Parameters
    ----------
    X
        Unit-normalized observations with shape (d, n).
    mu
        Unit mean direction with shape (d,) or (d, 1).
    kappa
        Scalar concentration parameter.

    Returns
    -------
    Tensor
        Log-density for each sample, shape (n,).
    """
    if X.ndim != 2:
        raise ValueError(
            f"X must have shape (d, n), got {tuple(X.shape)}."
        )

    d, n = X.shape

    if d < 2:
        raise ValueError("The vMF implementation requires d >= 2.")

    mu = mu.reshape(d).to(device=X.device, dtype=X.dtype)

    kappa = torch.as_tensor(
        kappa,
        device=X.device,
        dtype=X.dtype,
    )

    if kappa.numel() != 1:
        raise ValueError("kappa must be scalar.")

    kappa = kappa.squeeze()

    if bool(kappa < 0):
        raise ValueError(
            "The concentration parameter kappa must be nonnegative."
        )

    # Uniform distribution on the unit sphere S^(d-1).
    if bool(kappa == 0):
        log_normalizer = (
            torch.special.gammaln(X.new_tensor(d / 2.0))
            - math.log(2.0)
            - (d / 2.0) * math.log(math.pi)
        )

        return log_normalizer.expand(n)

    nu = d / 2.0 - 1.0

    log_normalizer = (
        nu * torch.log(kappa)
        - (d / 2.0) * math.log(2.0 * math.pi)
        - _logbesseli(nu, kappa)
    )

    # mu @ X has shape (n,)
    return kappa * (mu @ X) + log_normalizer

def _lognakagamipdf(
    X: Tensor,
    m: Tensor,
    om: Tensor,
    ) -> Tensor:
    """
    Log-density of a Nakagami distribution.
    Broadcasting is supported. In this application:

        X.shape  = (n, 1)
        m.shape  = (1, k)
        om.shape = (1, k)

    so the output has shape (n, k).
    """
    X, m, om = torch.broadcast_tensors(X, m, om)

    if torch.any(X <= 0):
        raise ValueError(
            "Nakagami observations must be strictly positive."
        )

    if torch.any(m <= 0):
        raise ValueError(
            "Nakagami shape parameter m must be positive."
        )

    if torch.any(om <= 0):
        raise ValueError(
            "Nakagami spread parameter omega must be positive."
        )

    return (
        math.log(2.0)
        + m
        * (
            torch.log(m)
            - torch.log(om)
            - X.square() / om
        )
        + (2.0 * m - 1.0) * torch.log(X)
        - torch.special.gammaln(m)
    )

def _dummyvar(idx: Tensor) -> Tensor:
    """
    PyTorch replacement for dummyvar.
    Parameters
    ----------
    idx
        Integer labels with shape (n,), starting at zero.

    Returns
    -------
    Tensor
        One-hot integer tensor with shape
        (n, max(idx) + 1).
    """
    idx = torch.as_tensor(idx, dtype=torch.long)

    if idx.ndim != 1:
        idx = idx.reshape(-1)

    if idx.numel() == 0:
        return torch.empty(
            (0, 0),
            dtype=torch.long,
            device=idx.device,
        )

    if torch.any(idx < 0):
        raise ValueError("Indices in idx must be nonnegative.")

    num_classes = int(idx.max().item()) + 1

    return F.one_hot(
        idx,
        num_classes=num_classes,
    )

def _vmfnm_maximization(
    X: Tensor,
    W: Tensor,
    R: Tensor,
    M: Tensor,
    *,
    eps: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Maximization step of the weighted vMF-Nakagami mixture model.
    Parameters
    ----------
    X
        Unit-normalized data, shape (d, n).
    W
        Sample weights, shape (n, 1).
    R
        Radial magnitudes, shape (n, 1).
    M
        Responsibilities, shape (n, k).

    Returns
    -------
    mu
        Mean directions, shape (d, k).
    kappa
        vMF concentration parameters, shape (k,).
    m
        Nakagami shape parameters, shape (1, k).
    omega
        Nakagami spread parameters, shape (1, k).
    alpha
        Mixture probabilities, shape (k,).
    """
    if eps is None:
        eps = torch.finfo(X.dtype).eps

    # Weighted responsibilities: (n, k)
    weighted_M = W * M

    d = X.shape[0]

    # Effective weighted membership of each component: (k,)
    nk = weighted_M.sum(dim=0)

    if torch.any(nk <= eps):
        raise RuntimeError(
            "At least one mixture component has zero effective weight."
        )

    # Mixture probabilities: (k,)
    alpha = nk / W.sum()

    # Mean directions
    mu_unnormalized = X @ weighted_M       # (d, k)
    norm_mu = torch.linalg.vector_norm(
        mu_unnormalized,
        dim=0,
    )                                     # (k,)

    mu = mu_unnormalized / norm_mu.clamp_min(eps)

    # Approximate vMF concentration parameter
    xi = (norm_mu / nk).clamp(max=0.95)

    kappa = (
        xi * d - xi.pow(3)
    ) / (
        1.0 - xi.square()
    )

    # Nakagami spread parameter
    R2 = R.square()                       # (n, 1)

    omega = (
        weighted_M.T @ R2
    ).T / nk                              # (1, k)

    # Fourth radial moment
    mu4 = (
        weighted_M.T @ R2.square()
    ).T / nk                              # (1, k)

    # Approximate Nakagami shape parameter
    m = omega.square() / (
        mu4 - omega.square()
    )

    fallback_m = torch.full_like(
        m,
        fill_value=d / 2.0,
    )

    invalid_m = (
        ~torch.isfinite(m)
        | (m < 0)
        | (m > 20.0 * d)
    )

    m = torch.where(
        invalid_m,
        fallback_m,
        m,
    )

    return mu, kappa, m, omega, alpha

def _vmfnm_expectation(
    X: Tensor,
    W: Tensor,
    R: Tensor,
    mu: Tensor,
    kappa: Tensor,
    m: Tensor,
    omega: Tensor,
    alpha: Tensor,
    *,
    responsibility_floor: float = 1e-3,
    ) -> tuple[Tensor, Tensor]:
    """
    Expectation step of the weighted vMF-Nakagami mixture model.
    Returns
    -------
    M
        Posterior responsibilities, shape (n, k).
    llh
        Two convergence diagnostics:
            llh[0] = weighted directional log-likelihood
            llh[1] = weighted radial log-likelihood
    """
    n = X.shape[1]
    k = mu.shape[1]

    logvMF = torch.empty(
        (n, k),
        device=X.device,
        dtype=X.dtype,
    )

    # Keep the loop because logvMFpdf accepts one scalar kappa at a time.
    for i in range(k):
        logvMF[:, i] = _logvMFpdf(
            X,
            mu[:, i],
            kappa[i],
        )

    # Broadcasting:
    # R:     (n, 1)
    # m:     (1, k)
    # omega: (1, k)
    # output:(n, k)
    lognakagami = _lognakagamipdf(
        R,
        m,
        omega,
    )

    log_alpha = torch.log(alpha).reshape(1, k)

    logpdf = (
        logvMF
        + lognakagami
        + log_alpha
    )

    # Mixture log-density for each observation: (n, 1)
    log_normalizer = torch.logsumexp(
        logpdf,
        dim=1,
        keepdim=True,
    )

    # Posterior responsibilities
    M_soft = torch.exp(
        logpdf - log_normalizer
    )

    # Preserve the truncation used in the original implementation.
    M = torch.where(
        M_soft < responsibility_floor,
        torch.zeros_like(M_soft),
        M_soft,
    )

    row_sum = M.sum(
        dim=1,
        keepdim=True,
    )

    # Normally at least one responsibility remains after thresholding.
    # Use the unthresholded responsibilities as a safe fallback.
    safe_denominator = row_sum.clamp_min(
        torch.finfo(X.dtype).tiny
    )

    M = torch.where(
        row_sum > 0,
        M / safe_denominator,
        M_soft,
    )

    # Directional and radial likelihood diagnostics
    logvMF_weighted = logvMF + log_alpha
    lognakagami_weighted = lognakagami + log_alpha

    T_vMF = torch.logsumexp(
        logvMF_weighted,
        dim=1,
        keepdim=True,
    )

    T_nakagami = torch.logsumexp(
        lognakagami_weighted,
        dim=1,
        keepdim=True,
    )

    weight_sum = W.sum()

    llh_vMF = (
        W * T_vMF
    ).sum() / weight_sum

    llh_nakagami = (
        W * T_nakagami
    ).sum() / weight_sum

    llh = torch.stack(
        [llh_vMF, llh_nakagami]
    )

    return M, llh