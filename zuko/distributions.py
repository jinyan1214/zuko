r"""Parameterizable probability distributions."""

__all__ = [
    "BoxUniform",
    "DiagNormal",
    "GeneralizedNormal",
    "Joint",
    "Maximum",
    "Minimum",
    "Mixture",
    "NormalizingFlow",
    "Sort",
    "TopK",
    "TransformedUniform",
    "Truncated",
    "vMFNMDistribution",
]

import math
import torch

from textwrap import indent
from torch import Size, Tensor
from torch.distributions import *  # noqa: F403
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    Normal,
    Transform,
    Uniform,
    constraints,
)
from torch.distributions.utils import _sum_rightmost

s_vae_path = "/resnick/groups/enceladus/jyzhao/NormalizingFlows/s-vae-pytorch"
import sys
sys.path.append(s_vae_path)
from hyperspherical_vae.distributions import VonMisesFisher

Distribution._validate_args = False
Distribution.arg_constraints = {}


class NormalizingFlow(Distribution):
    r"""Creates a normalizing flow for a random variable :math:`X` towards a base
    distribution :math:`p(Z)` through a transformation :math:`f`.

    The density of a realization :math:`x` is given by the change of variables

    .. math:: p(X = x) = p(Z = f(x)) \left| \det \frac{\partial f(x)}{\partial x} \right| .

    To sample from :math:`p(X)`, realizations :math:`z \sim p(Z)` are mapped through
    the inverse transformation :math:`g = f^{-1}`.

    References:
        | A Family of Non-parametric Density Estimation Algorithms (Tabak et al., 2013)
        | https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.21423

        | Variational Inference with Normalizing Flows (Rezende et al., 2015)
        | https://arxiv.org/abs/1505.05770

        | Normalizing Flows for Probabilistic Modeling and Inference (Papamakarios et al., 2021)
        | https://arxiv.org/abs/1912.02762

    Arguments:
        transform: A transformation :math:`f`.
        base: A base distribution :math:`p(Z)`.

    Example:
        >>> d = NormalizingFlow(ExpTransform(), Gamma(2.0, 1.0))
        >>> d.sample()
        tensor(1.5157)
    """

    has_rsample = True

    def __init__(
        self,
        transform: Transform,
        base: Distribution,
    ) -> None:
        super().__init__()

        reinterpreted = transform.codomain.event_dim - len(base.event_shape)

        if reinterpreted > 0:
            base = Independent(base, reinterpreted)

        self.transform = transform
        self.base = base
        self.reinterpreted = max(-reinterpreted, 0)

    def __repr__(self) -> str:
        lines = [
            f"(transform): {self.transform}",
            f"(base): {self.base}",
        ]
        lines = indent("\n".join(lines), "  ")

        return self.__class__.__name__ + "(\n" + lines + "\n)"

    @property
    def batch_shape(self) -> Size:
        return self.base.batch_shape

    @property
    def event_shape(self) -> Size:
        return self.transform.inverse_shape(self.base.event_shape)

    def expand(self, batch_shape: Size, new: Distribution | None = None) -> Distribution:
        new = self._get_checked_instance(NormalizingFlow, new)
        new.transform = self.transform
        new.base = self.base.expand(batch_shape)
        new.reinterpreted = self.reinterpreted

        Distribution.__init__(new, validate_args=False)

        return new

    def log_prob(self, x: Tensor) -> Tensor:
        z, ladj = self.transform.call_and_ladj(x)
        ladj = _sum_rightmost(ladj, self.reinterpreted)

        return self.base.log_prob(z) + ladj

    def rsample(self, shape: Size = ()) -> Tensor:
        if self.base.has_rsample:
            z = self.base.rsample(shape)
        else:
            z = self.base.sample(shape)

        return self.transform.inv(z)

    def rsample_and_log_prob(self, shape: Size = ()) -> tuple[Tensor, Tensor]:
        if self.base.has_rsample:
            z = self.base.rsample(shape)
        else:
            z = self.base.sample(shape)

        x, ladj = self.transform.inv.call_and_ladj(z)
        ladj = _sum_rightmost(ladj, self.reinterpreted)

        return x, self.base.log_prob(z) - ladj


class Joint(Distribution):
    r"""Creates a distribution for a multivariate random variable :math:`X` which
    is the concatenation of :math:`n` independent random variables :math:`Z_i`.

    .. math:: p(X = x) = \prod_i p(Z_i = x_i)

    Arguments:
        marginals: A list of distributions :math:`p(Z_i)`.

    Example:
        >>> d = Joint(Uniform(0.0, 1.0), Normal(0.0, 1.0))
        >>> d.event_shape
        torch.Size([2])
        >>> d.sample()
        tensor([0.4963, 0.2072])
    """

    has_rsample = True

    def __init__(self, *marginals: Distribution) -> None:
        super().__init__()

        batch_shape = torch.broadcast_shapes(*(m.batch_shape for m in marginals))

        self.marginals = [m.expand(batch_shape) for m in marginals]

    def __repr__(self) -> str:
        lines = map(repr, self.marginals)
        lines = indent("\n".join(lines), "  ")

        return self.__class__.__name__ + "(\n" + lines + "\n)"

    @property
    def batch_shape(self) -> Size:
        return self.marginals[0].batch_shape

    @property
    def event_shape(self) -> Size:
        return Size([sum(m.event_shape.numel() for m in self.marginals)])

    def expand(self, batch_shape: Size, new: Distribution | None = None) -> Distribution:
        new = self._get_checked_instance(Joint, new)
        new.marginals = [m.expand(batch_shape) for m in self.marginals]

        Distribution.__init__(new, validate_args=False)

        return new

    def log_prob(self, x: Tensor) -> Tensor:
        shape = x.shape[:-1]
        i, lp = 0, 0

        for m in self.marginals:
            j = i + m.event_shape.numel()
            z = x[..., i:j].reshape(shape + m.event_shape)
            lp = lp + m.log_prob(z)
            i = j

        return lp

    def rsample(self, shape: Size = ()) -> Tensor:
        x = []

        for m in self.marginals:
            if m.has_rsample:
                z = m.rsample(shape)
            else:
                z = m.sample(shape)

            z = z.reshape(shape + m.batch_shape + (-1,))
            x.append(z)

        return torch.cat(x, dim=-1)


class Mixture(Distribution):
    r"""Creates a mixture of distributions for a random variable :math:`X`.

    .. math:: p(X = x) = \frac{1}{\sum_i w_i} \sum_i w_i \, p(Z_i = x)

    Wikipedia:
        https://wikipedia.org/wiki/Mixture_model

    Arguments:
        base: A batch of base distributions :math:`p(Z_i)`.
        logits: The unnormalized log-weights :math:`\log w_i`.

    Example:
        >>> d = Mixture(Normal(torch.randn(2), torch.ones(2)), torch.randn(2))
        >>> d.event_shape
        torch.Size([])
        >>> d.sample()
        tensor(-1.6920)
    """

    has_rsample = False

    def __init__(self, base: Distribution, logits: Tensor) -> None:
        super().__init__()

        assert base.batch_shape[-1] == logits.shape[-1]
        assert len(base.batch_shape) >= len(logits.shape)

        self.base = base
        self.logits = logits

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base})"

    @property
    def batch_shape(self) -> Size:
        return self.base.batch_shape[:-1]

    @property
    def event_shape(self) -> Size:
        return self.base.event_shape

    @property
    def num_components(self) -> int:
        return len(self.logits)

    def expand(self, batch_shape: Size, new: Distribution | None = None) -> Distribution:
        new = self._get_checked_instance(Mixture, new)
        new.base = self.base.expand(batch_shape + self.base.batch_shape[-1:])
        new.logits = self.logits

        Distribution.__init__(new, validate_args=False)

        return new

    def log_prob(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(dim=-(len(self.event_shape) + 1))

        log_w = torch.log_softmax(self.logits, dim=-1)
        log_p = self.base.log_prob(x)

        return torch.logsumexp(log_w + log_p, dim=-1)

    def sample(self, shape: Size = (),
               return_component_ids: bool = False) -> Tensor:
        with torch.no_grad():
            x = self.base.sample(shape)
            i = Categorical(logits=self.logits).expand(self.batch_shape).sample(shape)

            if i.dim() > 0:
                x = x.flatten(0, i.dim() - 1)
                x = x[torch.arange(i.numel(), device=i.device), i.flatten()]

                if return_component_ids:
                    return x.reshape(shape + self.batch_shape + self.event_shape), i.reshape(shape + self.batch_shape)
                return x.reshape(shape + self.batch_shape + self.event_shape)
            else:
                if return_component_ids:
                    return x[i], i
                return x[i]

    def sample_component(self, component: int, shape: Size = ()) -> Tensor:
        with torch.no_grad():
            x = self.base.sample(shape)
            return x[:, component, :]


class GeneralizedNormal(Distribution):
    r"""Creates a generalized normal distribution.

    .. math:: p(X = x) = \frac{\beta}{2 \Gamma(1 / \beta)} \exp(-|x|^\beta)

    Wikipedia:
        https://wikipedia.org/wiki/Generalized_normal_distribution

    Arguments:
        beta: The shape parameter :math:`\beta`.

    Example:
        >>> d = GeneralizedNormal(2.0)
        >>> d.sample()
        tensor(-0.0281)
    """

    arg_constraints = {"beta": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, beta: Tensor) -> None:
        super().__init__()

        self.beta = torch.as_tensor(beta)

    @property
    def batch_shape(self) -> Size:
        return self.beta.shape

    def expand(self, batch_shape: Size, new: Distribution | None = None) -> Distribution:
        new = self._get_checked_instance(GeneralizedNormal, new)
        new.beta = self.beta.expand(batch_shape)

        Distribution.__init__(new, validate_args=False)

        return new

    def log_prob(self, x: Tensor) -> Tensor:
        return torch.log(self.beta / 2) - torch.lgamma(1 / self.beta) - abs(x) ** self.beta

    def rsample(self, shape: Size = ()) -> Tensor:
        beta = self.beta.expand(shape + self.beta.shape)
        x = torch._standard_gamma(1 / beta) ** (1 / beta)
        x = x * torch.sign(2 * torch.rand_like(x) - 1)
        return x


class DiagNormal(Independent):
    r"""Creates a multivariate normal distribution parametrized by the variables
    mean :math:`\mu` and standard deviation :math:`\sigma`, but assumes no
    correlation between the variables.

    Arguments:
        loc: The mean :math:`\mu` of the variables.
        scale: The standard deviation :math:`\sigma` of the variables.
        ndims: The number of batch dimensions to interpret as event dimensions.

    Example:
        >>> d = DiagNormal(torch.zeros(3), torch.ones(3))
        >>> d.event_shape
        torch.Size([3])
        >>> d.sample()
        tensor([ 1.5410, -0.2934, -2.1788])
    """

    def __init__(self, loc: Tensor, scale: Tensor, ndims: int = 1) -> None:
        super().__init__(Normal(torch.as_tensor(loc), torch.as_tensor(scale)), ndims)

    def __repr__(self) -> str:
        return "Diag" + repr(self.base_dist)

    def expand(self, batch_shape: Size, new: Distribution | None = None) -> Distribution:
        new = self._get_checked_instance(DiagNormal, new)
        return super().expand(batch_shape, new)


class BoxUniform(Independent):
    r"""Creates a distribution for a multivariate random variable :math:`X`
    distributed uniformly over an hypercube domain. Formally,

    .. math:: l_i \leq X_i < u_i ,

    where :math:`l_i` and :math:`u_i` are respectively the lower and upper bounds
    of the domain in the :math:`i`-th dimension.

    Arguments:
        lower: The lower bounds (inclusive).
        upper: The upper bounds (exclusive).
        ndims: The number of batch dimensions to interpret as event dimensions.

    Example:
        >>> d = BoxUniform(-torch.ones(3), torch.ones(3))
        >>> d.event_shape
        torch.Size([3])
        >>> d.sample()
        tensor([-0.0075,  0.5364, -0.8230])
    """

    def __init__(self, lower: Tensor, upper: Tensor, ndims: int = 1) -> None:
        super().__init__(Uniform(torch.as_tensor(lower), torch.as_tensor(upper)), ndims)

    def __repr__(self) -> str:
        return "Box" + repr(self.base_dist)

    def expand(self, batch_shape: Size, new: Distribution | None = None) -> Distribution:
        new = self._get_checked_instance(BoxUniform, new)
        return super().expand(batch_shape, new)


class TransformedUniform(NormalizingFlow):
    r"""Creates a distribution for a random variable :math:`X`, whose
    transformation :math:`f(X)` is uniformly distributed over the interval
    :math:`[f(l), f(u)]`.

    .. math:: p(X = x) = \frac{1}{f(u) - f(l)}
        \begin{cases}
            f'(x) & \text{if } f(l) \leq f(x) < f(u) \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        f: A transformation :math:`f`, monotonically increasing over :math:`[l, u]`.
        lower: A lower bound :math:`l` (inclusive).
        upper: An upper bound :math:`u` (exclusive).

    Example:
        >>> d = TransformedUniform(ExpTransform(), -1.0, 1.0)
        >>> d.sample()
        tensor(0.4281)
    """

    def __init__(self, f: Transform, lower: Tensor, upper: Tensor) -> None:
        super().__init__(f, Uniform(*map(f, map(torch.as_tensor, (lower, upper)))))

    def expand(self, batch_shape: Size, new: Distribution | None = None) -> Distribution:
        new = self._get_checked_instance(TransformedUniform, new)
        return super().expand(batch_shape, new)


class Truncated(Distribution):
    r"""Truncates a base distribution :math:`p(X)` between a lower bound
    :math:`l` and an upper bound :math:`u`.

    .. math:: p(X = x | l \leq X < u) = \frac{1}{P(X \leq u) - P(X \leq l)}
        \begin{cases}
            p(X = x) & \text{if } l \leq x < u \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        base: A base distribution :math:`p(X)`.
        lower: A lower bound :math:`l` (inclusive).
        upper: An upper bound :math:`u` (exclusive).

    Example:
        >>> d = Truncated(Normal(0.0, 1.0), 1.0, 2.0)
        >>> d.sample()
        tensor(1.3333)
    """

    has_rsample = True

    def __init__(
        self,
        base: Distribution,
        lower: Tensor = float("-inf"),
        upper: Tensor = float("+inf"),
    ) -> None:
        super().__init__()

        assert not base.event_shape, "'base' has to be univariate"

        self.base = base
        self.uniform = Uniform(base.cdf(lower), base.cdf(upper))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base})"

    @property
    def batch_shape(self) -> Size:
        return self.base.batch_shape

    def expand(self, batch_shape: Size, new: Distribution | None = None) -> Distribution:
        new = self._get_checked_instance(Truncated, new)
        new.base = self.base.expand(batch_shape)
        new.uniform = self.uniform.expand(batch_shape)

        Distribution.__init__(new, validate_args=False)

        return new

    def cdf(self, x: Tensor) -> Tensor:
        return self.uniform.cdf(self.base.cdf(x))

    def log_prob(self, x: Tensor) -> Tensor:
        return self.uniform.log_prob(self.base.cdf(x)) + self.base.log_prob(x)

    def rsample(self, shape: Size = ()) -> Tensor:
        return self.base.icdf((1 - 2e-6) * self.uniform.rsample(shape) + 1e-6)


class Sort(Distribution):
    r"""Creates a distribution for a :math:`n`-d random variable :math:`X`, whose elements
    :math:`X_i` are :math:`n` draws from a base distribution :math:`p(Z)`, ordered
    such that :math:`X_i \leq X_{i + 1}`.

    .. math:: p(X = x) = \begin{cases}
            n! \, \prod_{i = 1}^n p(Z = x_i) & \text{if $x$ is ordered} \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        base: A base distribution :math:`p(Z)`.
        n: The number of draws :math:`n`.
        descending: Whether the elements are sorted in descending order or not.

    Example:
        >>> d = Sort(Normal(0.0, 1.0), 3)
        >>> d.event_shape
        torch.Size([3])
        >>> d.sample()
        tensor([-2.1788, -0.2934,  1.5410])
    """

    def __init__(
        self,
        base: Distribution,
        n: int = 2,
        descending: bool = False,
    ) -> None:
        super().__init__()

        assert not base.event_shape, "'base' has to be univariate"

        self.base = base
        self.n = n
        self.descending = descending
        self.log_fact = math.log(math.factorial(n))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base}, {self.n})"

    @property
    def batch_shape(self) -> Size:
        return self.base.batch_shape

    @property
    def event_shape(self) -> Size:
        return Size([self.n])

    def expand(self, batch_shape: Size, new: Distribution | None = None) -> Distribution:
        new = self._get_checked_instance(Sort, new)
        new.base = self.base.expand(batch_shape)
        new.n = self.n
        new.descending = self.descending
        new.log_fact = self.log_fact

        Distribution.__init__(new, validate_args=False)

        return new

    def log_prob(self, x: Tensor) -> Tensor:
        x = torch.movedim(x, -1, 0)

        if self.descending:
            ordered = x[:-1] >= x[1:]
        else:
            ordered = x[:-1] <= x[1:]

        ordered = ordered.all(dim=0)

        return ordered.log() + self.log_fact + self.base.log_prob(x).sum(dim=0)

    def sample(self, shape: Size = ()) -> Tensor:
        x = torch.movedim(self.base.sample((self.n, *shape)), 0, -1)
        x = torch.sort(x, dim=-1, descending=self.descending).values

        return x


class TopK(Sort):
    r"""Creates a distribution for a :math:`k`-d random variable :math:`X`, whose elements
    :math:`X_i` are the top :math:`k` among :math:`n` draws from a base distribution
    :math:`p(Z)`, ordered such that :math:`X_i \leq X_{i + 1}`.

    .. math:: p(X = x) = \begin{cases}
            \frac{n!}{(n - k)!} \, \prod_{i = 1}^k p(Z = x_i)
                \, P(Z \geq x_k)^{n - k} & \text{if $x$ is ordered} \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        base: A base distribution :math:`p(Z)`.
        k: The number of selected elements :math:`k`.
        n: The number of draws :math:`n`.
        kwargs: Keyword arguments passed to :class:`Sort`.

    Example:
        >>> d = TopK(Normal(0.0, 1.0), 2, 3)
        >>> d.event_shape
        torch.Size([2])
        >>> d.sample()
        tensor([-2.1788, -0.2934])
    """

    def __init__(
        self,
        base: Distribution,
        k: int = 1,
        n: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(base, n, **kwargs)

        assert 1 <= k < n, "k has to be in [1, n)"

        self.k = k
        self.log_fact = self.log_fact - math.log(math.factorial(n - k))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base}, {self.k}, {self.n})"

    @property
    def event_shape(self) -> Size:
        return Size([self.k])

    def expand(self, batch_shape: Size, new: Distribution | None = None) -> Distribution:
        new = self._get_checked_instance(TopK, new)
        new.k = self.k
        return super().expand(batch_shape, new)

    def log_prob(self, x: Tensor) -> Tensor:
        cdf = self.base.cdf(x[..., -1])

        if not self.descending:
            cdf = 1 - cdf

        return (self.n - self.k) * cdf.log() + super().log_prob(x)

    def sample(self, shape: Size = ()) -> Tensor:
        return super().sample(shape)[..., : self.k]


class Minimum(TopK):
    r"""Creates a distribution for a random variable :math:`X`, which is the
    minimum among :math:`n` draws from a base distribution :math:`p(Z)`.

    .. math:: p(X = x) = n \, p(Z = x) \, P(Z \geq x)^{n - 1}

    Arguments:
        base: A base distribution :math:`p(Z)`.
        n: The number of draws :math:`n`.

    Example:
        >>> d = Minimum(Normal(0.0, 1.0), 3)
        >>> d.event_shape
        torch.Size([])
        >>> d.sample()
        tensor(-2.1788)
    """

    def __init__(self, base: Distribution, n: int = 2) -> None:
        super().__init__(base, 1, n)

        self.descending = False

    def __repr__(self) -> str:
        return Sort.__repr__(self)

    @property
    def event_shape(self) -> Size:
        return Size([])

    def expand(self, batch_shape: Size, new: Distribution | None = None) -> Distribution:
        new = self._get_checked_instance(Minimum, new)
        return super().expand(batch_shape, new)

    def log_prob(self, x: Tensor) -> Tensor:
        return super().log_prob(x.unsqueeze(dim=-1))

    def sample(self, shape: Size = ()) -> Tensor:
        return super().sample(shape).squeeze(dim=-1)


class Maximum(Minimum):
    r"""Creates a distribution for a random variable :math:`X`, which is the
    maximum among :math:`n` draws from a base distribution :math:`p(Z)`.

    .. math:: p(X = x) = n \, p(Z = x) \, P(Z \leq x)^{n - 1}

    Arguments:
        base: A base distribution :math:`p(Z)`.
        n: The number of draws :math:`n`.

    Example:
        >>> d = Maximum(Normal(0.0, 1.0), 3)
        >>> d.event_shape
        torch.Size([])
        >>> d.sample()
        tensor(1.5410)
    """

    def __init__(self, base: Distribution, n: int = 2) -> None:
        super().__init__(base, n)

        self.descending = True

    def expand(self, batch_shape: Size, new: Distribution | None = None) -> Distribution:
        new = self._get_checked_instance(Maximum, new)
        return super().expand(batch_shape, new)


# ============================================================
# Density helper functions
# ============================================================

def _log_besseli(nu: float, x: Tensor) -> Tensor:
    """
    Compute/approximate log(I_nu(x)).

    Exact stable formulas are used for nu=0 and nu=1/2.
    For other orders, the approximation from the original code is used.
    """
    if torch.any(x <= 0):
        raise ValueError("x must be strictly positive.")

    # I_0(x)
    if abs(nu) < 1e-12:
        return torch.log(torch.special.i0e(x)) + x

    # I_{1/2}(x) = sqrt(2 / (pi*x)) sinh(x)
    if abs(nu - 0.5) < 1e-12:
        log_sinh = (
            x
            + torch.log(-torch.expm1(-2.0 * x))
            - math.log(2.0)
        )

        return (
            0.5
            * (
                math.log(2.0)
                - math.log(math.pi)
                - torch.log(x)
            )
            + log_sinh
        )

    if nu < 0:
        raise ValueError("nu must be nonnegative.")

    # Abramowitz and Stegun 9.7.7 approximation
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
        * torch.log(
            x.new_tensor(2.0 * math.pi) * nu_tensor
        )
        + nu_tensor * eta
        - 0.25 * torch.log(square)
    )


def _vmf_log_normalizer(
    kappa: Tensor,
    dimension: int,
) -> Tensor:
    """
    Log vMF normalizing constant:

        log C_d(kappa)

    where

        C_d(kappa)
        = kappa^(d/2 - 1)
          / ((2*pi)^(d/2) I_(d/2 - 1)(kappa)).
    """
    d = dimension
    nu = d / 2.0 - 1.0

    # Uniform density on S^(d-1)
    uniform_log_normalizer = (
        torch.special.gammaln(
            kappa.new_tensor(d / 2.0)
        )
        - math.log(2.0)
        - (d / 2.0) * math.log(math.pi)
    )

    safe_kappa = kappa.clamp_min(
        torch.finfo(kappa.dtype).tiny
    )

    positive_log_normalizer = (
        nu * torch.log(safe_kappa)
        - (d / 2.0) * math.log(2.0 * math.pi)
        - _log_besseli(nu, safe_kappa)
    )

    return torch.where(
        kappa > 0,
        positive_log_normalizer,
        uniform_log_normalizer,
    )


def _nakagami_log_prob(
    radius: Tensor,
    m: Tensor,
    omega: Tensor,
) -> Tensor:
    """
    Nakagami log-density:

        p(r) =
            2 m^m / (Gamma(m) omega^m)
            * r^(2m-1)
            * exp(-m r^2 / omega)
    """
    return (
        math.log(2.0)
        + m
        * (
            torch.log(m)
            - torch.log(omega)
            - radius.square() / omega
        )
        + (2.0 * m - 1.0) * torch.log(radius)
        - torch.special.gammaln(m)
    )


# ============================================================================
# Batched sampling helpers
# ============================================================================

def _sample_uniform_sphere(
    shape: torch.Size,
    dimension: int,
    reference: Tensor,
) -> Tensor:
    """
    Sample uniformly from S^(dimension-1).

    Returns
    -------
    Tensor
        Shape: shape + (dimension,)
    """
    z = torch.randn(
        shape + (dimension,),
        dtype=reference.dtype,
        device=reference.device,
    )
    z[z==0] = torch.finfo(z.dtype).tiny

    return z / torch.linalg.vector_norm(
        z,
        dim=-1,
        keepdim=True,
    )


def _expand_parameter_for_samples(
    parameter: Tensor,
    sample_shape: torch.Size,
) -> Tensor:
    """
    Expand a batched parameter over sample dimensions.

    If:
        parameter.shape == batch_shape

    then the output has shape:
        sample_shape + batch_shape
    """
    leading_ones = (1,) * len(sample_shape)

    return parameter.reshape(
        leading_ones + parameter.shape
    ).expand(
        sample_shape + parameter.shape
    )


def _sample_vmf_base_3d_batched(
    kappa: Tensor,
    sample_shape: torch.Size,
    reference: Tensor,
) -> Tensor:
    """
    Reparameterized vMF samples in three dimensions centered at e_3.

    Parameters
    ----------
    kappa
        Shape: batch_shape
    sample_shape
        Requested sample shape.
    reference
        Tensor providing device and dtype.

    Returns
    -------
    Tensor
        Shape: sample_shape + batch_shape + (3,)
    """
    full_shape = sample_shape + kappa.shape

    kappa_expanded = _expand_parameter_for_samples(
        kappa,
        sample_shape,
    )

    eps = torch.finfo(reference.dtype).eps

    uniform = torch.rand(
        full_shape,
        dtype=reference.dtype,
        device=reference.device,
    ).clamp(eps, 1.0 - eps)

    # Uniform axial coordinate at kappa=0.
    a = 2.0 * uniform - 1.0

    small_kappa = kappa_expanded.abs() < 1e-4

    safe_kappa = torch.where(
        small_kappa,
        torch.ones_like(kappa_expanded),
        kappa_expanded,
    )

    # Exact inverse CDF in d=3:
    #
    # w = log((1-u)e^(-kappa) + u e^(kappa)) / kappa
    log_mixture = torch.logaddexp(
        torch.log1p(-uniform) - safe_kappa,
        torch.log(uniform) + safe_kappa,
    )

    w_exact = log_mixture / safe_kappa

    # Taylor expansion around kappa=0.
    w_series = (
        a
        + 0.5
        * kappa_expanded
        * (1.0 - a.square())
        + kappa_expanded.square()
        * (a.pow(3) - a)
        / 3.0
    )

    w = torch.where(
        small_kappa,
        w_series,
        w_exact,
    ).clamp(-1.0, 1.0)

    phi = (
        2.0
        * math.pi
        * torch.rand(
            full_shape,
            dtype=reference.dtype,
            device=reference.device,
        )
    )

    transverse_radius = torch.sqrt(
        torch.clamp_min(
            1.0 - w.square(),
            0.0,
        )
    )

    return torch.stack(
        (
            transverse_radius * torch.cos(phi),
            transverse_radius * torch.sin(phi),
            w,
        ),
        dim=-1,
    )


@torch.no_grad()
def _sample_vmf_base_rejection_batched(
    kappa: Tensor,
    sample_shape: torch.Size,
    dimension: int,
    reference: Tensor,
    *,
    max_rejection_rounds: int = 10_000,
) -> Tensor:
    """
    Batched exact rejection sampler for vMF distributions centered at e_d.

    Parameters
    ----------
    kappa
        Shape: batch_shape
    sample_shape
        Requested sample shape.
    dimension
        Event dimension.
    reference
        Tensor providing device and dtype.

    Returns
    -------
    Tensor
        Shape: sample_shape + batch_shape + (dimension,)
    """
    full_shape = sample_shape + kappa.shape

    kappa_flat = _expand_parameter_for_samples(
        kappa,
        sample_shape,
    ).reshape(-1)

    number_of_samples = kappa_flat.numel()
    d_minus_1 = float(dimension - 1)

    t1 = torch.sqrt(
        4.0 * kappa_flat.square()
        + d_minus_1**2
    )

    # Stable equivalent of:
    #
    # b = (-2*kappa + t1) / (d-1)
    b = d_minus_1 / (
        2.0 * kappa_flat + t1
    )

    x0 = (1.0 - b) / (1.0 + b)

    c = (
        kappa_flat * x0
        + d_minus_1
        * torch.log1p(-x0.square())
    )

    beta_shape = reference.new_tensor(
        d_minus_1 / 2.0
    )

    beta_distribution = Beta(
        beta_shape,
        beta_shape,
    )

    w = reference.new_empty(number_of_samples)

    remaining = torch.arange(
        number_of_samples,
        device=reference.device,
    )

    tiny = torch.finfo(reference.dtype).tiny

    for _ in range(max_rejection_rounds):
        if remaining.numel() == 0:
            break

        count = remaining.numel()

        z = beta_distribution.sample((count,))

        uniform = torch.rand(
            count,
            dtype=reference.dtype,
            device=reference.device,
        ).clamp_min(tiny)

        current_kappa = kappa_flat[remaining]
        current_b = b[remaining]
        current_x0 = x0[remaining]
        current_c = c[remaining]

        candidate_w = (
            1.0 - (1.0 + current_b) * z
        ) / (
            1.0 - (1.0 - current_b) * z
        )

        test_statistic = (
            current_kappa * candidate_w
            + d_minus_1
            * torch.log1p(
                -current_x0 * candidate_w
            )
            - current_c
        )

        accepted = (
            test_statistic >= torch.log(uniform)
        )

        accepted_indices = remaining[accepted]

        w[accepted_indices] = candidate_w[accepted]

        remaining = remaining[~accepted]

    if remaining.numel() > 0:
        raise RuntimeError(
            "The vMF rejection sampler exceeded "
            f"{max_rejection_rounds} rounds."
        )

    transverse_direction = _sample_uniform_sphere(
        torch.Size([number_of_samples]),
        dimension - 1,
        reference,
    )

    transverse_radius = torch.sqrt(
        torch.clamp_min(
            1.0 - w.square(),
            0.0,
        )
    )

    base_samples = torch.cat(
        (
            transverse_radius.unsqueeze(-1)
            * transverse_direction,
            w.unsqueeze(-1),
        ),
        dim=-1,
    )

    return base_samples.reshape(
        full_shape + (dimension,)
    )


def _rotate_from_last_axis_batched(
    base_samples: Tensor,
    mu: Tensor,
) -> Tensor:
    """
    Rotate samples centered at e_d to batched mean directions mu.

    Parameters
    ----------
    base_samples
        Shape:
            sample_shape + batch_shape + (d,)
    mu
        Shape:
            batch_shape + (d,)

    Returns
    -------
    Tensor
        Shape:
            sample_shape + batch_shape + (d,)
    """
    dimension = mu.shape[-1]

    e_d = mu.new_zeros(dimension)
    e_d[-1] = 1.0

    tiny = torch.finfo(mu.dtype).tiny

    # Chart 1:
    #
    # Q = 2aa^T - I
    # a = (e_d + mu) / ||e_d + mu||
    #
    # This is stable except near mu = -e_d.
    a = e_d + mu

    a = a / torch.linalg.vector_norm(
        a,
        dim=-1,
        keepdim=True,
    ).clamp_min(tiny)

    rotated_chart_1 = (
        2.0
        * (
            base_samples * a
        ).sum(
            dim=-1,
            keepdim=True,
        )
        * a
        - base_samples
    )

    # Chart 2:
    #
    # Q = I - 2vv^T / (v^Tv)
    # v = e_d - mu
    #
    # This is stable near mu = -e_d.
    v = e_d - mu

    denominator = v.square().sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(tiny)

    rotated_chart_2 = (
        base_samples
        - 2.0
        * (
            base_samples * v
        ).sum(
            dim=-1,
            keepdim=True,
        )
        * v
        / denominator
    )

    # The branch is used only to select a numerically stable
    # Householder chart.
    use_chart_1 = (
        mu[..., -1:] > -0.5
    ).detach()

    return torch.where(
        use_chart_1,
        rotated_chart_1,
        rotated_chart_2,
    )


# ============================================================================
# Batched vMF-Nakagami distribution
# ============================================================================

class vMFNDistribution(Distribution):
    """
    Batched von Mises-Fisher-Nakagami distribution.

    Generative model
    ----------------
        U ~ vMF(mu, kappa)
        R ~ Nakagami(m, omega)
        X = R U

    Parameter shapes
    ----------------
    mu
        batch_shape + (d,)

    kappa
        Any shape broadcastable to batch_shape.

    m
        Any shape broadcastable to batch_shape.

    omega
        Any shape broadcastable to batch_shape.

    Distribution shapes
    -------------------
    batch_shape
        Broadcasted shape of mu[..., 0], kappa, m, and omega.

    event_shape
        (d,)

    Reparameterization
    ------------------
    dimension == 3:
        rsample() is differentiable with respect to:
            mu, kappa, m, omega

    dimension != 3:
        rsample() is differentiable with respect to:
            mu, m, omega

        but only when kappa.requires_grad is False.
    """

    arg_constraints = {
        "mu": constraints.independent(
            constraints.real,
            1,
        ),
        "kappa": constraints.nonnegative,
        "m": constraints.positive,
        "omega": constraints.positive,
    }

    support = constraints.independent(
        constraints.real,
        1,
    )

    has_rsample = False

    def __init__(
        self,
        mu: Tensor,
        kappa: Tensor | float,
        m: Tensor | float,
        omega: Tensor | float,
        validate_args: bool | None = None,
    ):
        mu = torch.as_tensor(mu)

        if not mu.is_floating_point():
            mu = mu.float()

        if mu.ndim < 1:
            raise ValueError(
                "mu must have shape (..., d)."
            )

        dimension = mu.shape[-1]

        if dimension < 2:
            raise ValueError(
                "The event dimension d must be at least 2."
            )

        def convert_parameter(
            value: Tensor | float,
        ) -> Tensor:
            return torch.as_tensor(
                value,
                dtype=mu.dtype,
                device=mu.device,
            )

        kappa = convert_parameter(kappa)
        m = convert_parameter(m)
        omega = convert_parameter(omega)

        # The final dimension of mu is the event dimension.
        mu_batch_shape = mu.shape[:-1]

        batch_shape = torch.broadcast_shapes(
            mu_batch_shape,
            kappa.shape,
            m.shape,
            omega.shape,
        )

        # self.batch_shape = batch_shape
        # self.event_shape = torch.Size([dimension])

        # expand() does not copy data.
        mu = mu.expand(
            batch_shape + (dimension,)
        )

        kappa = kappa.expand(batch_shape)
        m = m.expand(batch_shape)
        omega = omega.expand(batch_shape)

        mu_norm = torch.linalg.vector_norm(
            mu,
            dim=-1,
            keepdim=True,
        )

        if bool(torch.any(mu_norm <= 0)):
            raise ValueError(
                "Every mean-direction vector in mu must be nonzero."
            )

        # Normalize every vector independently.
        mu = mu / mu_norm

        if bool(torch.any(kappa < 0)):
            raise ValueError(
                "Every kappa value must be nonnegative."
            )

        if bool(torch.any(m <= 0)):
            raise ValueError(
                "Every m value must be positive."
            )

        if bool(torch.any(omega <= 0)):
            raise ValueError(
                "Every omega value must be positive."
            )

        self.mu = mu
        self.kappa = kappa
        self.m = m
        self.omega = omega

        self.dimension = dimension

        # In d=3 the inverse-CDF directional sampler is fully
        # reparameterized. In other dimensions, kappa must be fixed.
        self.has_rsample = (
            dimension == 3
            or not self.kappa.requires_grad
        )

        super().__init__(
            batch_shape=torch.Size(batch_shape),
            event_shape=torch.Size([dimension]),
            validate_args=validate_args,
        )

    def expand(
        self,
        batch_shape: torch.Size,
        _instance=None,
    ) -> "vMFNDistribution":
        """
        Expand the distribution to a new batch shape without copying
        parameter storage.
        """
        batch_shape = torch.Size(batch_shape)

        new = self._get_checked_instance(
            vMFNDistribution,
            _instance,
        )

        new.mu = self.mu.expand(
            batch_shape + (self.dimension,)
        )

        new.kappa = self.kappa.expand(batch_shape)
        new.m = self.m.expand(batch_shape)
        new.omega = self.omega.expand(batch_shape)

        new.dimension = self.dimension

        new.has_rsample = (
            new.dimension == 3
            or not new.kappa.requires_grad
        )

        super(vMFNDistribution, new).__init__(
            batch_shape=batch_shape,
            event_shape=self.event_shape,
            validate_args=False,
        )

        new._validate_args = self._validate_args

        return new

    def vmf_log_prob(
        self,
        unit_vector: Tensor,
    ) -> Tensor:
        """
        vMF log-density with respect to surface measure.

        Parameters
        ----------
        unit_vector
            Shape:
                sample_shape + batch_shape + (d,)

        Returns
        -------
        Tensor
            Shape:
                sample_shape + batch_shape
        """
        return (
            _vmf_log_normalizer(
                self.kappa,
                self.dimension,
            )
            + self.kappa
            * (
                unit_vector * self.mu
            ).sum(dim=-1)
        )

    def nakagami_log_prob(
        self,
        radius: Tensor,
    ) -> Tensor:
        """
        Nakagami radial log-density.

        Parameters
        ----------
        radius
            Shape:
                sample_shape + batch_shape
        """
        return _nakagami_log_prob(
            radius,
            self.m,
            self.omega,
        )

    def log_prob_polar(
        self,
        value: Tensor,
    ) -> Tensor:
        """
        Log-density with respect to:

            dr d(surface measure)

        This does not include the Cartesian Jacobian.
        """
        value = torch.as_tensor(
            value,
            dtype=self.mu.dtype,
            device=self.mu.device,
        )

        if self._validate_args:
            self._validate_sample(value)

        radius = torch.linalg.vector_norm(
            value,
            dim=-1,
        )

        safe_radius = radius.clamp_min(
            torch.finfo(value.dtype).tiny
        )

        unit_vector = (
            value / safe_radius.unsqueeze(-1)
        )

        result = (
            self.vmf_log_prob(unit_vector)
            + self.nakagami_log_prob(safe_radius)
        )

        return torch.where(
            radius > 0,
            result,
            torch.full_like(result, -torch.inf),
        )

    def log_prob(
        self,
        value: Tensor,
    ) -> Tensor:
        """
        Cartesian log-density with respect to dx.

        Because:

            dx = r^(d-1) dr d(surface),

        the Cartesian density includes:

            -(d-1) log(r).
        """
        value = torch.as_tensor(
            value,
            dtype=self.mu.dtype,
            device=self.mu.device,
        )

        radius = torch.linalg.vector_norm(
            value,
            dim=-1,
        )

        safe_radius = radius.clamp_min(
            torch.finfo(value.dtype).tiny
        )

        result = (
            self.log_prob_polar(value)
            - (self.dimension - 1)
            * torch.log(safe_radius)
        )

        return torch.where(
            radius > 0,
            result,
            torch.full_like(result, -torch.inf),
        )

    def _sample_direction(
        self,
        sample_shape: torch.Size,
        *,
        reparameterized: bool,
    ) -> Tensor:
        if self.dimension == 3:
            if reparameterized:
                base_samples = (
                    _sample_vmf_base_3d_batched(
                        self.kappa,
                        sample_shape,
                        self.mu,
                    )
                )
            else:
                with torch.no_grad():
                    base_samples = (
                        _sample_vmf_base_3d_batched(
                            self.kappa.detach(),
                            sample_shape,
                            self.mu,
                        )
                    )
        else:
            if (
                reparameterized
                and self.kappa.requires_grad
            ):
                raise NotImplementedError(
                    "For dimension != 3, rsample() does not "
                    "provide an ordinary pathwise derivative "
                    "with respect to kappa. Detach kappa or use "
                    "a specialized rejection-sampling gradient "
                    "estimator."
                )

            base_samples = (
                _sample_vmf_base_rejection_batched(
                    self.kappa.detach(),
                    sample_shape,
                    self.dimension,
                    self.mu,
                )
            )

        rotation_mu = (
            self.mu
            if reparameterized
            else self.mu.detach()
        )

        return _rotate_from_last_axis_batched(
            base_samples,
            rotation_mu,
        )

    def sample(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> Tensor:
        """
        Non-reparameterized sampling.

        Returns
        -------
        Tensor
            Shape:
                sample_shape + batch_shape + event_shape
        """
        sample_shape = torch.Size(sample_shape)

        with torch.no_grad():
            # If R ~ Nakagami(m, omega), then:
            #
            # R² ~ Gamma(
            #     concentration=m,
            #     rate=m/omega
            # )
            radius_squared = Gamma(
                concentration=self.m,
                rate=self.m / self.omega,
            ).sample(sample_shape)

            radius = torch.sqrt(radius_squared)

            direction = self._sample_direction(
                sample_shape,
                reparameterized=False,
            )

        return radius.unsqueeze(-1) * direction

    def rsample(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> Tensor:
        """
        Reparameterized sampling when supported.

        For d=3:
            gradients propagate through mu, kappa, m, and omega.

        For d!=3 with fixed kappa:
            gradients propagate through mu, m, and omega.
        """
        sample_shape = torch.Size(sample_shape)

        if not self.has_rsample:
            raise NotImplementedError(
                "A learnable kappa is currently supported by "
                "rsample() only when the event dimension is 3. "
                "For other dimensions, use kappa.detach()."
            )

        radius_squared = Gamma(
            concentration=self.m,
            rate=self.m / self.omega,
        ).rsample(sample_shape)

        radius = torch.sqrt(radius_squared)

        direction = self._sample_direction(
            sample_shape,
            reparameterized=True,
        )

        return radius.unsqueeze(-1) * direction

    def standard_normal_log_weight(
        self,
        value: Tensor,
    ) -> Tensor:
        """
        Importance-sampling log weight:

            log N(value; 0, I) - log q(value)
        """
        value = torch.as_tensor(
            value,
            dtype=self.mu.dtype,
            device=self.mu.device,
        )

        standard_normal_log_prob = (
            -0.5 * value.square().sum(dim=-1)
            - 0.5
            * self.dimension
            * math.log(2.0 * math.pi)
        )

        return (
            standard_normal_log_prob
            - self.log_prob(value)
        )

    def likelihood_ratio_log(
        self,
        value: Tensor,
    ) -> Tensor:
        return self.standard_normal_log_weight(value)