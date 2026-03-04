"""Compatibility package for moved model modules.

Canonical implementations now live under `quant.models`.
"""

from core.bergomi import RoughBergomiModel
from core.stochastic_process import JAXFractionalBrownianMotion

__all__ = [
	"RoughBergomiModel",
	"JAXFractionalBrownianMotion",
]
