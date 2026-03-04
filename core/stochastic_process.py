"""Compatibility shim: canonical process moved to quant.models.stochastic_process."""

from quant.models.stochastic_process import JAXFractionalBrownianMotion

__all__ = ["JAXFractionalBrownianMotion"]