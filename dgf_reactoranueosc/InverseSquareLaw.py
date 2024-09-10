from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
from numpy import multiply, pi

from dagflow.lib.OneToOneNode import OneToOneNode

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray

_forth_over_pi = 0.25 / pi


@njit(cache=True)
def _inv_sq_law(data: NDArray, out: NDArray):
    for i in range(len(out)):
        L = data[i]
        out[i] = _forth_over_pi / (L * L)


_scales = {"km_to_cm": 1e-10, "m_to_cm": 1e-4, None: 1.0}


class InverseSquareLaw(OneToOneNode):
    """
    inputs:
        `i`: array of the distances

    outputs:
        `i`: f(L)=1/(4πL²)

    Calcultes an inverse-square law distribution
    """

    __slots__ = ("_scale",)
    _scale: float

    def __init__(
        self, *args, scale: Literal["km_to_cm", "m_to_cm", None] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "1/(4πL²)")
        self._scale = _scales[scale]

        self._functions.update({"normal": self._fcn_normal, "scaled": self._fcn_scaled})
        if scale is None or self._scale == 1.0:
            self.fcn = self._fcn_normal
        else:
            self.fcn = self._fcn_scaled

    def _fcn_normal(self):
        for inp, out in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _inv_sq_law(inp.ravel(), out.ravel())

    def _fcn_scaled(self):
        scale = self._scale
        for inp, out in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _inv_sq_law(inp.ravel(), out.ravel())
            multiply(out, scale, out=out)
