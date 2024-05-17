from typing import Literal

from numba import njit
from numpy import multiply, pi
from numpy.typing import NDArray

from dagflow.lib.OneToOneNode import OneToOneNode

_pi4 = 0.25 / pi


@njit(cache=True)
def _inv_sq_law(data: NDArray, out: NDArray):
    for i in range(len(out)):
        out[i] = _pi4 / data[i] ** 2


_scales = {
    "km_to_cm": 1e-10,
    "m_to_cm": 1e-4,
    None: None
}


class InverseSquareLaw(OneToOneNode):
    """
    inputs:
        `i`: array of the distances

    outputs:
        `i`: f(L)=1/(4πL²)

    Calcultes an inverse-square law distribution
    """

    __slots__ = ("_scale",)
    _scale: float | None

    def __init__(
        self,
        *args,
        scale: Literal["km_to_cm", "m_to_cm", None] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "1/(4πL²)")
        self._scale = _scales[scale]

    def _fcn(self):
        for inp, out in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _inv_sq_law(inp.ravel(), out.ravel())

            if (scale:=self._scale) is not None:
                multiply(out, scale, out=out)
