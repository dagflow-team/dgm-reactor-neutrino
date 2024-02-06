from dagflow.lib.OneToOneNode import OneToOneNode
from numba import njit
from numpy import pi
from numpy.typing import NDArray

_pi4 = 0.25 / pi


@njit(cache=True)
def _inv_sq_law(data: NDArray, out: NDArray):
    for i in range(len(out)):
        out[i] = _pi4 / data[i] ** 2


class InverseSquareLaw(OneToOneNode):
    """
    inputs:
        `i`: array of the distances

    outputs:
        `i`: f(L)=1/(4πL²)

    Calcultes an inverse-square law distribution
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "1/(4πL²)")

    def _fcn(self):
        for inp, out in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _inv_sq_law(inp.ravel(), out.ravel())
