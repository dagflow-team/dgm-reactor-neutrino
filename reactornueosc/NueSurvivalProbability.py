from dagflow.nodes import FunctionNode
from dagflow.typefunctions import (
    check_input_shape,
    check_input_subtype,
    copy_from_input_to_output,
)
from numba import float64, njit, void
from numpy import float_, integer, pi, sin, sqrt
from numpy.typing import NDArray
from scipy.constants import value

_oscprobArgConversion = (
    pi * 2e-3 * value("electron volt-inverse meter relationship")
)


@njit(
    void(
        float64[:],
        float64[:],
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
    ),
    cache=True,
)
def _osc_prob(
    out: NDArray[float_],
    E: NDArray[float_],
    L: float,
    sinSq2Theta12: float,
    sinSq2Theta13: float,
    DeltaMSq21: float,
    DeltaMSq32: float,
    alpha: float,
    oscprobArgConversion: float,
) -> None:
    _DeltaMSq32 = alpha * DeltaMSq32  # Δm²₃₂ = α*|Δm²₃₂|
    _DeltaMSq31 = alpha * DeltaMSq32 + DeltaMSq21  # Δm²₃₁ = α*|Δm²₃₂| + Δm²₂₁
    _sinSqTheta12 = 0.5 * (1 - sqrt(1 - sinSq2Theta12))  # sin²θ₁₂
    _cosSqTheta12 = 1.0 - _sinSqTheta12  # cos²θ₁₂
    _cosQuTheta13 = (0.5 * (1 - sqrt(1 - sinSq2Theta13))) ** 2  # cos⁴θ₁₃

    for i in range(len(out)):
        L4E = oscprobArgConversion * L / 4.0 / E[i]  # common factor
        out[i] = (
            1
            - sinSq2Theta13
            * (
                _sinSqTheta12 * sin(_DeltaMSq32 * L4E) ** 2
                + _cosSqTheta12 * sin(_DeltaMSq31 * L4E) ** 2
            )
            - sinSq2Theta12 * _cosQuTheta13 * sin(DeltaMSq21 * L4E) ** 2
        )


class NueSurvivalProbability(FunctionNode):
    """
    inputs:
        `E`: array of the energies
        `L`: the distance
        `sinSq2Theta12`: sin²2θ₁₂
        `sinSq2Theta13`: sin²2θ₁₃
        `DeltaMSq21`: Δm²₂₁ = |Δm²₂₁|
        `DeltaMSq32`: |Δm²₃₂|
        `alpha`: α - the mass ordering constant

    optional inputs:
        `oscprobArgConversion`: Convert Δm²[eV²]L[km]/E[MeV] to natural units.
        If the input is not given a default value will be used:
        `2*pi*1e-3*scipy.value('electron volt-inverse meter relationship')`

    outputs:
        `0` or `result`: array of probabilities

    Calcultes a survival probability for the neutrino
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "P(ee)")
        self.add_input(
            (
                "L",
                "sinSq2Theta12",
                "sinSq2Theta13",
                "DeltaMSq21",
                "DeltaMSq32",
                "alpha",
            ),
            positional=False,
        )
        self._add_output("result")

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_input_shape(
            self,
            (
                "L",
                "sinSq2Theta12",
                "sinSq2Theta13",
                "DeltaMSq21",
                "DeltaMSq32",
                "alpha",
            ),
            (1,),
        )
        check_input_subtype(self, "alpha", integer)
        copy_from_input_to_output(self, "E", "result")

    def _fcn(self, _, inputs, outputs):
        out = outputs["result"].data.ravel()
        E = inputs["E"].data.ravel()
        L = inputs["L"].data[0]
        sinSq2Theta12 = inputs["sinSq2Theta12"].data[0]
        sinSq2Theta13 = inputs["sinSq2Theta13"].data[0]
        DeltaMSq21 = inputs["DeltaMSq21"].data[0]
        DeltaMSq32 = inputs["DeltaMSq32"].data[0]
        alpha = inputs["alpha"].data[0]

        if (conversionInput := inputs.get("oscprobArgConversion")) is not None:
            oscprobArgConversion = conversionInput.data[0]
        else:
            oscprobArgConversion = _oscprobArgConversion

        _osc_prob(
            out,
            E,
            L,
            sinSq2Theta12,
            sinSq2Theta13,
            DeltaMSq21,
            DeltaMSq32,
            alpha,
            oscprobArgConversion,
        )
        return out
