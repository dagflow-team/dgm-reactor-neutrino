from typing import Tuple, Optional, Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from dagflow.node import Node
from multikeydict.typing import KeyLike

from dagflow.nodes import FunctionNode
from dagflow.typefunctions import (
    assign_output_axes_from_inputs,
    check_input_shape,
    check_input_subtype,
    copy_from_input_to_output,
)
from dagflow.storage import NodeStorage

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
    SinSq2Theta12: float,
    SinSq2Theta13: float,
    DeltaMSq21: float,
    DeltaMSq32: float,
    nmo: float,
    oscprobArgConversion: float,
) -> None:
    _DeltaMSq32 = nmo * DeltaMSq32  # Δm²₃₂ = α*|Δm²₃₂|
    _DeltaMSq31 = nmo * DeltaMSq32 + DeltaMSq21  # Δm²₃₁ = α*|Δm²₃₂| + Δm²₂₁
    _SinSqTheta12 = 0.5 * (1 - sqrt(1 - SinSq2Theta12))  # sin²θ₁₂
    _CosSqTheta12 = 1.0 - _SinSqTheta12  # cos²θ₁₂
    _CosQuTheta13 = (0.5 * (1 - sqrt(1 - SinSq2Theta13))) ** 2  # cos⁴θ₁₃

    sinCommonArg = oscprobArgConversion * L / 4.0
    for i in range(len(out)):
        L4E = sinCommonArg / E[i]  # common factor
        out[i] = (
            1
            - SinSq2Theta13
            * (
                _SinSqTheta12 * sin(_DeltaMSq32 * L4E) ** 2
                + _CosSqTheta12 * sin(_DeltaMSq31 * L4E) ** 2
            )
            - SinSq2Theta12 * _CosQuTheta13 * sin(DeltaMSq21 * L4E) ** 2
        )


class NueSurvivalProbability(FunctionNode):
    """
    inputs:
        `E`: array of the energies
        `L`: the distance
        `SinSq2Theta12`: sin²2θ₁₂
        `SinSq2Theta13`: sin²2θ₁₃
        `DeltaMSq21`: Δm²₂₁
        `DeltaMSq32`: |Δm²₃₂|
        `nmo`: α - the mass ordering constant

    optional inputs:
        `oscprobArgConversion`: Convert Δm²[eV²]L[km]/E[MeV] to natural units.
        If the input is not given a default value will be used:
        `2*pi*1e-3*scipy.value('electron volt-inverse meter relationship')`

    outputs:
        `0` or `result`: array of probabilities

    Calcultes a survival probability for the neutrino
    """

    __slots__ = ('_baseline_scale',)
    _baseline_scale: float

    def __init__(self, *args, distance_unit: Literal['km', 'm']='km', **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "P(ee)")
        self.add_input("E", positional=True)
        self.add_input(
            (
                "L",
                "SinSq2Theta12",
                "SinSq2Theta13",
                "DeltaMSq21",
                "DeltaMSq32",
                "nmo",
            ),
            positional=False,
        )
        self._add_output("result")

        try:
            self._baseline_scale = {'km': 1, 'm': 1.e-3}[distance_unit]
        except KeyError as e:
            raise RuntimeError(f"Invalid distance unit {distance_unit}") from e

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_input_shape(
            self,
            (
                "L",
                "SinSq2Theta12",
                "SinSq2Theta13",
                "DeltaMSq21",
                "DeltaMSq32",
                "nmo",
            ),
            (1,),
        )
        # check_input_subtype(self, "nmo", integer)
        copy_from_input_to_output(self, "E", "result")
        assign_output_axes_from_inputs(self, "E", "result", assign_meshes=True, ignore_assigned=True)

    def _fcn(self, _, inputs, outputs):
        out = outputs["result"].data.ravel()
        E = inputs["E"].data.ravel()
        L = inputs["L"].data[0]
        SinSq2Theta12 = inputs["SinSq2Theta12"].data[0]
        SinSq2Theta13 = inputs["SinSq2Theta13"].data[0]
        DeltaMSq21 = inputs["DeltaMSq21"].data[0]
        DeltaMSq32 = inputs["DeltaMSq32"].data[0]
        nmo = inputs["nmo"].data[0]

        if (conversionInput := inputs.get("oscprobArgConversion")) is not None:
            oscprobArgConversion = conversionInput.data[0]
        else:
            oscprobArgConversion = _oscprobArgConversion

        _osc_prob(
            out,
            E,
            L*self._baseline_scale,
            SinSq2Theta12,
            SinSq2Theta13,
            DeltaMSq21,
            DeltaMSq32,
            nmo,
            oscprobArgConversion,
        )
        return out

    @classmethod
    def replicate(
        cls,
        name: str,
        *args,
        replicate: Tuple[KeyLike,...]=((),),
        **kwargs
    ) -> Tuple[Optional["Node"], NodeStorage]:
        storage = NodeStorage()
        nodes = storage.child('nodes')
        inputs = storage.child('inputs')
        outputs = storage.child('outputs')

        name: tuple = tuple(name.split('.'))
        for key in replicate:
            ckey = name + key
            cname = ".".join(ckey)
            oscprob = cls(cname, *args, **kwargs)
            nodes[ckey] = oscprob
            inputs[name + ('enu',) + key] = oscprob.inputs[0]
            inputs[name + ('L',) + key] = oscprob.inputs['L']
            outputs[ckey] = oscprob.outputs[0]

        NodeStorage.update_current(storage, strict=True)

        return None, storage
