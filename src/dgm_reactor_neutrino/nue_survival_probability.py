from __future__ import annotations

from typing import TYPE_CHECKING

from dag_modelling.core.node import Node
from dag_modelling.core.storage import NodeStorage
from dag_modelling.core.type_functions import (
    assign_axes_from_inputs_to_outputs,
    check_shape_of_inputs,
    copy_from_inputs_to_outputs,
)
from numba import njit
from numpy import array, pi, sin, sqrt
from scipy.constants import value

if TYPE_CHECKING:
    from typing import Literal

    from dag_modelling.core.node import Node
    from dag_modelling.core.output import Output
    from nested_mapping.typing import KeyLike
    from numpy import double
    from numpy.typing import NDArray

_surprobArgConversion = pi * 2e-3 * value("electron volt-inverse meter relationship")


@njit(cache=True)
def _sur_prob(
    out: NDArray[double],
    E: NDArray[double],
    L: NDArray[double],
    L_scale: float,
    SinSq2Theta12: NDArray[double],
    SinSq2Theta13: NDArray[double],
    DeltaMSq21: NDArray[double],
    DeltaMSq32: NDArray[double],
    nmo: NDArray[double],
    surprobArgConversion: NDArray[double],
) -> None:
    _DeltaMSq21 = DeltaMSq21[0]
    _DeltaMSq32 = nmo[0] * DeltaMSq32[0]  # Δm²₃₂ = α*|Δm²₃₂|
    _DeltaMSq31 = nmo[0] * DeltaMSq32[0] + _DeltaMSq21  # Δm²₃₁ = α*|Δm²₃₂| + Δm²₂₁
    _SinSq2Theta13 = SinSq2Theta13[0]
    _SinSq2Theta12 = SinSq2Theta12[0]
    _SinSqTheta12 = 0.5 * (1 - sqrt(1 - _SinSq2Theta12))  # sin²θ₁₂
    _CosSqTheta12 = 1.0 - _SinSqTheta12  # cos²θ₁₂
    _CosSqTheta13 = 1 - 0.5 * (1 - sqrt(1 - SinSq2Theta13[0]))  # cos²θ₁₃
    _CosQuTheta13 = _CosSqTheta13 * _CosSqTheta13  # cos⁴θ₁₃

    sinCommonArg = surprobArgConversion[0] * L[0] * 0.25 * L_scale
    for i in range(len(out)):
        L4E = sinCommonArg / E[i]  # common factor
        Sin32 = sin(_DeltaMSq32 * L4E)
        Sin31 = sin(_DeltaMSq31 * L4E)
        Sin21 = sin(_DeltaMSq21 * L4E)
        out[i] = (
            1.0
            - _SinSq2Theta13
            * (_SinSqTheta12 * Sin32 * Sin32 + _CosSqTheta12 * Sin31 * Sin31)
            - _SinSq2Theta12 * _CosQuTheta13 * Sin21 * Sin21
        )


class NueSurvivalProbability(Node):
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
        `surprobArgConversion`: Convert Δm²[eV²]L[km]/E[MeV] to natural units.
        If the input is not given a default value will be used:
        `2*pi*1e-3*scipy.value('electron volt-inverse meter relationship')`

    outputs:
        `0` or `result`: array of probabilities

    Calcultes a survival probability for the neutrino
    """

    __slots__ = (
        "_baseline_scale",
        "_result",
        "_E",
        "_L",
        "_SinSq2Theta12",
        "_SinSq2Theta13",
        "_DeltaMSq21",
        "_DeltaMSq32",
        "_nmo",
        "_surprob_arg_conversion_factor",
    )

    _baseline_scale: float
    _E: NDArray
    _L: NDArray
    _nmo: NDArray
    _DeltaMSq21: NDArray
    _DeltaMSq32: NDArray
    _SinSq2Theta12: NDArray
    _SinSq2Theta13: NDArray
    _surprob_arg_conversion_factor: NDArray
    _result: NDArray

    def __init__(self, *args, distance_unit: Literal["km", "m"] = "km", **kwargs):
        super().__init__(
            *args,
            **kwargs,
            allowed_kw_inputs=(
                "E",
                "L",
                "SinSq2Theta13",
                "SinSq2Theta12",
                "DeltaMSq32",
                "DeltaMSq21",
                "nmo",
                "surprobArgConversion",
            ),
        )
        self._labels.setdefault("mark", "P(ee)")
        self._add_pair("E", "result")
        self._add_input("L", positional=False)
        self._add_input("SinSq2Theta12", positional=False)
        self._add_input("SinSq2Theta13", positional=False)
        self._add_input("DeltaMSq21", positional=False)
        self._add_input("DeltaMSq32", positional=False)
        self._add_input("nmo", positional=False)
        try:
            self._baseline_scale = {"km": 1, "m": 1.0e-3}[distance_unit]
        except KeyError as e:
            raise RuntimeError(f"Invalid distance unit {distance_unit}") from e

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_shape_of_inputs(
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
        copy_from_inputs_to_outputs(self, "E", "result")
        assign_axes_from_inputs_to_outputs(
            self, "E", "result", assign_meshes=True, overwrite_assigned=True
        )

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        _sur_prob(
            self._result,
            self._E,
            self._L,
            self._baseline_scale,
            self._SinSq2Theta12,
            self._SinSq2Theta13,
            self._DeltaMSq21,
            self._DeltaMSq32,
            self._nmo,
            self._surprob_arg_conversion_factor,
        )

    def _post_allocate(self):
        super()._post_allocate()

        self._result = self.outputs["result"]._data.ravel()

        self._E = self.inputs["E"]._data.ravel()
        self._L = self.inputs["L"]._data
        self._SinSq2Theta12 = self.inputs["SinSq2Theta12"]._data
        self._SinSq2Theta13 = self.inputs["SinSq2Theta13"]._data
        self._DeltaMSq21 = self.inputs["DeltaMSq21"]._data
        self._DeltaMSq32 = self.inputs["DeltaMSq32"]._data
        self._nmo = self.inputs["nmo"]._data

        if conversion_input := self.inputs.get("surprobArgConversion"):
            self._surprob_arg_conversion_factor = conversion_input._data
        else:
            self._surprob_arg_conversion_factor = array([_surprobArgConversion])

    @classmethod
    def replicate(
        cls,
        *args,
        name: str,
        replicate_outputs: tuple[KeyLike, ...] = ((),),
        surprobArgConversion: Output | Literal[True] | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        storage = NodeStorage()
        nodes = storage.create_child("nodes")
        inputs = storage.create_child("inputs")
        outputs = storage.create_child("outputs")

        nametuple = tuple(name.split("."))
        for key in replicate_outputs:
            ckey = nametuple + (key,) if isinstance(key, str) else nametuple + key
            cname = ".".join(ckey)
            surprob = cls(cname, *args, **kwargs)
            nodes[ckey] = surprob
            inputs[nametuple + ("enu",) + key] = surprob.inputs[0]
            inputs[nametuple + ("L",) + key] = surprob.inputs["L"]
            outputs[ckey] = surprob.outputs[0]

            if surprobArgConversion:
                if surprobArgConversion == True:
                    inputs[nametuple + ("surprobArgConversion",) + key] = surprob(
                        "surprobArgConversion"
                    )
                else:
                    surprobArgConversion >> surprob("surprobArgConversion")

        NodeStorage.update_current(storage, strict=True, verbose=verbose)

        return None, storage
