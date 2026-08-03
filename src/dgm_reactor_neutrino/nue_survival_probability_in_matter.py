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
_aem_g = value('atomic mass constant') * 1000
_Gf = value('Fermi coupling constant')
_hbar = value('reduced Planck constant')
_c = value('speed of light in vacuum')
_e = value('elementary charge')
_hbar_c_eV_cm = (_hbar * _c / _e) * 100
_SqGeV_to_SqMeV = 1e-18
_GF_CONVERSION = _SqGeV_to_SqMeV * (_hbar_c_eV_cm ** 3)
_MeV_to_eV = 1e-6

@njit(cache=True)
def _sur_prob(
        out: NDArray[double],
        E: NDArray[double],
        L: NDArray[double],
        L_scale: float,
        SinSq2Theta12: NDArray[double],
        SinSq2Theta13: NDArray[double],
        DeltaMSq21: NDArray[double],
        DeltaMSq3lAbs: NDArray[double],
        is_dm32_leading: float,
        nmo: NDArray[double],
        ElectronToNucleonFraction: NDArray[double],
        density: NDArray[double],
        surprobArgConversion: NDArray[double],
) -> None:
    _DeltaMSq21 = DeltaMSq21[0]
    if is_dm32_leading:
        _DeltaMSq32 = nmo[0] * DeltaMSq3lAbs[0]  # Δm²₃₂ = α*|Δm²₃ₗ|
        _DeltaMSq31 = _DeltaMSq32 + _DeltaMSq21  # Δm²₃₁ = Δm²₃₂ + Δm²₂₁
    else:
        _DeltaMSq31 = nmo[0] * DeltaMSq3lAbs[0]  # Δm²₃₁ = α*|Δm²₃ₗ|
        _DeltaMSq32 = _DeltaMSq31 - _DeltaMSq21  # Δm²₃₂ = Δm²₃₁ - Δm²₂₁
    _SinSq2Theta13 = SinSq2Theta13[0]
    _SinSq2Theta12 = SinSq2Theta12[0]

    _SinSqTheta12 = 0.5 * (1.0 - sqrt(1.0 - _SinSq2Theta12))
    _CosSqTheta12 = 1.0 - _SinSqTheta12
    _SinSqTheta13 = 0.5 * (1.0 - sqrt(1.0 - _SinSq2Theta13))
    _CosSqTheta13 = 1.0 - _SinSqTheta13
    _CosQuTheta13 = _CosSqTheta13 * _CosSqTheta13

    _SinSqDouble12 = _SinSqTheta12 * _CosSqTheta12
    _CosSqDouble12 = _CosSqTheta12 - _SinSqTheta12

    _DeltaMSqee = _CosSqTheta12 * _DeltaMSq31 + _SinSqTheta12 * _DeltaMSq32

    _density_const = (ElectronToNucleonFraction[0] / _aem_g * (-2.0) * sqrt(2.0) * _Gf *
                      _GF_CONVERSION / _MeV_to_eV)
    msw_factor = _density_const * density[0]

    Lcq = surprobArgConversion[0] * 0.25 * L[0] * L_scale

    for i in range(len(out)):
        a = msw_factor * E[i]

        common_21 = (_CosSqTheta13 / _DeltaMSq21) * a
        common_21_squared = common_21 * common_21

        SinSq13_mod = _SinSqTheta13 + (2.0 * _SinSqTheta13 * _CosSqTheta13 / _DeltaMSqee) * a
        CosSq13_mod = 1.0 - SinSq13_mod
        CosQu13_mod = CosSq13_mod * CosSq13_mod

        SinSq12_mod = (_SinSqTheta12
                         + 2.0 * _SinSqDouble12 * common_21
                         + 3.0 * _SinSqDouble12 * _CosSqDouble12 * common_21_squared)
        CosSq12_mod = 1.0 - SinSq12_mod

        DeltaMSq21_mod = (_DeltaMSq21
                      - _DeltaMSq21 * _CosSqDouble12 * common_21
                      + 2.0 * _DeltaMSq21 * _SinSqDouble12 * common_21_squared)

        common_3 = _SinSqDouble12 * _CosSqTheta13 * common_21

        DeltaMSq31_mod = (_DeltaMSq31
                      - a * ((_CosSqTheta12 * _CosSqTheta13 - _SinSqTheta13) - common_3))

        DeltaMSq32_mod = (_DeltaMSq32
                      - a * ((_SinSqTheta12 * _CosSqTheta13 - _SinSqTheta13) + common_3))

        SinArg21 = Lcq * (DeltaMSq21_mod / E[i])
        SinArg31 = Lcq * (DeltaMSq31_mod / E[i])
        SinArg32 = Lcq * (DeltaMSq32_mod / E[i])

        SinDelta31_mod = sin(SinArg31)
        SinSqDelta31_mod = SinDelta31_mod* SinDelta31_mod

        SinDelta21_mod = sin(SinArg21)
        SinSqDelta21_mod = SinDelta21_mod * SinDelta21_mod

        SinDelta32_mod = sin(SinArg32)
        SinSqDelta32_mod = SinDelta32_mod * SinDelta32_mod

        out[i] = (1.0
                - 4.0 * (CosQu13_mod
                         * SinSq12_mod * CosSq12_mod
                         * SinSqDelta21_mod)
                - 4.0 * (SinSq13_mod * CosSq13_mod
                         * (CosSq12_mod * SinSqDelta31_mod
                            + SinSq12_mod * SinSqDelta32_mod)))


class NueSurvivalProbabilityKhan(Node):
    """
    inputs:
        `E`: array of the energies
        `L`: the distance
        `SinSq2Theta12`: Sin²2θ₁₂
        `SinSq2Theta13`: Sin²2θ₁₃
        `DeltaMSq21`: Δm²₂₁
        `DeltaMSq32` or `DeltaMSq31`: |Δm²₃₂| or |Δm²₃₁| depending on `leading_mass_splitting_3l_name`
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
        "_DeltaMSq3lAbs",
        "_is_dm32_leading",
        "_nmo",
        "_ElectronToNucleonFraction",
        "_density",
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
    _ElectronToNucleonFraction: NDArray
    _density: NDArray
    _surprob_arg_conversion_factor: NDArray
    _result: NDArray

    def __init__(
            self,
            *args,
            leading_mass_splitting_3l_name: Literal["DeltaMSq31", "DeltaMSq32"],
            distance_unit: Literal["km", "m"] = "km",
            **kwargs,
    ):
        if leading_mass_splitting_3l_name not in {"DeltaMSq31", "DeltaMSq32"}:
            raise RuntimeError(f"Do not support switch: {leading_mass_splitting_3l_name=}")
        super().__init__(
            *args,
            **kwargs,
            allowed_kw_inputs=(
                "E",
                "L",
                "SinSq2Theta13",
                "SinSq2Theta12",
                leading_mass_splitting_3l_name,
                "DeltaMSq21",
                "nmo",
                "ElectronToNucleonFraction",
                "density",
                "surprobArgConversion",
            ),
        )
        self._fd.needs_post_allocate = True
        self._is_dm32_leading = leading_mass_splitting_3l_name == "DeltaMSq32"
        self._labels.setdefault("mark", "P(ee)")
        self._add_pair("E", "result")
        self._add_input("L", positional=False)
        self._add_input("SinSq2Theta12", positional=False)
        self._add_input("SinSq2Theta13", positional=False)
        self._add_input("DeltaMSq21", positional=False)
        self._add_input(leading_mass_splitting_3l_name, positional=False)
        self._add_input("nmo", positional=False)
        self._add_input("ElectronToNucleonFraction", positional=False)
        self._add_input("density", positional=False)
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
                "DeltaMSq32" if self._is_dm32_leading else "DeltaMSq31",
                "nmo",
                "ElectronToNucleonFraction",
                "density",
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
            self._DeltaMSq3lAbs,
            self._is_dm32_leading,
            self._nmo,
            self._ElectronToNucleonFraction,
            self._density,
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
        self._DeltaMSq3lAbs = self.inputs[
            "DeltaMSq32" if self._is_dm32_leading else "DeltaMSq31"
        ]._data
        self._nmo = self.inputs["nmo"]._data
        self._ElectronToNucleonFraction = self.inputs["ElectronToNucleonFraction"]._data
        self._density = self.inputs["density"]._data

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
