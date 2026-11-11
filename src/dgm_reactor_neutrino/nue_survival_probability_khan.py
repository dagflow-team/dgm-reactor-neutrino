from __future__ import annotations

from typing import TYPE_CHECKING

from dag_modelling.core.global_parameters import NUMBA_CACHE_ENABLE
from dag_modelling.core.node import Node
from dag_modelling.core.storage import NodeStorage
from dag_modelling.core.type_functions import (
    assign_axes_from_inputs_to_outputs,
    check_shape_of_inputs,
    copy_from_inputs_to_outputs,
)
from numba import njit
from numpy import array, pi, sin, sqrt
from scipy.constants import value as constant

if TYPE_CHECKING:
    from typing import Literal

    from dag_modelling.core.output import Output
    from nested_mapping.typing import KeyLike
    from numpy import double
    from numpy.typing import NDArray

_surprobArgConversion = pi * 2e-3 * constant("electron volt-inverse meter relationship")
_constant_aem_g = constant("atomic mass constant") * 1000
_constant_Gf = constant("Fermi coupling constant")
_constant_hbar = constant("reduced Planck constant")
_constant_c = constant("speed of light in vacuum")
_constant_e = constant("elementary charge")
_constant_hbar_c_eV_cm = (_constant_hbar * _constant_c / _constant_e) * 100
_constant_SqGeV_to_SqMeV = 1e-18
_constant_GF_CONVERSION = _constant_SqGeV_to_SqMeV * (_constant_hbar_c_eV_cm**3)
_constant_MeV_to_eV = 1e-6
_electron_density_to_matter_potential = (
    (-2.0)
    / _constant_aem_g
    * sqrt(2.0)
    * _constant_Gf
    * _constant_GF_CONVERSION
    / _constant_MeV_to_eV
)
del (
    _constant_hbar,
    _constant_c,
    _constant_e,
    _constant_hbar_c_eV_cm,
    _constant_SqGeV_to_SqMeV,
    _constant_aem_g,
    _constant_Gf,
    _constant_GF_CONVERSION,
    _constant_MeV_to_eV,
)


@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_expansion(
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
    matter_density: NDArray[double],
    surprobArgConversion: NDArray[double],
) -> None:
    _DeltaMSq21 = DeltaMSq21[0]

    _SinSq2Theta13 = SinSq2Theta13[0]
    _SinSq2Theta12 = SinSq2Theta12[0]

    Cos2Theta12 = sqrt(1.0 - _SinSq2Theta12)  # cos(2θ₁₂) = √(1 - sin²(2θ₁₂))
    SinSqTheta12 = 0.5 * (1.0 - Cos2Theta12)  # sin²(θ₁₂) = (1 - cos(2θ₁₂)) / 2
    CosSqTheta12 = 1.0 - SinSqTheta12  # cos²(θ₁₂) = 1 - sin²(θ₁₂)

    Cos2Theta13 = sqrt(1.0 - _SinSq2Theta13)  # cos(2θ₁₃) = √(1 - sin²(2θ₁₃))
    SinSqTheta13 = 0.5 * (1.0 - Cos2Theta13)  # sin²(θ₁₃) = (1 - cos(2θ₁₃)) / 2
    CosSqTheta13 = 1.0 - SinSqTheta13  # cos²(θ₁₃) = 1 - sin²(θ₁₃)

    if is_dm32_leading:
        DeltaMSq32 = nmo[0] * DeltaMSq3lAbs[0]  # Δm²₃₂ = α·|Δm²₃ₗ|
        DeltaMSq31 = DeltaMSq32 + _DeltaMSq21  # Δm²₃₁ = Δm²₃₂ + Δm²₂₁
        DeltaMSqee = DeltaMSq32 + CosSqTheta12 * _DeltaMSq21  # Δm²(ee)= Δm²₃₂ + cos²(θ₁₂) · Δm²₂₁
    else:
        DeltaMSq31 = nmo[0] * DeltaMSq3lAbs[0]  # Δm²₃₁ = α·|Δm²₃ₗ|
        DeltaMSq32 = DeltaMSq31 - _DeltaMSq21  # Δm²₃₂ = Δm²₃₁ - Δm²₂₁
        DeltaMSqee = DeltaMSq31 - SinSqTheta12 * _DeltaMSq21  # Δm²(ee)= Δm²₃₁ - sin²(θ₁₂) · Δm²₂₁

    _density_const = ElectronToNucleonFraction[0] * _electron_density_to_matter_potential
    msw_factor = _density_const * matter_density[0]

    Lcq = surprobArgConversion[0] * 0.25 * L[0] * L_scale

    for i in range(len(out)):
        enu = E[i]
        variable_a = msw_factor * enu  # eq. (5)

        variable_a_over_dm21 = variable_a / _DeltaMSq21
        variable_a_over_dmee = variable_a / DeltaMSqee

        common_ratio = CosSqTheta13 * variable_a_over_dm21
        common_ratio_squared = common_ratio * common_ratio

        SinSqTheta13_tilde = SinSqTheta13 * (
            1 + 2.0 * CosSqTheta13 * variable_a_over_dmee
        )  # eq. (17)
        CosSqTheta13_tilde = 1.0 - SinSqTheta13_tilde  # cos²(θ₁₃~) = 1 - sin²(θ₁₃~)
        SinSq2Theta13_tilde = (
            4.0 * SinSqTheta13_tilde * CosSqTheta13_tilde
        )  # sin²2(θ₁₃~) = 4sin²(θ₁₃~)cos²(θ₁₃~)
        CosQu13_tilde = CosSqTheta13_tilde * CosSqTheta13_tilde  # cos⁴(θ₁₃~) = (cos²(θ₁₃~))²

        SinSqTheta12_tilde = (
            SinSqTheta12
            + 0.5 * _SinSq2Theta12 * common_ratio
            + 0.75 * _SinSq2Theta12 * Cos2Theta12 * common_ratio_squared
        )  # eq. (15)
        # Imply sin²θ₁₂·cos²θ₁₂ = sin²2θ₁₂/4

        CosSqTheta12_tilde = 1.0 - SinSqTheta12_tilde  # cos²(θ₁₂~) = 1 - sin²(θ₁₂~)
        SinSq2Theta12_tilde = (
            4.0 * SinSqTheta12_tilde * CosSqTheta12_tilde
        )  # sin²2(θ₁₂~) = 4sin²(θ₁₂~)cos²(θ₁₂~)

        DeltaMSq21_tilde = _DeltaMSq21 * (
            1.0 - Cos2Theta12 * common_ratio + 0.5 * _SinSq2Theta12 * common_ratio_squared
        )  # eq. (16)
        # Imply 2·sin²θ₁₂·cos²θ₁₂ = sin²2θ₁₂/2

        common_3l = 0.25 * _SinSq2Theta12 * CosSqTheta13 * common_ratio  # in eq. (19−20)
        # Imply sin²θ₁₂·cos²θ₁₂ = sin²2θ₁₂/4

        DeltaMSq31_tilde = DeltaMSq31 - variable_a * (
            (CosSqTheta12 * CosSqTheta13 - SinSqTheta13) - common_3l
        )  # eq. (19)

        DeltaMSq32_tilde = DeltaMSq32 - variable_a * (
            (SinSqTheta12 * CosSqTheta13 - SinSqTheta13) + common_3l
        )  # eq. (20)

        Lcq_over_enu = Lcq / enu
        SinArg21 = Lcq_over_enu * DeltaMSq21_tilde
        SinArg31 = Lcq_over_enu * DeltaMSq31_tilde
        SinArg32 = Lcq_over_enu * DeltaMSq32_tilde

        SinDelta31_tilde = sin(SinArg31)  # sin(Δ₃₁)
        SinSqDelta31_tilde = SinDelta31_tilde * SinDelta31_tilde

        SinDelta21_tilde = sin(SinArg21)  # sin(Δ₂₁)
        SinSqDelta21_tilde = SinDelta21_tilde * SinDelta21_tilde

        SinDelta32_tilde = sin(SinArg32)  # sin(Δ₃₂)
        SinSqDelta32_tilde = SinDelta32_tilde * SinDelta32_tilde

        out[i] = (
            1.0
            - CosQu13_tilde * SinSq2Theta12_tilde * SinSqDelta21_tilde
            - SinSq2Theta13_tilde
            * (CosSqTheta12_tilde * SinSqDelta31_tilde + SinSqTheta12_tilde * SinSqDelta32_tilde)
        )  # eq. (4)


@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_main(
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
    _SinSq2Theta13 = SinSq2Theta13[0]
    _SinSq2Theta12 = SinSq2Theta12[0]

    Cos2Theta12 = sqrt(1.0 - _SinSq2Theta12)  # cos(2θ₁₂) = √(1 - sin²(2θ₁₂))
    SinSqTheta12 = 0.5 * (1.0 - Cos2Theta12)  # sin²(θ₁₂) = (1 - cos(2θ₁₂)) / 2
    CosSqTheta12 = 1.0 - SinSqTheta12  # cos²(θ₁₂) = 1 - sin²(θ₁₂)

    Cos2Theta13 = sqrt(1.0 - _SinSq2Theta13)  # cos(2θ₁₃) = √(1 - sin²(2θ₁₃))
    if is_dm32_leading:
        DeltaMSq32 = nmo[0] * DeltaMSq3lAbs[0]  # Δm²₃₂ = α·|Δm²₃ₗ|
        DeltaMSqee = DeltaMSq32 + CosSqTheta12 * _DeltaMSq21  # Δm²(ee)= Δm²₃₂ + cos²(θ₁₂) · Δm²₂₁
    else:
        DeltaMSq31 = nmo[0] * DeltaMSq3lAbs[0]  # Δm²₃₁ = α·|Δm²₃ₗ|
        DeltaMSqee = DeltaMSq31 - SinSqTheta12 * _DeltaMSq21  # Δm²(ee)= Δm²₃₁ - sin²(θ₁₂) · Δm²₂₁

    _density_const = ElectronToNucleonFraction[0] * _electron_density_to_matter_potential
    msw_factor = _density_const * density[0]

    Lcq = surprobArgConversion[0] * 0.25 * L[0] * L_scale

    for i in range(len(out)):
        enu = E[i]
        variable_a = msw_factor * enu  # eq. (5)

        a_over_dmee = variable_a / DeltaMSqee
        Cos2Theta13_shifted = Cos2Theta13 - a_over_dmee
        DeltaMSqee_tilde = DeltaMSqee * sqrt(
            Cos2Theta13_shifted * Cos2Theta13_shifted + _SinSq2Theta13
        )  # eq. (6)

        Cos2Theta13_tilde = (DeltaMSqee * Cos2Theta13 - variable_a) / DeltaMSqee_tilde  # eq. (7)
        SinSq2Theta13_tilde = (
            1.0 - Cos2Theta13_tilde * Cos2Theta13_tilde
        )  # sin²(2θ₁₃~) = 1 - cos²(2θ₁₃~)
        CosSqTheta13_tilde = (1.0 + Cos2Theta13_tilde) * 0.5  # cos²(θ₁₃~) = (1 + cos(2θ₁₃~)) / 2
        CosQuTheta13_tilde = CosSqTheta13_tilde * CosSqTheta13_tilde

        variable_a_prime = (variable_a + DeltaMSqee - DeltaMSqee_tilde) * 0.5  # eq. (11)
        a_prime_over_dm21 = variable_a_prime / _DeltaMSq21
        Cos2Theta12_shifted = Cos2Theta12 - a_prime_over_dm21
        CosSqDiff_Theta13tilde_Theta13 = (
            0.5 * (DeltaMSqee_tilde + DeltaMSqee - variable_a * Cos2Theta13) / DeltaMSqee_tilde
        )  # eq. (12)
        DeltaMSq21_tilde = _DeltaMSq21 * sqrt(
            Cos2Theta12_shifted * Cos2Theta12_shifted
            + CosSqDiff_Theta13tilde_Theta13 * _SinSq2Theta12
        )  # eq. (10)

        Cos2Theta12_tilde = (
            _DeltaMSq21 * Cos2Theta12 - variable_a_prime
        ) / DeltaMSq21_tilde  # eq. (9)
        SinSq2Theta12_tilde = (
            1.0 - Cos2Theta12_tilde * Cos2Theta12_tilde
        )  # sin²(2θ₁₂~) = 1 - cos²(2θ₁₂~)
        SinSqTheta12_tilde = (1.0 - Cos2Theta12_tilde) * 0.5  # sin²(θ₁₂~) = (1 - cos(2θ₁₂~)) / 2
        CosSqTheta12_tilde = (1.0 + Cos2Theta12_tilde) * 0.5  # cos²(θ₁₂~) = (1 + cos(2θ₁₂~)) / 2

        DeltaMSq31_tilde = DeltaMSqee_tilde + SinSqTheta12_tilde * DeltaMSq21_tilde  # eq. (13)
        DeltaMSq32_tilde = DeltaMSq31_tilde - DeltaMSq21_tilde  # eq. (14)

        Lcq_over_enu = Lcq / enu
        SinArg21 = Lcq_over_enu * DeltaMSq21_tilde
        SinArg31 = Lcq_over_enu * DeltaMSq31_tilde
        SinArg32 = Lcq_over_enu * DeltaMSq32_tilde

        SinDelta31_tilde = sin(SinArg31)  # sin(Δ₃₁)
        SinSqDelta31_tilde = SinDelta31_tilde * SinDelta31_tilde

        SinDelta21_tilde = sin(SinArg21)  # sin(Δ₂₁)
        SinSqDelta21_tilde = SinDelta21_tilde * SinDelta21_tilde

        SinDelta32_tilde = sin(SinArg32)  # sin(Δ₃₂)
        SinSqDelta32_tilde = SinDelta32_tilde * SinDelta32_tilde

        out[i] = (
            1.0
            - CosQuTheta13_tilde * SinSq2Theta12_tilde * SinSqDelta21_tilde
            - SinSq2Theta13_tilde
            * (CosSqTheta12_tilde * SinSqDelta31_tilde + SinSqTheta12_tilde * SinSqDelta32_tilde)
        )  # eq. (4)


class NueSurvivalProbabilityKhan(Node):
    """
    Calculates the electron neutrino survival probability P(ee) using the analytical approximation
    by Khan et al.

    Two computation modes are available:
    - "main" (default): uses the exact matter‑modified effective mixing parameters (sines,
      mass‑squared differences) by solving the evolution equations in matter. This mode is accurate
      for all densities and should be used for general cases.
    - "expansion": uses a perturbative expansion of the oscillation parameters (sin²2θ, Δm²) around
      zero matter potential (a = 0). This approximation is valid for small matter potentials and is
      computationally faster, but may become inaccurate at high densities or for large matter
      effects.

    inputs:
        `E`: array of the energies
        `L`: the distance
        `SinSq2Theta12`: Sin²2θ₁₂
        `SinSq2Theta13`: Sin²2θ₁₃
        `DeltaMSq21`: Δm²₂₁
        `DeltaMSq32` or `DeltaMSq31`: |Δm²₃₂| or |Δm²₃₁| depending on `leading_mass_splitting_3l_name`
        `nmo`: α - the mass ordering constant

    optional inputs:
        - `surprobArgConversion`: Convert Δm²[eV²]L[km]/E[MeV] to natural units. If the input is not
           given a default value will be used:
               `2*pi*1e-3*scipy.value('electron volt-inverse meter relationship')`

    outputs:
        `0` or `result`: array of probabilities

    Parameters
    ----------
        leading_mass_splitting_3l_name : Literal["DeltaMSq31", "DeltaMSq32"]
            Selects which mass-squared difference will be provided as input and which one will be
            calculated.
        mode: Literal["main", "expansion"], default="main"
            Selects, which sets of formulas from the paper will be used. See beginning of the
            documentation.
        distance_unit: Literal["km", "m"], default="km"
            Distance units for L.
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
        "_matter_density",
        "_surprob_arg_conversion_factor",
    )

    _baseline_scale: float
    _E: NDArray
    _L: NDArray
    _nmo: NDArray
    _DeltaMSq21: NDArray
    _DeltaMSq3lAbs: NDArray
    _SinSq2Theta12: NDArray
    _SinSq2Theta13: NDArray
    _ElectronToNucleonFraction: NDArray
    _matter_density: NDArray
    _surprob_arg_conversion_factor: NDArray
    _result: NDArray

    def __init__(
        self,
        *args,
        leading_mass_splitting_3l_name: Literal["DeltaMSq31", "DeltaMSq32"],
        mode: Literal["main", "expansion"] = "main",
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
        self.needs_post_allocate = True
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
        self._add_input("matter_density", positional=False)
        try:
            self._baseline_scale = {"km": 1, "m": 1.0e-3}[distance_unit]
        except KeyError as e:
            raise RuntimeError(f"Invalid distance unit {distance_unit}") from e

        self._functions_dict.update(
            {
                "expansion": self._fcn_expansion,
                "main": self._fcn_main,
            }
        )

        if mode not in self._functions_dict:
            raise ValueError(
                f"Unknown mode '{mode}'. Expected one of {tuple(self._functions_dict.keys())}"
            )
        self.function = self._functions_dict[mode]

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
                "matter_density",
            ),
            (1,),
        )
        copy_from_inputs_to_outputs(self, "E", "result")
        assign_axes_from_inputs_to_outputs(
            self, "E", "result", assign_meshes=True, overwrite_assigned=True
        )

    def _fcn_expansion(self):
        for callback in self._input_nodes_callbacks:
            callback()

        _sur_prob_expansion(
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
            self._matter_density,
            self._surprob_arg_conversion_factor,
        )

    def _fcn_main(self):
        for callback in self._input_nodes_callbacks:
            callback()

        _sur_prob_main(
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
            self._matter_density,
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
        self._matter_density = self.inputs["matter_density"]._data

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
                if surprobArgConversion is True:
                    inputs[nametuple + ("surprobArgConversion",) + key] = surprob(
                        "surprobArgConversion"
                    )
                else:
                    surprobArgConversion >> surprob("surprobArgConversion")

        NodeStorage.update_current(storage, strict=True, verbose=verbose)

        return None, storage
