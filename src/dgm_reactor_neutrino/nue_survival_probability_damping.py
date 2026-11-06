from __future__ import annotations

from typing import TYPE_CHECKING, override

from dag_modelling.core.global_parameters import NUMBA_CACHE_ENABLE
from dag_modelling.core.node import Node
from dag_modelling.core.storage import NodeStorage
from dag_modelling.core.type_functions import (
    assign_axes_from_inputs_to_outputs,
    check_shape_of_inputs,
    copy_from_inputs_to_outputs,
)
from numba import njit
from numpy import array, cos, exp, expm1, pi, sin, sqrt
from scipy.constants import value

if TYPE_CHECKING:
    from typing import Literal

    from dag_modelling.core.node import Node
    from dag_modelling.core.output import Output
    from nested_mapping.typing import KeyLike
    from numpy import double
    from numpy.typing import NDArray

_surprobArgConversion = pi * 2e-3 * value("electron volt-inverse meter relationship")

_DAMPING_MODELS = {"none", "decay", "wpd1", "wpd2"}


@njit(cache=NUMBA_CACHE_ENABLE)
def _get_common_amplitudes(SinSq2Theta12, SinSq2Theta13):
    SinSqTheta12 = 0.5 * (1 - sqrt(1 - SinSq2Theta12))  # sin²θ₁₂
    CosSqTheta12 = 1.0 - SinSqTheta12  # cos²θ₁₂
    SinSqTheta13 = 0.5 * (1 - sqrt(1 - SinSq2Theta13))  # sin²θ₁₃
    CosSqTheta13 = 1.0 - SinSqTheta13  # cos²θ₁₃
    CosQuTheta13 = CosSqTheta13 * CosSqTheta13  # cos⁴θ₁₃
    SinQuTheta13 = SinSqTheta13 * SinSqTheta13  # sin⁴θ₁₃

    norm = 1.0 - 0.5 * (CosQuTheta13 * SinSq2Theta12 + SinSq2Theta13)
    amp21 = 0.5 * CosQuTheta13 * SinSq2Theta12
    amp31 = 0.5 * SinSq2Theta13 * CosSqTheta12
    amp32 = 0.5 * SinSq2Theta13 * SinSqTheta12

    return norm, amp21, amp31, amp32, SinQuTheta13


@njit(cache=NUMBA_CACHE_ENABLE)
def _get_mass_splittings(DeltaMSq21, DeltaMSq3lAbs, is_dm32_leading, nmo):
    if is_dm32_leading:
        DeltaMSq32 = nmo * DeltaMSq3lAbs
        DeltaMSq31 = DeltaMSq32 + DeltaMSq21
    else:
        DeltaMSq31 = nmo * DeltaMSq3lAbs
        DeltaMSq32 = DeltaMSq31 - DeltaMSq21

    return DeltaMSq31, DeltaMSq32


# ---------------------------------------------------------------------------
# One numba kernel per model.
# Each kernel only receives the parameters that model uses.
# ---------------------------------------------------------------------------
@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_none(
    out: NDArray[double],
    E: NDArray[double],
    L: NDArray[double],
    L_scale: float,
    SinSq2Theta12: NDArray[double],
    SinSq2Theta13: NDArray[double],
    DeltaMSq21: NDArray[double],
    DeltaMSq3lAbs: NDArray[double],
    is_dm32_leading,
    nmo: NDArray[double],
    surprobArgConversion: NDArray[double],
) -> None:
    norm, amp21, amp31, amp32, SinQuTheta13 = _get_common_amplitudes(
        SinSq2Theta12[0], SinSq2Theta13[0]
    )
    _DeltaMSq21 = DeltaMSq21[0]
    DeltaMSq31, DeltaMSq32 = _get_mass_splittings(
        _DeltaMSq21, DeltaMSq3lAbs[0], is_dm32_leading, nmo[0]
    )
    cosCommonArg = surprobArgConversion[0] * L[0] * 0.5 * L_scale
    for i in range(len(out)):
        L2E = cosCommonArg / E[i]
        out[i] = (
            norm
            + amp21 * cos(_DeltaMSq21 * L2E)
            + amp31 * cos(DeltaMSq31 * L2E)
            + amp32 * cos(DeltaMSq32 * L2E)
        )


@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_decay(
    out: NDArray[double],
    E: NDArray[double],
    L: NDArray[double],
    L_scale: float,
    SinSq2Theta12: NDArray[double],
    SinSq2Theta13: NDArray[double],
    DeltaMSq21: NDArray[double],
    DeltaMSq3lAbs: NDArray[double],
    is_dm32_leading,
    nmo: NDArray[double],
    surprobArgConversion: NDArray[double],
    alpha_decay: NDArray[double],
) -> None:
    norm, amp21, amp31, amp32, SinQuTheta13 = _get_common_amplitudes(
        SinSq2Theta12[0], SinSq2Theta13[0]
    )
    _DeltaMSq21 = DeltaMSq21[0]
    DeltaMSq31, DeltaMSq32 = _get_mass_splittings(
        _DeltaMSq21, DeltaMSq3lAbs[0], is_dm32_leading, nmo[0]
    )
    L_km = L[0] * L_scale
    sinCommonArg = surprobArgConversion[0] * 0.5 * L_km
    alpha_L_m = alpha_decay[0] * L_km * 1000.0
    for i in range(len(out)):
        E_i = E[i]
        L2E = sinCommonArg / E_i
        alpha_E = alpha_L_m / E_i
        DampingFactor = exp(-0.5 * alpha_E)
        correction = SinQuTheta13 * expm1(-alpha_E)
        out[i] = (
            norm
            + amp21 * cos(_DeltaMSq21 * L2E)
            + amp31 * DampingFactor * cos(DeltaMSq31 * L2E)
            + amp32 * DampingFactor * cos(DeltaMSq32 * L2E)
            + correction
        )


@njit(cache=NUMBA_CACHE_ENABLE)
def _get_damping_arguments_wpd(alpha_wpd, DeltaMSq21, DeltaMSq31, DeltaMSq32, L):
    L2 = L * L
    damping_arg_21 = alpha_wpd * DeltaMSq21 * DeltaMSq21 * L2
    damping_arg_31 = alpha_wpd * DeltaMSq31 * DeltaMSq31 * L2
    damping_arg_32 = alpha_wpd * DeltaMSq32 * DeltaMSq32 * L2

    return damping_arg_21, damping_arg_31, damping_arg_32


@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_wpd1(
    out: NDArray[double],
    E: NDArray[double],
    L: NDArray[double],
    L_scale: float,
    SinSq2Theta12: NDArray[double],
    SinSq2Theta13: NDArray[double],
    DeltaMSq21: NDArray[double],
    DeltaMSq3lAbs: NDArray[double],
    is_dm32_leading,
    nmo: NDArray[double],
    surprobArgConversion: NDArray[double],
    alpha_wpd1: NDArray[double],
) -> None:
    norm, amp21, amp31, amp32, SinQuTheta13 = _get_common_amplitudes(
        SinSq2Theta12[0], SinSq2Theta13[0]
    )
    _DeltaMSq21 = DeltaMSq21[0]
    DeltaMSq31, DeltaMSq32 = _get_mass_splittings(
        _DeltaMSq21, DeltaMSq3lAbs[0], is_dm32_leading, nmo[0]
    )
    sinCommonArg = surprobArgConversion[0] * L[0] * 0.5 * L_scale
    damping_arg_21, damping_arg_31, damping_arg_32 = _get_damping_arguments_wpd(
        alpha_wpd1[0], _DeltaMSq21, DeltaMSq31, DeltaMSq32, L[0]
    )
    for i in range(len(out)):
        E_i = E[i]
        E2 = E_i * E_i
        inv_E4 = 1.0 / (E2 * E2)
        L2E = sinCommonArg / E_i
        out[i] = (
            norm
            + amp21 * exp(-damping_arg_21 * inv_E4) * cos(_DeltaMSq21 * L2E)
            + amp31 * exp(-damping_arg_31 * inv_E4) * cos(DeltaMSq31 * L2E)
            + amp32 * exp(-damping_arg_32 * inv_E4) * cos(DeltaMSq32 * L2E)
        )


@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_wpd2(
    out: NDArray[double],
    E: NDArray[double],
    L: NDArray[double],
    L_scale: float,
    SinSq2Theta12: NDArray[double],
    SinSq2Theta13: NDArray[double],
    DeltaMSq21: NDArray[double],
    DeltaMSq3lAbs: NDArray[double],
    is_dm32_leading,
    nmo: NDArray[double],
    surprobArgConversion: NDArray[double],
    alpha_wpd2: NDArray[double],
) -> None:
    norm, amp21, amp31, amp32, SinQuTheta13 = _get_common_amplitudes(
        SinSq2Theta12[0], SinSq2Theta13[0]
    )
    _DeltaMSq21 = DeltaMSq21[0]
    DeltaMSq31, DeltaMSq32 = _get_mass_splittings(
        _DeltaMSq21, DeltaMSq3lAbs[0], is_dm32_leading, nmo[0]
    )
    sinCommonArg = surprobArgConversion[0] * L[0] * 0.5 * L_scale
    damping_arg_21, damping_arg_31, damping_arg_32 = _get_damping_arguments_wpd(
        alpha_wpd2[0], _DeltaMSq21, DeltaMSq31, DeltaMSq32, L[0]
    )
    for i in range(len(out)):
        E_i = E[i]
        inv_E2 = 1.0 / (E_i * E_i)
        L2E = sinCommonArg / E_i
        out[i] = (
            norm
            + amp21 * exp(-damping_arg_21 * inv_E2) * cos(_DeltaMSq21 * L2E)
            + amp31 * exp(-damping_arg_31 * inv_E2) * cos(DeltaMSq31 * L2E)
            + amp32 * exp(-damping_arg_32 * inv_E2) * cos(DeltaMSq32 * L2E)
        )


class NueSurvivalProbabilityDamping(Node):
    """Electron antineutrino survival probability in vacuum with damping.

    Implements the standard three-flavor P(ν̄ₑ → ν̄ₑ) modulated by one of
    three damping signatures from the JUNO sensitivity paper
    (JHEP 06 (2022) 062):

      * ``"decay"`` : ν₃ invisible decay (rate suppression)
      * ``"wpd1"``  : Wave Packet Decoherence I (rate preserving)
      * ``"wpd2"``  : Wave Packet Decoherence II (rate preserving)
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
        "_surprob_arg_conversion_factor",
        "_alpha",
        "_model_name",
    )

    _baseline_scale: float
    _E: NDArray
    _L: NDArray
    _nmo: NDArray
    _DeltaMSq21: NDArray
    _DeltaMSq3lAbs: NDArray
    _SinSq2Theta12: NDArray
    _SinSq2Theta13: NDArray
    _surprob_arg_conversion_factor: NDArray
    _alpha: NDArray
    _result: NDArray
    _model_name: str

    def __init__(
        self,
        *args,
        leading_mass_splitting_3l_name: Literal["DeltaMSq31", "DeltaMSq32"],
        distance_unit: Literal["km", "m"] = "km",
        damping_model: Literal["none", "decay", "wpd1", "wpd2"] = "none",
        **kwargs,
    ):
        if leading_mass_splitting_3l_name not in {"DeltaMSq31", "DeltaMSq32"}:
            raise RuntimeError(f"Do not support switch: {leading_mass_splitting_3l_name=}")

        # Resolve the model at construction (validation only; selection
        # happens in _type_function, matching Chi2's lifecycle).
        if damping_model not in _DAMPING_MODELS:
            raise ValueError(
                f"Unknown damping_model: {damping_model!r} "
                f"(expected one of {sorted(_DAMPING_MODELS)})"
            )
        model_name = damping_model

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
                "surprobArgConversion",
                "alpha",
            ),
        )
        self.needs_post_allocate = True
        self._is_dm32_leading = leading_mass_splitting_3l_name == "DeltaMSq32"
        self._labels.setdefault("mark", "P(ee)")
        self._model_name = model_name

        # ---- Register variants in _evaluator._functions_dict ----
        self._evaluator._functions_dict.update(
            {
                "none": self._function_none,
                "decay": self._function_decay,
                "wpd1": self._function_wpd1,
                "wpd2": self._function_wpd2,
            }
        )

        self._add_pair("E", "result")
        self._add_input("L", positional=False)
        self._add_input("SinSq2Theta12", positional=False)
        self._add_input("SinSq2Theta13", positional=False)
        self._add_input("DeltaMSq21", positional=False)
        self._add_input(leading_mass_splitting_3l_name, positional=False)
        self._add_input("nmo", positional=False)

        try:
            self._baseline_scale = {"km": 1, "m": 1.0e-3}[distance_unit]
        except KeyError as exc:
            raise RuntimeError(f"Invalid distance unit {distance_unit}") from exc

    # ------------------------------------------------------------------
    # One function as per model
    # ------------------------------------------------------------------
    def _function_none(self) -> None:
        for callback in self._evaluator._input_nodes_callbacks:
            callback()
        _sur_prob_none(
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
            self._surprob_arg_conversion_factor,
        )

    def _function_decay(self) -> None:
        for callback in self._evaluator._input_nodes_callbacks:
            callback()
        _sur_prob_decay(
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
            self._surprob_arg_conversion_factor,
            self._alpha,
        )

    def _function_wpd1(self) -> None:
        for callback in self._evaluator._input_nodes_callbacks:
            callback()
        _sur_prob_wpd1(
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
            self._surprob_arg_conversion_factor,
            self._alpha,
        )

    def _function_wpd2(self) -> None:
        for callback in self._evaluator._input_nodes_callbacks:
            callback()
        _sur_prob_wpd2(
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
            self._surprob_arg_conversion_factor,
            self._alpha,
        )

    # ----------------------------------------
    #    Selection of damping function
    # ----------------------------------------
    @override
    def _type_function(self) -> None:
        check_shape_of_inputs(
            self,
            (
                "L",
                "SinSq2Theta12",
                "SinSq2Theta13",
                "DeltaMSq21",
                "DeltaMSq32" if self._is_dm32_leading else "DeltaMSq31",
                "nmo",
            ),
            (1,),
        )
        for opt_name in ("surprobArgConversion", "alpha"):
            if self.inputs.get(opt_name) is not None:
                check_shape_of_inputs(self, (opt_name,), (1,))
        copy_from_inputs_to_outputs(self, "E", "result")
        assign_axes_from_inputs_to_outputs(
            self, "E", "result", assign_meshes=True, overwrite_assigned=True
        )

        # ---- Select the variant here ----
        self._evaluator.function = self._evaluator._functions_dict[self._model_name]

    @override
    def _post_allocate(self) -> None:
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

        self._surprob_arg_conversion_factor = (
            cin._data
            if (cin := self.inputs.get("surprobArgConversion"))
            else array([_surprobArgConversion])
        )
        self._alpha = (
            cin._data if (cin := self.inputs.get("alpha")) else array([_surprobArgConversion])
        )

    @classmethod
    def replicate(
        cls,
        *args,
        name: str,
        replicate_outputs: tuple[KeyLike, ...] = ((),),
        surprobArgConversion: Output | Literal[True] | None = None,
        verbose: bool = False,
        **kwargs,
    ):
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

            inputs[nametuple + ("alpha",) + key] = surprob("alpha")

        NodeStorage.update_current(storage, strict=True, verbose=verbose)
        return None, storage
