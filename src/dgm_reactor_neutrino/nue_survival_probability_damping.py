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

_DAMPING_NONE = 0
_DAMPING_DECAY = 1
_DAMPING_WPD1 = 2
_DAMPING_WPD2 = 3

_DAMPING_MODEL_MAP = {
    "none": _DAMPING_NONE,
    "decay": _DAMPING_DECAY,
    "wpd1": _DAMPING_WPD1,
    "wpd2": _DAMPING_WPD2,
}


# ---------------------------------------------------------------------------
# One numba kernel per model — "Medium+".
# Each kernel only receives the parameters that model uses.
# ---------------------------------------------------------------------------
@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_none(out, E, sinCommonArg, Dm21, Dm31, Dm32,
                   norm, amp21, amp31, amp32) -> None:
    for i in range(len(out)):
        L4E = sinCommonArg / E[i]
        out[i] = (norm
                  + amp21 * cos(2.0 * Dm21 * L4E)
                  + amp31 * cos(2.0 * Dm31 * L4E)
                  + amp32 * cos(2.0 * Dm32 * L4E))


@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_decay(out, E, sinCommonArg, Dm21, Dm31, Dm32,
                    norm, amp21, amp31, amp32, aL_m, sin4t13) -> None:
    for i in range(len(out)):
        E_i = E[i]
        L4E = sinCommonArg / E_i
        d = exp(-0.5 * aL_m / E_i)
        correction = sin4t13 * expm1(-aL_m / E_i)
        out[i] = (norm
                  + amp21 * cos(2.0 * Dm21 * L4E)
                  + amp31 * d * cos(2.0 * Dm31 * L4E)
                  + amp32 * d * cos(2.0 * Dm32 * L4E)
                  + correction)


@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_wpd1(out, E, sinCommonArg, Dm21, Dm31, Dm32,
                   norm, amp21, amp31, amp32, pre21, pre31, pre32) -> None:
    for i in range(len(out)):
        E_i = E[i]
        E2 = E_i * E_i
        inv_E4 = 1.0 / (E2 * E2)
        L4E = sinCommonArg / E_i
        out[i] = (norm
                  + amp21 * exp(-pre21 * inv_E4) * cos(2.0 * Dm21 * L4E)
                  + amp31 * exp(-pre31 * inv_E4) * cos(2.0 * Dm31 * L4E)
                  + amp32 * exp(-pre32 * inv_E4) * cos(2.0 * Dm32 * L4E))


@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_wpd2(out, E, sinCommonArg, Dm21, Dm31, Dm32,
                   norm, amp21, amp31, amp32, pre21, pre31, pre32) -> None:
    for i in range(len(out)):
        E_i = E[i]
        inv_E2 = 1.0 / (E_i * E_i)
        L4E = sinCommonArg / E_i
        out[i] = (norm
                  + amp21 * exp(-pre21 * inv_E2) * cos(2.0 * Dm21 * L4E)
                  + amp31 * exp(-pre31 * inv_E2) * cos(2.0 * Dm31 * L4E)
                  + amp32 * exp(-pre32 * inv_E2) * cos(2.0 * Dm32 * L4E))


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
        "_baseline_scale", "_result", "_E", "_L",
        "_SinSq2Theta12", "_SinSq2Theta13",
        "_DeltaMSq21", "_DeltaMSq3lAbs",
        "_is_dm32_leading", "_nmo",
        "_surprob_arg_conversion_factor",
        "_alpha_decay", "_alpha_wpd1", "_alpha_wpd2",
        "_model_name",
        "_sinCommonArg", "_Dm21", "_Dm31", "_Dm32",
        "_norm", "_amp21", "_amp31", "_amp32",
        "_L_m", "_L2", "_s4T13",
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
    _alpha_decay: NDArray
    _alpha_wpd1: NDArray
    _alpha_wpd2: NDArray
    _result: NDArray
    _model_name: str

    def __init__(
        self,
        *args,
        leading_mass_splitting_3l_name: Literal["DeltaMSq31", "DeltaMSq32"],
        distance_unit: Literal["km", "m"] = "km",
        damping_model: Literal["none", "decay", "wpd1", "wpd2"] | int = "none",
        **kwargs,
    ):
        if leading_mass_splitting_3l_name not in {"DeltaMSq31", "DeltaMSq32"}:
            raise RuntimeError(
                f"Do not support switch: {leading_mass_splitting_3l_name=}"
            )

        # Resolve the model at construction (validation only; selection
        # happens in _type_function, matching Chi2's lifecycle).
        if isinstance(damping_model, int):
            if damping_model not in _DAMPING_MODEL_MAP.values():
                raise ValueError(
                    f"Unknown damping_model code: {damping_model} "
                    f"(expected one of {sorted(_DAMPING_MODEL_MAP.values())})"
                )
            model_name = {v: k for k, v in _DAMPING_MODEL_MAP.items()}[damping_model]
        else:
            if damping_model not in _DAMPING_MODEL_MAP:
                raise ValueError(
                    f"Unknown damping_model: {damping_model!r} "
                    f"(expected one of {sorted(_DAMPING_MODEL_MAP)})"
                )
            model_name = damping_model

        super().__init__(
            *args, **kwargs,
            allowed_kw_inputs=(
                "E", "L", "SinSq2Theta13", "SinSq2Theta12",
                leading_mass_splitting_3l_name,
                "DeltaMSq21", "nmo",
                "surprobArgConversion",
                "alpha_decay", "alpha_wpd1", "alpha_wpd2",
            ),
        )
        self.needs_post_allocate = True
        self._is_dm32_leading = leading_mass_splitting_3l_name == "DeltaMSq32"
        self._labels.setdefault("mark", "P(ee)")
        self._model_name = model_name

        # ---- Register variants in _evaluator._functions_dict ----
        self._evaluator._functions_dict.update({
            "none":  self._function_none,
            "decay": self._function_decay,
            "wpd1":  self._function_wpd1,
            "wpd2":  self._function_wpd2,
        })

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
    # Invariants (shared across variants)
    # ------------------------------------------------------------------
    def _compute_invariants(self) -> None:
        L_km = self._L[0] * self._baseline_scale
        self._L_m = L_km * 1000.0
        self._L2 = L_km * L_km

        Dm21 = self._DeltaMSq21[0]
        if self._is_dm32_leading:
            Dm32 = self._nmo[0] * self._DeltaMSq3lAbs[0]
            Dm31 = Dm32 + Dm21
        else:
            Dm31 = self._nmo[0] * self._DeltaMSq3lAbs[0]
            Dm32 = Dm31 - Dm21
        self._Dm21, self._Dm31, self._Dm32 = Dm21, Dm31, Dm32

        s2t12 = self._SinSq2Theta12[0]
        s2t13 = self._SinSq2Theta13[0]
        sT12 = 0.5 * (1.0 - sqrt(1.0 - s2t12))
        cT12 = 1.0 - sT12
        sT13 = 0.5 * (1.0 - sqrt(1.0 - s2t13))
        cT13 = 1.0 - sT13
        c4T13 = cT13 * cT13
        self._s4T13 = sT13 * sT13

        self._norm  = 1.0 - 0.5 * (c4T13 * s2t12 + s2t13)
        self._amp21 = 0.5 * c4T13 * s2t12
        self._amp31 = 0.5 * s2t13 * cT12
        self._amp32 = 0.5 * s2t13 * sT12

        self._sinCommonArg = (
            self._surprob_arg_conversion_factor[0]
            * self._L[0] * 0.25 * self._baseline_scale
        )

    # ------------------------------------------------------------------
    # One function as per model 
    # ------------------------------------------------------------------
    def _function_none(self) -> None:
        for callback in self._evaluator._input_nodes_callbacks:
            callback()
        self._compute_invariants()
        _sur_prob_none(
            self._result, self._E, self._sinCommonArg,
            self._Dm21, self._Dm31, self._Dm32,
            self._norm, self._amp21, self._amp31, self._amp32,
        )

    def _function_decay(self) -> None:
        for callback in self._evaluator._input_nodes_callbacks:
            callback()
        self._compute_invariants()
        aL_m = self._alpha_decay[0] * self._L_m
        _sur_prob_decay(
            self._result, self._E, self._sinCommonArg,
            self._Dm21, self._Dm31, self._Dm32,
            self._norm, self._amp21, self._amp31, self._amp32,
            aL_m, self._s4T13,
        )

    def _function_wpd1(self) -> None:
        for callback in self._evaluator._input_nodes_callbacks:
            callback()
        self._compute_invariants()
        a = self._alpha_wpd1[0]
        _sur_prob_wpd1(
            self._result, self._E, self._sinCommonArg,
            self._Dm21, self._Dm31, self._Dm32,
            self._norm, self._amp21, self._amp31, self._amp32,
            a * self._Dm21 * self._Dm21 * self._L2,
            a * self._Dm31 * self._Dm31 * self._L2,
            a * self._Dm32 * self._Dm32 * self._L2,
        )

    def _function_wpd2(self) -> None:
        for callback in self._evaluator._input_nodes_callbacks:
            callback()
        self._compute_invariants()
        a = self._alpha_wpd2[0]
        _sur_prob_wpd2(
            self._result, self._E, self._sinCommonArg,
            self._Dm21, self._Dm31, self._Dm32,
            self._norm, self._amp21, self._amp31, self._amp32,
            a * self._Dm21 * self._Dm21 * self._L2,
            a * self._Dm31 * self._Dm31 * self._L2,
            a * self._Dm32 * self._Dm32 * self._L2,
        )

    # ----------------------------------------
    #    Selection of damping function
    # ----------------------------------------
    @override
    def _type_function(self) -> None:
        check_shape_of_inputs(
            self,
            ("L", "SinSq2Theta12", "SinSq2Theta13",
             "DeltaMSq21",
             "DeltaMSq32" if self._is_dm32_leading else "DeltaMSq31",
             "nmo"),
            (1,),
        )
        for opt_name in ("surprobArgConversion",
                         "alpha_decay", "alpha_wpd1", "alpha_wpd2"):
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
            cin._data if (cin := self.inputs.get("surprobArgConversion"))
            else array([_surprobArgConversion])
        )
        self._alpha_decay = (
            cin._data if (cin := self.inputs.get("alpha_decay")) else array([0.0])
        )
        self._alpha_wpd1 = (
            cin._data if (cin := self.inputs.get("alpha_wpd1")) else array([0.0])
        )
        self._alpha_wpd2 = (
            cin._data if (cin := self.inputs.get("alpha_wpd2")) else array([0.0])
        )

    @classmethod
    def replicate(cls, *args, name: str,
                  replicate_outputs: tuple[KeyLike, ...] = ((),),
                  surprobArgConversion: Output | Literal[True] | None = None,
                  verbose: bool = False, **kwargs):
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