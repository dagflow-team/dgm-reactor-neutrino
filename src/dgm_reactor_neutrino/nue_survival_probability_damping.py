"""AI-assisted revision"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from dag_modelling.core.global_parameters import NUMBA_CACHE_ENABLE
from dag_modelling.core.node import Node
from dag_modelling.core.storage import NodeStorage
from dag_modelling.core.type_functions import (
    assign_axes_from_inputs_to_outputs,
    check_shape_of_inputs,
    check_subtype_of_inputs,
    copy_from_inputs_to_outputs,
)
from numba import njit
from numpy import array, cos, exp, expm1, floating, pi, sqrt
from scipy.constants import value

if TYPE_CHECKING:
    from typing import Literal

    from dag_modelling.core.output import Output
    from nested_mapping.typing import KeyLike
    from numpy import double
    from numpy.typing import NDArray

# Cosine phase = K * DeltaMSq[eV^2] * L[km] / (2 * E[MeV]).
_surprobArgConversion = pi * 2e-3 * value("electron volt-inverse meter relationship")

_DAMPING_MODELS = {"none", "decay", "wpd1", "wpd2"}


@njit(cache=NUMBA_CACHE_ENABLE)
def _get_common_amplitudes(sin_squared_double_theta_12, sin_squared_double_theta_13):
    sin_squared_theta_12 = 0.5 * (1 - sqrt(1 - sin_squared_double_theta_12))  # sin²θ₁₂
    cos_squared_theta_12 = 1.0 - sin_squared_theta_12  # cos²θ₁₂
    sin_squared_theta_13 = 0.5 * (1 - sqrt(1 - sin_squared_double_theta_13))  # sin²θ₁₃
    cos_squared_theta_13 = 1.0 - sin_squared_theta_13  # cos²θ₁₃
    cos_fourth_theta_13 = cos_squared_theta_13 * cos_squared_theta_13  # cos⁴θ₁₃
    sin_fourth_theta_13 = sin_squared_theta_13 * sin_squared_theta_13  # sin⁴θ₁₃

    diagonal_probability = 1.0 - 0.5 * (
        cos_fourth_theta_13 * sin_squared_double_theta_12 + sin_squared_double_theta_13
    )
    interference_amplitude_21 = 0.5 * cos_fourth_theta_13 * sin_squared_double_theta_12
    interference_amplitude_31 = 0.5 * sin_squared_double_theta_13 * cos_squared_theta_12
    interference_amplitude_32 = 0.5 * sin_squared_double_theta_13 * sin_squared_theta_12

    return (
        diagonal_probability,
        interference_amplitude_21,
        interference_amplitude_31,
        interference_amplitude_32,
        sin_fourth_theta_13,
    )


@njit(cache=NUMBA_CACHE_ENABLE)
def _get_mass_splittings(
    mass_squared_difference_21,
    leading_mass_squared_difference_abs,
    is_dm32_leading,
    mass_ordering_sign,
):
    if is_dm32_leading:
        mass_squared_difference_32 = mass_ordering_sign * leading_mass_squared_difference_abs
        mass_squared_difference_31 = mass_squared_difference_32 + mass_squared_difference_21
    else:
        mass_squared_difference_31 = mass_ordering_sign * leading_mass_squared_difference_abs
        mass_squared_difference_32 = mass_squared_difference_31 - mass_squared_difference_21

    return mass_squared_difference_31, mass_squared_difference_32


# ---------------------------------------------------------------------------
# One numba kernel per model.
# Each kernel only receives the parameters that model uses.
# ---------------------------------------------------------------------------
@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_none(
    probability: NDArray[double],
    energy_mev: NDArray[double],
    baseline: NDArray[double],
    baseline_scale_to_km: float,
    sin_squared_double_theta_12: NDArray[double],
    sin_squared_double_theta_13: NDArray[double],
    mass_squared_difference_21: NDArray[double],
    leading_mass_squared_difference_abs: NDArray[double],
    is_dm32_leading,
    mass_ordering_sign: NDArray[double],
    phase_conversion: NDArray[double],
) -> None:
    (
        diagonal_probability,
        interference_amplitude_21,
        interference_amplitude_31,
        interference_amplitude_32,
        _,
    ) = _get_common_amplitudes(sin_squared_double_theta_12[0], sin_squared_double_theta_13[0])
    mass_squared_difference_21_value = mass_squared_difference_21[0]
    mass_squared_difference_31, mass_squared_difference_32 = _get_mass_splittings(
        mass_squared_difference_21_value,
        leading_mass_squared_difference_abs[0],
        is_dm32_leading,
        mass_ordering_sign[0],
    )
    phase_numerator = phase_conversion[0] * baseline[0] * 0.5 * baseline_scale_to_km
    for i in range(len(probability)):
        phase_per_mass_squared_difference = phase_numerator / energy_mev[i]
        probability[i] = (
            diagonal_probability
            + interference_amplitude_21
            * cos(mass_squared_difference_21_value * phase_per_mass_squared_difference)
            + interference_amplitude_31
            * cos(mass_squared_difference_31 * phase_per_mass_squared_difference)
            + interference_amplitude_32
            * cos(mass_squared_difference_32 * phase_per_mass_squared_difference)
        )


@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_decay(
    probability: NDArray[double],
    energy_mev: NDArray[double],
    baseline: NDArray[double],
    baseline_scale_to_km: float,
    sin_squared_double_theta_12: NDArray[double],
    sin_squared_double_theta_13: NDArray[double],
    mass_squared_difference_21: NDArray[double],
    leading_mass_squared_difference_abs: NDArray[double],
    is_dm32_leading,
    mass_ordering_sign: NDArray[double],
    phase_conversion: NDArray[double],
    alpha_decay: NDArray[double],
) -> None:
    (
        diagonal_probability,
        interference_amplitude_21,
        interference_amplitude_31,
        interference_amplitude_32,
        sin_fourth_theta_13,
    ) = _get_common_amplitudes(sin_squared_double_theta_12[0], sin_squared_double_theta_13[0])
    mass_squared_difference_21_value = mass_squared_difference_21[0]
    mass_squared_difference_31, mass_squared_difference_32 = _get_mass_splittings(
        mass_squared_difference_21_value,
        leading_mass_squared_difference_abs[0],
        is_dm32_leading,
        mass_ordering_sign[0],
    )
    baseline_km = baseline[0] * baseline_scale_to_km
    phase_numerator = phase_conversion[0] * 0.5 * baseline_km
    decay_numerator = alpha_decay[0] * baseline_km * 1000.0
    for i in range(len(probability)):
        energy_value = energy_mev[i]
        phase_per_mass_squared_difference = phase_numerator / energy_value
        decay_exponent = decay_numerator / energy_value
        interference_damping = exp(-0.5 * decay_exponent)
        decay_diagonal_correction = sin_fourth_theta_13 * expm1(-decay_exponent)
        probability[i] = (
            diagonal_probability
            + interference_amplitude_21
            * cos(mass_squared_difference_21_value * phase_per_mass_squared_difference)
            + interference_amplitude_31
            * interference_damping
            * cos(mass_squared_difference_31 * phase_per_mass_squared_difference)
            + interference_amplitude_32
            * interference_damping
            * cos(mass_squared_difference_32 * phase_per_mass_squared_difference)
            + decay_diagonal_correction
        )


@njit(cache=NUMBA_CACHE_ENABLE)
def _get_damping_arguments_wpd(
    alpha_wpd,
    mass_squared_difference_21,
    mass_squared_difference_31,
    mass_squared_difference_32,
    baseline,
):
    baseline_squared = baseline * baseline
    damping_arg_21 = (
        alpha_wpd * mass_squared_difference_21 * mass_squared_difference_21 * baseline_squared
    )
    damping_arg_31 = (
        alpha_wpd * mass_squared_difference_31 * mass_squared_difference_31 * baseline_squared
    )
    damping_arg_32 = (
        alpha_wpd * mass_squared_difference_32 * mass_squared_difference_32 * baseline_squared
    )

    return damping_arg_21, damping_arg_31, damping_arg_32


@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_wpd1(
    probability: NDArray[double],
    energy_mev: NDArray[double],
    baseline: NDArray[double],
    baseline_scale_to_km: float,
    sin_squared_double_theta_12: NDArray[double],
    sin_squared_double_theta_13: NDArray[double],
    mass_squared_difference_21: NDArray[double],
    leading_mass_squared_difference_abs: NDArray[double],
    is_dm32_leading,
    mass_ordering_sign: NDArray[double],
    phase_conversion: NDArray[double],
    alpha_wpd1: NDArray[double],
) -> None:
    (
        diagonal_probability,
        interference_amplitude_21,
        interference_amplitude_31,
        interference_amplitude_32,
        _,
    ) = _get_common_amplitudes(sin_squared_double_theta_12[0], sin_squared_double_theta_13[0])
    mass_squared_difference_21_value = mass_squared_difference_21[0]
    mass_squared_difference_31, mass_squared_difference_32 = _get_mass_splittings(
        mass_squared_difference_21_value,
        leading_mass_squared_difference_abs[0],
        is_dm32_leading,
        mass_ordering_sign[0],
    )
    baseline_km = baseline[0] * baseline_scale_to_km
    phase_numerator = phase_conversion[0] * baseline_km * 0.5
    damping_arg_21, damping_arg_31, damping_arg_32 = _get_damping_arguments_wpd(
        alpha_wpd1[0],
        mass_squared_difference_21_value,
        mass_squared_difference_31,
        mass_squared_difference_32,
        baseline_km,
    )
    for i in range(len(probability)):
        energy_value = energy_mev[i]
        energy_squared = energy_value * energy_value
        inverse_energy_fourth = 1.0 / (energy_squared * energy_squared)
        phase_per_mass_squared_difference = phase_numerator / energy_value
        probability[i] = (
            diagonal_probability
            + interference_amplitude_21
            * exp(-damping_arg_21 * inverse_energy_fourth)
            * cos(mass_squared_difference_21_value * phase_per_mass_squared_difference)
            + interference_amplitude_31
            * exp(-damping_arg_31 * inverse_energy_fourth)
            * cos(mass_squared_difference_31 * phase_per_mass_squared_difference)
            + interference_amplitude_32
            * exp(-damping_arg_32 * inverse_energy_fourth)
            * cos(mass_squared_difference_32 * phase_per_mass_squared_difference)
        )


@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob_wpd2(
    probability: NDArray[double],
    energy_mev: NDArray[double],
    baseline: NDArray[double],
    baseline_scale_to_km: float,
    sin_squared_double_theta_12: NDArray[double],
    sin_squared_double_theta_13: NDArray[double],
    mass_squared_difference_21: NDArray[double],
    leading_mass_squared_difference_abs: NDArray[double],
    is_dm32_leading,
    mass_ordering_sign: NDArray[double],
    phase_conversion: NDArray[double],
    alpha_wpd2: NDArray[double],
) -> None:
    (
        diagonal_probability,
        interference_amplitude_21,
        interference_amplitude_31,
        interference_amplitude_32,
        _,
    ) = _get_common_amplitudes(sin_squared_double_theta_12[0], sin_squared_double_theta_13[0])
    mass_squared_difference_21_value = mass_squared_difference_21[0]
    mass_squared_difference_31, mass_squared_difference_32 = _get_mass_splittings(
        mass_squared_difference_21_value,
        leading_mass_squared_difference_abs[0],
        is_dm32_leading,
        mass_ordering_sign[0],
    )
    baseline_km = baseline[0] * baseline_scale_to_km
    phase_numerator = phase_conversion[0] * baseline_km * 0.5
    damping_arg_21, damping_arg_31, damping_arg_32 = _get_damping_arguments_wpd(
        alpha_wpd2[0],
        mass_squared_difference_21_value,
        mass_squared_difference_31,
        mass_squared_difference_32,
        baseline_km,
    )
    for i in range(len(probability)):
        energy_value = energy_mev[i]
        inverse_energy_squared = 1.0 / (energy_value * energy_value)
        phase_per_mass_squared_difference = phase_numerator / energy_value
        probability[i] = (
            diagonal_probability
            + interference_amplitude_21
            * exp(-damping_arg_21 * inverse_energy_squared)
            * cos(mass_squared_difference_21_value * phase_per_mass_squared_difference)
            + interference_amplitude_31
            * exp(-damping_arg_31 * inverse_energy_squared)
            * cos(mass_squared_difference_31 * phase_per_mass_squared_difference)
            + interference_amplitude_32
            * exp(-damping_arg_32 * inverse_energy_squared)
            * cos(mass_squared_difference_32 * phase_per_mass_squared_difference)
        )


class NueSurvivalProbabilityDamping(Node):
    """Three-flavor electron-antineutrino survival probability in vacuum.

    Select the model by its constructor string: "none" (standard oscillations
    without damping), "decay" (invisible nu_3 decay), "wpd1", or "wpd2".
    Numerical model codes are not supported. Model selection is fixed when the
    graph closes; alpha is a runtime scalar input, not a model selector.

    Public input conventions (unchanged from the senior implementation):
      E: floating energy array in MeV; result has the same shape and dtype.
      L: one-element baseline array, in distance_unit (km by default).
      SinSq2Theta12/13: sin^2(2 theta_12/13), NOT sin^2(theta_12/13).
        Angles are reconstructed on the lower-octant branch.
      DeltaMSq21: positive solar mass-squared difference, in eV^2.
      DeltaMSq31/32: positive magnitude of the selected leading splitting,
        in eV^2. nmo supplies its sign (+1 normal, -1 inverted).
      surprobArgConversion: optional phase factor K; default is
        4 * 1.26693e3 for cosine phase K * DeltaMSq * L_km / (2 * E_MeV).
      alpha: optional one-element damping coefficient, default zero:
        decay: d = alpha * L_m / E_MeV; alpha in MeV/m.
        wpd1: D_ij = exp[-alpha * (DeltaMSq_ij)^2 * L_km^2 / E_MeV^4];
          alpha in MeV^4 / (eV^4 km^2).
        wpd2: same expression with E_MeV^2 in the denominator;
          alpha in MeV^2 / (eV^4 km^2).
        "none" does not use alpha. These are formula coefficient conventions;
        converting microscopic model parameters requires a separate mapping.

    WPD damps interference and preserves mass-state populations; it does not
    preserve the electron-flavor detection rate at each energy. Invisible decay
    also removes the nu_3 population. All inputs other than E have shape (1,).
    Construct and connect nodes inside Graph(close_on_exit=True), then read
    output.data after leaving the context, once buffers have been allocated.
    """

    __slots__ = (
        "_baseline_scale_to_km",
        "_result",
        "_energy_mev",
        "_baseline",
        "_sin_squared_double_theta_12",
        "_sin_squared_double_theta_13",
        "_mass_squared_difference_21",
        "_leading_mass_squared_difference_abs",
        "_is_dm32_leading",
        "_mass_ordering_sign",
        "_phase_conversion",
        "_alpha",
        "_model_name",
    )

    _baseline_scale_to_km: float
    _energy_mev: NDArray
    _baseline: NDArray
    _mass_ordering_sign: NDArray
    _mass_squared_difference_21: NDArray
    _leading_mass_squared_difference_abs: NDArray
    _sin_squared_double_theta_12: NDArray
    _sin_squared_double_theta_13: NDArray
    _phase_conversion: NDArray
    _alpha: NDArray | None
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
        if not isinstance(damping_model, str) or damping_model not in _DAMPING_MODELS:
            raise ValueError(
                f"Unknown damping_model: {damping_model!r} "
                f"(expected one of {sorted(_DAMPING_MODELS)})"
            )

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
        self._model_name = damping_model

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
            self._baseline_scale_to_km = {"km": 1, "m": 1.0e-3}[distance_unit]
        except KeyError as exc:
            raise RuntimeError(f"Invalid distance unit {distance_unit}") from exc

    # ------------------------------------------------------------------
    # Update upstream nodes, then write directly into the allocated output.
    # ------------------------------------------------------------------
    def _function_none(self) -> None:
        for callback in self._evaluator._input_nodes_callbacks:
            callback()
        _sur_prob_none(
            self._result,
            self._energy_mev,
            self._baseline,
            self._baseline_scale_to_km,
            self._sin_squared_double_theta_12,
            self._sin_squared_double_theta_13,
            self._mass_squared_difference_21,
            self._leading_mass_squared_difference_abs,
            self._is_dm32_leading,
            self._mass_ordering_sign,
            self._phase_conversion,
        )

    def _function_decay(self) -> None:
        for callback in self._evaluator._input_nodes_callbacks:
            callback()
        _sur_prob_decay(
            self._result,
            self._energy_mev,
            self._baseline,
            self._baseline_scale_to_km,
            self._sin_squared_double_theta_12,
            self._sin_squared_double_theta_13,
            self._mass_squared_difference_21,
            self._leading_mass_squared_difference_abs,
            self._is_dm32_leading,
            self._mass_ordering_sign,
            self._phase_conversion,
            self._alpha,
        )

    def _function_wpd1(self) -> None:
        for callback in self._evaluator._input_nodes_callbacks:
            callback()
        _sur_prob_wpd1(
            self._result,
            self._energy_mev,
            self._baseline,
            self._baseline_scale_to_km,
            self._sin_squared_double_theta_12,
            self._sin_squared_double_theta_13,
            self._mass_squared_difference_21,
            self._leading_mass_squared_difference_abs,
            self._is_dm32_leading,
            self._mass_ordering_sign,
            self._phase_conversion,
            self._alpha,
        )

    def _function_wpd2(self) -> None:
        for callback in self._evaluator._input_nodes_callbacks:
            callback()
        _sur_prob_wpd2(
            self._result,
            self._energy_mev,
            self._baseline,
            self._baseline_scale_to_km,
            self._sin_squared_double_theta_12,
            self._sin_squared_double_theta_13,
            self._mass_squared_difference_21,
            self._leading_mass_squared_difference_abs,
            self._is_dm32_leading,
            self._mass_ordering_sign,
            self._phase_conversion,
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
        # The output inherits E's dtype: reject integers to prevent truncation.
        check_subtype_of_inputs(self, "E", dtype=floating)
        copy_from_inputs_to_outputs(self, "E", "result")
        assign_axes_from_inputs_to_outputs(
            self, "E", "result", assign_meshes=True, ignore_Nd=True, overwrite_assigned=True
        )

        # ---- Select the variant here ----
        self._evaluator.function = self._evaluator._functions_dict[self._model_name]

    @override
    def _post_allocate(self) -> None:
        super()._post_allocate()
        self._result = self.outputs["result"]._data.ravel()
        self._energy_mev = self.inputs["E"]._data.ravel()
        self._baseline = self.inputs["L"]._data
        self._sin_squared_double_theta_12 = self.inputs["SinSq2Theta12"]._data
        self._sin_squared_double_theta_13 = self.inputs["SinSq2Theta13"]._data
        self._mass_squared_difference_21 = self.inputs["DeltaMSq21"]._data
        self._leading_mass_squared_difference_abs = self.inputs[
            "DeltaMSq32" if self._is_dm32_leading else "DeltaMSq31"
        ]._data
        self._mass_ordering_sign = self.inputs["nmo"]._data

        self._phase_conversion = (
            conversion_input._data
            if (conversion_input := self.inputs.get("surprobArgConversion"))
            else array([_surprobArgConversion])
        )
        # No damping parameter or fallback array is needed for the standard model.
        self._alpha = None
        if self._model_name != "none":
            alpha_input = self.inputs.get("alpha")
            self._alpha = alpha_input._data if alpha_input is not None else array([0.0])

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
        """Register replicas and their E/L ports in NodeStorage.

        Damped replicas expose an alpha port that the caller must connect;
        "none" replicas expose no unused alpha port. With direct construction,
        an omitted alpha input instead defaults to zero. Optional shared phase
        conversion follows the existing Output / True / None convention.
        """
        storage = NodeStorage()
        nodes = storage.create_child("nodes")
        inputs = storage.create_child("inputs")
        outputs = storage.create_child("outputs")

        name_parts = tuple(name.split("."))
        for key in replicate_outputs:
            key_parts = (key,) if isinstance(key, str) else tuple(key)
            node_key = name_parts + key_parts
            node_name = ".".join(node_key)
            survival_probability = cls(node_name, *args, **kwargs)
            nodes[node_key] = survival_probability
            inputs[name_parts + ("enu",) + key_parts] = survival_probability.inputs[0]
            inputs[name_parts + ("L",) + key_parts] = survival_probability.inputs["L"]
            outputs[node_key] = survival_probability.outputs[0]

            if surprobArgConversion:
                if surprobArgConversion is True:
                    inputs[name_parts + ("surprobArgConversion",) + key_parts] = (
                        survival_probability("surprobArgConversion")
                    )
                else:
                    surprobArgConversion >> survival_probability("surprobArgConversion")

            if survival_probability._model_name != "none":
                inputs[name_parts + ("alpha",) + key_parts] = survival_probability("alpha")

        NodeStorage.update_current(storage, strict=True, verbose=verbose)
        return None, storage