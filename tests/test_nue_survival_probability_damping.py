"""AI-assisted revision.
"""

import numpy as np
import pytest
from dag_modelling.core.exception import TypeFunctionError
from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array

from dgm_reactor_neutrino.nue_survival_probability_damping import (
    NueSurvivalProbabilityDamping,
    _surprobArgConversion,
)

MODELS = ("none", "decay", "wpd1", "wpd2")
DAMPED_MODELS = ("decay", "wpd1", "wpd2")
# Coefficients produce visible damping at a JUNO-like baseline.
ALPHA_BY_MODEL = {
    "decay": 1.0e-4,  # MeV/m
    "wpd1": 1.0e4,  # MeV^4 / (eV^4 km^2)
    "wpd2": 1.0e3,  # MeV^2 / (eV^4 km^2)
}
ENERGY_MEV = np.linspace(1.8, 8.0, 96)
ABSOLUTE_TOLERANCE = 2.0e-13


def _default_inputs(leading_splitting="DeltaMSq32", mass_ordering_sign=1.0):
    """Keys are public port names; scalars always have shape (1,)."""
    return {
        "E": ENERGY_MEV.copy(),
        "L": np.array([52.5]),  # km unless the node declares distance_unit="m"
        "nmo": np.array([mass_ordering_sign]),
        "DeltaMSq21": np.array([7.53e-5]),  # eV^2
        leading_splitting: np.array([2.453e-3]),  # positive magnitude, eV^2
        "SinSq2Theta12": np.array([0.851]),  # sin^2(2 theta_12)
        "SinSq2Theta13": np.array([0.085]),  # sin^2(2 theta_13)
    }


def _connect_inputs(node, input_values):
    """Call only during graph construction; do not read .data here."""
    sources = {}
    for port_name, values in input_values.items():
        source = Array(port_name, values, mode="fill")
        source >> node(port_name)
        sources[port_name] = source
    return sources


def _build_node(model, input_values, *, leading_splitting="DeltaMSq32", distance_unit="km"):
    """Return a closed graph's node and its mutable input sources."""
    with Graph(close_on_exit=True):
        node = NueSurvivalProbabilityDamping(
            "survival_probability",
            damping_model=model,
            leading_mass_splitting_3l_name=leading_splitting,
            distance_unit=distance_unit,
        )
        sources = _connect_inputs(node, input_values)
    return node, sources


def _reference_probability(model, input_values, leading_splitting="DeltaMSq32", distance_unit="km"):
    """Independent NumPy reference from the three mass-state probabilities."""
    energy_mev = input_values["E"]
    baseline_km = input_values["L"][0] * (1.0 if distance_unit == "km" else 1.0e-3)
    theta_12 = 0.5 * np.arcsin(np.sqrt(input_values["SinSq2Theta12"][0]))
    theta_13 = 0.5 * np.arcsin(np.sqrt(input_values["SinSq2Theta13"][0]))
    electron_weights = (
        np.cos(theta_13) ** 2 * np.cos(theta_12) ** 2,
        np.cos(theta_13) ** 2 * np.sin(theta_12) ** 2,
        np.sin(theta_13) ** 2,
    )
    solar_splitting = input_values["DeltaMSq21"][0]
    signed_leading_splitting = input_values["nmo"][0] * input_values[leading_splitting][0]
    splitting_31 = (
        signed_leading_splitting + solar_splitting
        if leading_splitting == "DeltaMSq32"
        else signed_leading_splitting
    )
    # An arbitrary common mass-squared offset cancels from all interference terms.
    mass_squared_values = (0.0, solar_splitting, splitting_31)
    conversion = input_values.get("surprobArgConversion", [_surprobArgConversion])[0]
    alpha_damping = input_values.get("alpha_damping", [0.0])[0]
    if model in ("none", "decay"):
        amplitude = np.zeros_like(energy_mev, dtype=complex)
        for mass_index, (weight, mass_squared) in enumerate(
            zip(electron_weights, mass_squared_values)
        ):
            survival_amplitude = 1.0
            if model == "decay" and mass_index == 2:
                survival_amplitude = np.exp(-alpha_damping * baseline_km * 1000.0 / (2.0 * energy_mev))
            phase = conversion * mass_squared * baseline_km / (2.0 * energy_mev)
            amplitude += weight * survival_amplitude * np.exp(-1j * phase)
        return np.abs(amplitude) ** 2

    probability = np.full_like(energy_mev, sum(weight**2 for weight in electron_weights))
    energy_power = {"wpd1": 4, "wpd2": 2}[model]
    for lower_index, upper_index in ((0, 1), (0, 2), (1, 2)):
        splitting = mass_squared_values[upper_index] - mass_squared_values[lower_index]
        damping = np.exp(-alpha_damping * splitting**2 * baseline_km**2 / energy_mev**energy_power)
        phase = conversion * splitting * baseline_km / (2.0 * energy_mev)
        probability += (
            2.0
            * electron_weights[lower_index]
            * electron_weights[upper_index]
            * damping
            * np.cos(phase)
        )
    return probability


def _assert_probability_and_cache_state(node, expected_probability):
    """Check numerical agreement, clean state, and reuse of the output buffer."""
    assert node.tainted
    result = node.outputs["result"].data
    np.testing.assert_allclose(result, expected_probability, rtol=0, atol=ABSOLUTE_TOLERANCE)
    assert not node.tainted
    assert node.outputs["result"].data is result
    assert not node.tainted


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("leading_splitting", ("DeltaMSq31", "DeltaMSq32"))
@pytest.mark.parametrize("mass_ordering_sign", (1.0, -1.0))
def test_probability_matches_independent_reference(model, leading_splitting, mass_ordering_sign):
    input_values = _default_inputs(leading_splitting, mass_ordering_sign)
    if model != "none":
        input_values["alpha_damping"] = np.array([ALPHA_BY_MODEL[model]])
    node, _ = _build_node(model, input_values, leading_splitting=leading_splitting)
    # Closing selects the function; reading .data below triggers numerical evaluation.
    assert node._evaluator.function.__name__ == f"_function_{model}"
    _assert_probability_and_cache_state(
        node, _reference_probability(model, input_values, leading_splitting)
    )


@pytest.mark.parametrize("model", MODELS)
def test_changed_inputs_invalidate_cached_result(model):
    input_values = _default_inputs()
    input_values["surprobArgConversion"] = np.array([_surprobArgConversion])
    if model != "none":
        input_values["alpha_damping"] = np.array([ALPHA_BY_MODEL[model]])
    node, sources = _build_node(model, input_values)
    _assert_probability_and_cache_state(node, _reference_probability(model, input_values))
    # One graph per model; mutate each dependency instead of rebuilding graphs.
    updates = {
        "E": ENERGY_MEV * 1.03,
        "L": np.array([54.0]),
        "nmo": np.array([-1.0]),
        "DeltaMSq21": np.array([7.8e-5]),
        "DeltaMSq32": np.array([2.5e-3]),
        "SinSq2Theta12": np.array([0.83]),
        "SinSq2Theta13": np.array([0.09]),
        "surprobArgConversion": np.array([_surprobArgConversion * 1.01]),
    }
    if model != "none":
        updates["alpha_damping"] = np.array([ALPHA_BY_MODEL[model] * 2.0])
    for port_name, values in updates.items():
        input_values[port_name] = values
        sources[port_name].outputs[0].set(values)
        _assert_probability_and_cache_state(node, _reference_probability(model, input_values))


@pytest.fixture(scope="module")
def standard_probability():
    """Calculate the standard oscillation baseline once for all zero-alpha cases."""
    node, _ = _build_node("none", _default_inputs())
    return node.outputs["result"].data.copy()


@pytest.mark.parametrize("model", DAMPED_MODELS)
@pytest.mark.parametrize("explicit_zero", (False, True), ids=("omitted_alpha", "zero_alpha"))
def test_zero_damping_matches_standard_oscillations(model, explicit_zero, standard_probability):
    input_values = _default_inputs()
    if explicit_zero:
        input_values["alpha_damping"] = np.array([0.0])
    node, _ = _build_node(model, input_values)
    _assert_probability_and_cache_state(node, standard_probability)


@pytest.mark.parametrize("model", MODELS)
def test_metre_and_kilometre_baselines_are_equivalent(model):
    input_values = _default_inputs()
    if model != "none":
        input_values["alpha_damping"] = np.array([ALPHA_BY_MODEL[model]])
    expected = _reference_probability(model, input_values)
    input_values["L"] = np.array([52500.0])
    node, _ = _build_node(model, input_values, distance_unit="m")
    _assert_probability_and_cache_state(node, expected)


@pytest.mark.parametrize("invalid_model", ("unknown", "wpd3", 0, 1, -1, None, []))
def test_model_selector_accepts_only_supported_strings(invalid_model):
    with pytest.raises(ValueError, match="Unknown damping_model"):
        NueSurvivalProbabilityDamping(
            "survival_probability",
            leading_mass_splitting_3l_name="DeltaMSq32",
            damping_model=invalid_model,
        )


def test_integer_energy_is_rejected_before_allocation():
    input_values = _default_inputs()
    input_values["E"] = np.array([2, 4, 6])
    with pytest.raises(TypeFunctionError):
        _build_node("none", input_values)


@pytest.mark.parametrize("model", MODELS)
def test_multidimensional_energy_preserves_shape(model):
    input_values = _default_inputs()
    input_values["E"] = ENERGY_MEV.reshape(8, 12)
    if model != "none":
        input_values["alpha_damping"] = np.array([ALPHA_BY_MODEL[model]])
    node, _ = _build_node(model, input_values)
    _assert_probability_and_cache_state(node, _reference_probability(model, input_values))
    assert node.outputs["result"].data.shape == (8, 12)


@pytest.mark.parametrize("replica_key", ((), "AD11", ("EH1", "AD11")))
@pytest.mark.parametrize("model", ("none", "wpd1"))
def test_replication_keys_and_model_specific_alpha_port(replica_key, model):
    key_parts = (replica_key,) if isinstance(replica_key, str) else replica_key
    input_values = _default_inputs()
    if model != "none":
        input_values["alpha_damping"] = np.array([ALPHA_BY_MODEL[model]])
    with Graph(close_on_exit=True):
        _, storage = NueSurvivalProbabilityDamping.replicate(
            name="survival",
            replicate_outputs=(replica_key,),
            leading_mass_splitting_3l_name="DeltaMSq32",
            damping_model=model,
        )
        node = storage["nodes"][("survival",) + key_parts]
        assert storage["inputs"][("survival", "enu") + key_parts] is node.inputs["E"]
        alpha_damping_key = ("survival", "alpha_damping") + key_parts
        assert (alpha_damping_key in storage["inputs"]) == (model != "none")
        _connect_inputs(node, input_values)
    _assert_probability_and_cache_state(node, _reference_probability(model, input_values))
