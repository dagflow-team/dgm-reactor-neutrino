#!/usr/bin/env python

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.plot.plot import plot_auto
from matplotlib.pyplot import subplots
from numpy import allclose, arcsin, cos, exp, expm1, finfo, geomspace, sin, sqrt
from pytest import mark, raises

from dgm_reactor_neutrino import NueSurvivalProbabilityDamping
from dgm_reactor_neutrino.nue_survival_probability_damping import (
    _surprobArgConversion,
    _DAMPING_MODEL_MAP,
)

_DAMPING_CASES = ("none", "decay", "wpd1", "wpd2")
_DAMPING_CASES_INT = tuple(sorted(_DAMPING_MODEL_MAP.values()))   # (0, 1, 2, 3)

_ATOL = 10.0 * finfo("d").resolution


def _assert_close(surprob, reference):
    assert surprob.tainted is True
    assert allclose(surprob.outputs[0].data, reference(), rtol=0, atol=_ATOL)
    assert surprob.tainted is False


# ---------------------------------------------------------------------------
# The physics test — one parametrization per model, per L, per NMO.
# This is what actually exercises the four kernels end-to-end.
# ---------------------------------------------------------------------------
@mark.parametrize("nmo", (1, -1))
@mark.parametrize(
    "L,leading_mass_splitting_3l_name",
    ((2, "DeltaMSq32"), (52, "DeltaMSq31"), (180, "DeltaMSq32")),
)
@mark.parametrize("damping_model", _DAMPING_CASES)
def test_NueSurvivalProbability_01(
    debug_graph, test_name, L,
    leading_mass_splitting_3l_name, nmo,
    damping_model, output_path: str,
):
    E = geomspace(1, 100, 1000)  # MeV
    DeltaMSq21 = 7.39 * 1e-5       # eV^2
    DeltaMSq3lAbs = 2.45 * 1e-3    # eV^2
    SinSq2Theta12 = 3.1e-1
    SinSq2Theta13 = 2.241e-2
    is_dm32_leading = leading_mass_splitting_3l_name == "DeltaMSq32"
    is_dm31_leading = leading_mass_splitting_3l_name == "DeltaMSq31"

    damping_params = {
        "alpha_decay": 1e-5,   # MeV/m
        "alpha_wpd1":  1e-3,   # MeV^2
        "alpha_wpd2":  1e-5,   # dimensionless
    }

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        # damping_model is a constructor argument (Medium+), resolved in
        # __init__ and selected in _type_function (Chi2 pattern).
        surprob = NueSurvivalProbabilityDamping(
            "P(ee)",
            leading_mass_splitting_3l_name=leading_mass_splitting_3l_name,
            damping_model=damping_model,
        )
        (in_E := Array("E", E, mode="fill")) >> surprob("E")
        (in_L := Array("L", [L], mode="fill")) >> surprob("L")
        (in_nmo := Array("nmo", [nmo], mode="fill")) >> surprob("nmo")
        (in_Dm21 := Array("DeltaMSq21", [DeltaMSq21], mode="fill")) >> surprob("DeltaMSq21")
        (in_Dm3l := Array(leading_mass_splitting_3l_name, [DeltaMSq3lAbs], mode="fill")) >> surprob(
            leading_mass_splitting_3l_name
        )
        (in_t12 := Array("SinSq2Theta12", [SinSq2Theta12], mode="fill")) >> surprob("SinSq2Theta12")
        (in_t13 := Array("SinSq2Theta13", [SinSq2Theta13], mode="fill")) >> surprob("SinSq2Theta13")

        (in_ad := Array("alpha_decay", [damping_params["alpha_decay"]], mode="fill")) >> surprob(
            "alpha_decay"
        )
        (in_aw1 := Array("alpha_wpd1", [damping_params["alpha_wpd1"]], mode="fill")) >> surprob(
            "alpha_wpd1"
        )
        (in_aw2 := Array("alpha_wpd2", [damping_params["alpha_wpd2"]], mode="fill")) >> surprob(
            "alpha_wpd2"
        )

        conversionFactor = _surprobArgConversion
        (in_conv := Array(
            "surprobArgConversion", [conversionFactor], mode="fill"
        )) >> surprob("surprobArgConversion")

    def surprob_fcn():
        tmp = L * conversionFactor / 4.0 / E

        _DeltaMSq31 = nmo * DeltaMSq3lAbs + DeltaMSq21 * is_dm32_leading
        _DeltaMSq32 = nmo * DeltaMSq3lAbs - DeltaMSq21 * is_dm31_leading

        theta12 = 0.5 * arcsin(sqrt(SinSq2Theta12))
        theta13 = 0.5 * arcsin(sqrt(SinSq2Theta13))

        _SinSqTheta12 = sin(theta12) ** 2
        _CosSqTheta12 = cos(theta12) ** 2
        _CosQuTheta13 = (cos(theta13) ** 2) ** 2
        _SinQuTheta13 = (sin(theta13) ** 2) ** 2

        D21, D31, D32 = 1.0, 1.0, 1.0
        correction = 0.0

        if damping_model == "decay":
            L_m = L * 1000.0
            a = damping_params["alpha_decay"]
            D31 = exp(-a * L_m / (2.0 * E))
            D32 = D31
            correction = _SinQuTheta13 * expm1(-a * L_m / E)
        elif damping_model == "wpd1":
            a = damping_params["alpha_wpd1"]
            D21 = exp(-a * DeltaMSq21**2 * L**2 / E**4)
            D31 = exp(-a * _DeltaMSq31**2 * L**2 / E**4)
            D32 = exp(-a * _DeltaMSq32**2 * L**2 / E**4)
        elif damping_model == "wpd2":
            a = damping_params["alpha_wpd2"]
            D21 = exp(-a * DeltaMSq21**2 * L**2 / E**2)
            D31 = exp(-a * _DeltaMSq31**2 * L**2 / E**2)
            D32 = exp(-a * _DeltaMSq32**2 * L**2 / E**2)

        return (
            1.0
            - 0.5 * (_CosQuTheta13 * SinSq2Theta12 + SinSq2Theta13)
            + 0.5 * _CosQuTheta13 * SinSq2Theta12 * D21 * cos(2.0 * DeltaMSq21 * tmp)
            + 0.5 * SinSq2Theta13
            * (
                D31 * _CosSqTheta12 * cos(2.0 * _DeltaMSq31 * tmp)
                + D32 * _SinSqTheta12 * cos(2.0 * _DeltaMSq32 * tmp)
            )
            + correction
        )

    _assert_close(surprob, surprob_fcn)

    subplots(1, 1)
    plot_auto(
        surprob, filter_kw={"masked_value": 0},
        show=False, close=True,
        save=f"{output_path}/{test_name}_plot.pdf",
    )

    # ---- oscillation-parameter variations
    nmo *= -1; in_nmo.outputs[0].set(nmo)
    _assert_close(surprob, surprob_fcn)

    DeltaMSq21 *= 1.1; in_Dm21.outputs[0].set(DeltaMSq21)
    _assert_close(surprob, surprob_fcn)

    DeltaMSq3lAbs *= 0.9; in_Dm3l.outputs[0].set(DeltaMSq3lAbs)
    _assert_close(surprob, surprob_fcn)

    SinSq2Theta12 *= 1.2; in_t12.outputs[0].set(SinSq2Theta12)
    _assert_close(surprob, surprob_fcn)

    SinSq2Theta13 += 0.1; in_t13.outputs[0].set(SinSq2Theta13)
    _assert_close(surprob, surprob_fcn)

    L *= 10; in_L.outputs[0].set(L)
    _assert_close(surprob, surprob_fcn)

    E *= 15; in_E.outputs[0].set(E)
    _assert_close(surprob, surprob_fcn)

    # ---- conversion-factor variation
    conversionFactor *= 1.01; in_conv.outputs[0].set(conversionFactor)
    _assert_close(surprob, surprob_fcn)

    # ---- damping-parameter variation
    _alpha_input = {
        "decay": (in_ad, "alpha_decay"),
        "wpd1":  (in_aw1, "alpha_wpd1"),
        "wpd2":  (in_aw2, "alpha_wpd2"),
        "none":  (None, None),
    }[damping_model]
    if _alpha_input[0] is not None:
        in_alpha, alpha_key = _alpha_input
        damping_params[alpha_key] *= 3.0
        in_alpha.outputs[0].set(damping_params[alpha_key])
        _assert_close(surprob, surprob_fcn)

    savegraph(graph, f"{output_path}/{test_name}.png")


# ---------------------------------------------------------------------------
# Chi2-dispatch check: after _type_function has run, the evaluator
# must point to the model-specific _function_X.  This catches the class
# of bug where the registry is populated but the wrong key is selected.
# ---------------------------------------------------------------------------
@mark.parametrize(
    "damping_model, expected_fn_name",
    (
        ("none",  "_function_none"),
        ("decay", "_function_decay"),
        ("wpd1",  "_function_wpd1"),
        ("wpd2",  "_function_wpd2"),
        (0,       "_function_none"),   # integer codes must also dispatch correctly
        (1,       "_function_decay"),
        (2,       "_function_wpd1"),
        (3,       "_function_wpd2"),
    ),
)
def test_NueSurvivalProbability_dispatch(damping_model, expected_fn_name):
    """_type_function must bind the model-specific _function_X (Chi2 pattern)."""
    E = geomspace(1, 100, 10)
    with Graph(close_on_exit=True) as graph:
        surprob = NueSurvivalProbabilityDamping(
            "P(ee)",
            leading_mass_splitting_3l_name="DeltaMSq32",
            damping_model=damping_model,
        )
        Array("E", E, mode="fill") >> surprob("E")
        Array("L", [52.0], mode="fill") >> surprob("L")
        Array("nmo", [1.0], mode="fill") >> surprob("nmo")
        Array("DeltaMSq21", [7.39e-5], mode="fill") >> surprob("DeltaMSq21")
        Array("DeltaMSq32", [2.45e-3], mode="fill") >> surprob("DeltaMSq32")
        Array("SinSq2Theta12", [0.31], mode="fill") >> surprob("SinSq2Theta12")
        Array("SinSq2Theta13", [0.02241], mode="fill") >> surprob("SinSq2Theta13")

        # Accessing data triggers _type_function + the selected _function_X.
        _ = surprob.outputs[0].data

        # Chi2-style assertion: the evaluator was bound to the right variant.
        assert surprob._evaluator.function.__name__ == expected_fn_name, (
            f"damping_model={damping_model!r} dispatched to "
            f"{surprob._evaluator.function.__name__!r}, expected {expected_fn_name!r}"
        )


# ---------------------------------------------------------------------------
# Integer codes and string names must give identical results.
# ---------------------------------------------------------------------------
@mark.parametrize("name, code", tuple(_DAMPING_MODEL_MAP.items()))
def test_NueSurvivalProbability_string_and_int_equivalent(name, code):
    E = geomspace(1, 100, 100)

    def build(model):
        n = NueSurvivalProbabilityDamping(
            "P(ee)",
            leading_mass_splitting_3l_name="DeltaMSq32",
            damping_model=model,
        )
        Array("E", E, mode="fill") >> n("E")
        Array("L", [52.0], mode="fill") >> n("L")
        Array("nmo", [1.0], mode="fill") >> n("nmo")
        Array("DeltaMSq21", [7.39e-5], mode="fill") >> n("DeltaMSq21")
        Array("DeltaMSq32", [2.45e-3], mode="fill") >> n("DeltaMSq32")
        Array("SinSq2Theta12", [0.31], mode="fill") >> n("SinSq2Theta12")
        Array("SinSq2Theta13", [0.02241], mode="fill") >> n("SinSq2Theta13")
        Array("alpha_decay", [1e-5], mode="fill") >> n("alpha_decay")
        Array("alpha_wpd1",  [1e-3], mode="fill") >> n("alpha_wpd1")
        Array("alpha_wpd2",  [1e-5], mode="fill") >> n("alpha_wpd2")
        Array("surprobArgConversion", [_surprobArgConversion], mode="fill") >> n("surprobArgConversion")
        return n

    with Graph(close_on_exit=True):
        p_name = build(name)
        p_code = build(code)
        assert allclose(p_name.outputs[0].data, p_code.outputs[0].data,
                        rtol=0, atol=_ATOL)


# ---------------------------------------------------------------------------
# Constructor must reject unknown / non-integer damping-model codes.
# ---------------------------------------------------------------------------
@mark.parametrize("bad_model", ("unknown", "wpd3", 4, -1, 99))
def test_NueSurvivalProbability_invalid_model(bad_model):
    with raises(ValueError):
        NueSurvivalProbabilityDamping(
            "P(ee)",
            leading_mass_splitting_3l_name="DeltaMSq32",
            damping_model=bad_model,
        )


# ---------------------------------------------------------------------------
# With α = 0 all damping models reduce to the no-damping result.
# ---------------------------------------------------------------------------
@mark.parametrize("damping_model", ("decay", "wpd1", "wpd2"))
def test_NueSurvivalProbability_zero_alpha(debug_graph, damping_model):
    E = geomspace(1, 100, 100)
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        def build(model_name):
            n = NueSurvivalProbabilityDamping(
                "P(ee)",
                leading_mass_splitting_3l_name="DeltaMSq32",
                damping_model=model_name,
            )
            (Array("E", E, mode="fill") >> n("E"))
            (Array("L", [52.0], mode="fill") >> n("L"))
            (Array("nmo", [1.0], mode="fill") >> n("nmo"))
            (Array("DeltaMSq21", [7.39e-5], mode="fill") >> n("DeltaMSq21"))
            (Array("DeltaMSq32", [2.45e-3], mode="fill") >> n("DeltaMSq32"))
            (Array("SinSq2Theta12", [0.31], mode="fill") >> n("SinSq2Theta12"))
            (Array("SinSq2Theta13", [0.02241], mode="fill") >> n("SinSq2Theta13"))
            (Array("alpha_decay", [0.0], mode="fill") >> n("alpha_decay"))
            (Array("alpha_wpd1",  [0.0], mode="fill") >> n("alpha_wpd1"))
            (Array("alpha_wpd2",  [0.0], mode="fill") >> n("alpha_wpd2"))
            (Array("surprobArgConversion", [_surprobArgConversion], mode="fill")
             >> n("surprobArgConversion"))
            return n

        p_none = build("none")
        p_test = build(damping_model)
        assert allclose(p_none.outputs[0].data, p_test.outputs[0].data,
                        rtol=0, atol=_ATOL)