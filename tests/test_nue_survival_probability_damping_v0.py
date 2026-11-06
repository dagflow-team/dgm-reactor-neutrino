#!/usr/bin/env python

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.plot.plot import plot_auto
from matplotlib.pyplot import subplots
from numpy import allclose, arcsin, cos, exp, finfo, geomspace, sin, sqrt
from pytest import mark

from dgm_reactor_neutrino import NueSurvivalProbabilityDamping
from dgm_reactor_neutrino.nue_survival_probability_damping import (
    _surprobArgConversion,
    _DAMPING_MODEL_MAP,
)

_DAMPING_CASES = ("none", "decay", "wpd1", "wpd2")


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

    # Damping parameters (mutable dict so both DAG + reference see the same value)
    damping_params = {
        "alpha_decay": 1e-5,   # MeV/m
        "alpha_wpd1":  1e-3,   # MeV^2
        "alpha_wpd2":  1e-5,   # dimensionless
    }

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        surprob = NueSurvivalProbabilityDamping(
            "P(ee)",
            leading_mass_splitting_3l_name=leading_mass_splitting_3l_name,
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

        (in_dm := Array(
            "damping_model",
            [float(_DAMPING_MODEL_MAP[damping_model])],
            mode="fill",
        )) >> surprob("damping_model")

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

    def surprob_fcn() -> float:
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
            correction = _SinQuTheta13 * (exp(-a * L_m / E) - 1.0)
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

        res = (
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
        return res

    atol = finfo("d").resolution * 2

    # ---- first evaluation
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    subplots(1, 1)
    plot_auto(
        surprob, filter_kw={"masked_value": 0},
        show=False, close=True,
        save=f"{output_path}/{test_name}_plot.pdf",
    )

    # ---- oscillation-parameter variations
    nmo *= -1; in_nmo.outputs[0].set(nmo)
    assert surprob.tainted is True
    assert allclose(surprob.outputs[0].data, surprob_fcn(), rtol=0, atol=atol)

    DeltaMSq21 *= 1.1; in_Dm21.outputs[0].set(DeltaMSq21)
    assert allclose(surprob.outputs[0].data, surprob_fcn(), rtol=0, atol=atol)

    DeltaMSq3lAbs *= 0.9; in_Dm3l.outputs[0].set(DeltaMSq3lAbs)
    assert allclose(surprob.outputs[0].data, surprob_fcn(), rtol=0, atol=atol)

    SinSq2Theta12 *= 1.2; in_t12.outputs[0].set(SinSq2Theta12)
    assert allclose(surprob.outputs[0].data, surprob_fcn(), rtol=0, atol=atol)

    SinSq2Theta13 += 0.1; in_t13.outputs[0].set(SinSq2Theta13)
    assert allclose(surprob.outputs[0].data, surprob_fcn(), rtol=0, atol=atol)

    L *= 10; in_L.outputs[0].set(L)
    assert allclose(surprob.outputs[0].data, surprob_fcn(), rtol=0, atol=atol)

    E *= 15; in_E.outputs[0].set(E)
    assert allclose(surprob.outputs[0].data, surprob_fcn(), rtol=0, atol=atol)

    # ---- conversion-factor variation
    conversionFactor *= 1.01; in_conv.outputs[0].set(conversionFactor)
    assert surprob.tainted is True
    assert allclose(surprob.outputs[0].data, surprob_fcn(), rtol=0, atol=atol)

    # ---- damping-parameter variations
    if damping_model == "decay":
        damping_params["alpha_decay"] *= 3.0
        in_ad.outputs[0].set(damping_params["alpha_decay"])
        assert surprob.tainted is True
        assert allclose(surprob.outputs[0].data, surprob_fcn(), rtol=0, atol=atol)

    elif damping_model == "wpd1":
        damping_params["alpha_wpd1"] *= 3.0
        in_aw1.outputs[0].set(damping_params["alpha_wpd1"])
        assert surprob.tainted is True
        assert allclose(surprob.outputs[0].data, surprob_fcn(), rtol=0, atol=atol)

    elif damping_model == "wpd2":
        damping_params["alpha_wpd2"] *= 3.0
        in_aw2.outputs[0].set(damping_params["alpha_wpd2"])
        assert surprob.tainted is True
        assert allclose(surprob.outputs[0].data, surprob_fcn(), rtol=0, atol=atol)

    savegraph(graph, f"{output_path}/{test_name}.png")