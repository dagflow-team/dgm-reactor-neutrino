#!/usr/bin/env python

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.plot.plot import plot_auto
from matplotlib.pyplot import subplots
from numpy import allclose, arcsin, cos, finfo, geomspace, sin, sqrt
from pytest import mark

from dgm_reactor_neutrino import NueSurvivalProbability
from dgm_reactor_neutrino.nue_survival_probability import _surprobArgConversion


@mark.parametrize("nmo", (1, -1))  # mass ordering
@mark.parametrize(
    "L,leading_mass_splitting_3l_name", ((2, "DeltaMSq32"), (52, "DeltaMSq31"), (180, "DeltaMSq32"))
)  # km
@mark.parametrize(
    "conversionFactor",
    (None, _surprobArgConversion, 0.9 * _surprobArgConversion),
)
def test_NueSurvivalProbability_01(
    debug_graph,
    test_name,
    L,
    leading_mass_splitting_3l_name,
    nmo,
    conversionFactor,
    output_path: str,
):
    E = geomspace(1, 100, 1000)  # MeV
    DeltaMSq21 = 7.39 * 1e-5  # eV^2
    DeltaMSq3lAbs = 2.45 * 1e-3  # eV^2
    SinSq2Theta12 = 3.1 * 1e-1  # [-]
    SinSq2Theta13 = 2.241 * 1e-2  # [-]
    is_dm32_leading = leading_mass_splitting_3l_name == "DeltaMSq32"
    is_dm31_leading = leading_mass_splitting_3l_name == "DeltaMSq31"

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        surprob = NueSurvivalProbability(
            "P(ee)", leading_mass_splitting_3l_name=leading_mass_splitting_3l_name
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
        if conversionFactor is not None:
            (
                in_conversion := Array("surprobArgConversion", [conversionFactor], mode="fill")
            ) >> surprob("surprobArgConversion")
        else:
            in_conversion = None
    if conversionFactor is None:
        conversionFactor = _surprobArgConversion

    def surprob_fcn() -> float:
        tmp = L * conversionFactor / 4.0 / E
        _DeltaMSq31 = (
            nmo * DeltaMSq3lAbs + DeltaMSq21 * is_dm32_leading
        )  # Δm²₃₁ = α*|Δm²₃ₗ| + β*Δm²₂₁
        _DeltaMSq32 = (
            nmo * DeltaMSq3lAbs - DeltaMSq21 * is_dm31_leading
        )  # Δm²₃₂ = α*|Δm²₃ₗ| - β*Δm²₂₁
        theta12 = 0.5 * arcsin(sqrt(SinSq2Theta12))
        theta13 = 0.5 * arcsin(sqrt(SinSq2Theta13))
        _SinSqTheta12 = sin(theta12) ** 2  # sin²θ₁₂
        _CosSqTheta12 = cos(theta12) ** 2  # cos²θ₁₂
        _CosQuTheta13 = (cos(theta13) ** 2) ** 2  # cos^4 θ₁₃
        res = (
            1
            - SinSq2Theta13
            * (
                _SinSqTheta12 * sin(_DeltaMSq32 * tmp) ** 2
                + _CosSqTheta12 * sin(_DeltaMSq31 * tmp) ** 2
            )
            - SinSq2Theta12 * _CosQuTheta13 * sin(DeltaMSq21 * tmp) ** 2
        )
        return res

    atol = finfo("d").resolution * 2
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    subplots(1, 1)
    plot_auto(
        surprob,
        filter_kw={"masked_value": 0},
        show=False,
        close=True,
        save=f"{output_path}/{test_name}_plot.pdf",
    )

    nmo *= -1
    in_nmo.outputs[0].set(nmo)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    DeltaMSq21 *= 1.1
    in_Dm21.outputs[0].set(DeltaMSq21)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    DeltaMSq3lAbs *= 0.9
    in_Dm3l.outputs[0].set(DeltaMSq3lAbs)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    SinSq2Theta12 *= 1.2
    in_t12.outputs[0].set(SinSq2Theta12)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    SinSq2Theta13 += 0.1
    in_t13.outputs[0].set(SinSq2Theta13)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    L *= 10
    in_L.outputs[0].set(L)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    E *= 15
    in_E.outputs[0].set(E)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    if in_conversion is not None:
        conversionFactor *= 1.01
        in_conversion.outputs[0].set(conversionFactor)
        assert surprob.tainted is True
        res = surprob_fcn()
        assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
        assert surprob.tainted is False

    savegraph(graph, f"{output_path}/{test_name}.png")
