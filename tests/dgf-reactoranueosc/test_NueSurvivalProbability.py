#!/usr/bin/env python

from matplotlib.pyplot import subplots
from numpy import allclose, arcsin, cos, finfo, geomspace, sin, sqrt
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.plot import plot_auto
from dgf_reactoranueosc.NueSurvivalProbability import (NueSurvivalProbability,
                                                       _surprobArgConversion)


@mark.parametrize("nmo", (1, -1))  # mass ordering
@mark.parametrize("L", (2, 52, 180))  # km
@mark.parametrize(
    "conversionFactor",
    (None, _surprobArgConversion, 0.9 * _surprobArgConversion),
)
def test_NueSurvivalProbability_01(
    debug_graph, testname, L, nmo, conversionFactor
):
    E = geomspace(1, 100, 1000)  # MeV
    DeltaMSq21 = 7.39 * 1e-5  # eV^2
    DeltaMSq32 = 2.45 * 1e-3  # eV^2
    SinSq2Theta12 = 3.1 * 1e-1  # [-]
    SinSq2Theta13 = 2.241 * 1e-2  # [-]

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        surprob = NueSurvivalProbability("P(ee)")
        (in_E:=Array("E", E)) >> surprob("E")
        (in_L:=Array("L", [L])) >> surprob("L")
        (in_nmo:=Array("nmo", [nmo])) >> surprob("nmo")
        (in_Dm21:=Array("DeltaMSq21", [DeltaMSq21])) >> surprob("DeltaMSq21")
        (in_Dm32:=Array("DeltaMSq32", [DeltaMSq32])) >> surprob("DeltaMSq32")
        (in_t12:=Array("SinSq2Theta12", [SinSq2Theta12])) >> surprob("SinSq2Theta12")
        (in_t13:=Array("SinSq2Theta13", [SinSq2Theta13])) >> surprob("SinSq2Theta13")
        if conversionFactor is not None:
            (in_conversion:=Array("surprobArgConversion", [conversionFactor])) >> surprob(
                "surprobArgConversion"
            )
        else:
            in_conversion = None
    if conversionFactor is None:
        conversionFactor = _surprobArgConversion

    def surprob_fcn() -> float:
        tmp = L * conversionFactor / 4.0 / E
        _DeltaMSq32 = nmo * DeltaMSq32  # Δm²₃₂ = α*|Δm²₃₂|
        _DeltaMSq31 = nmo * DeltaMSq32 + DeltaMSq21  # Δm²₃₁ = α*|Δm²₃₂| + Δm²₂₁
        theta12 = 0.5*arcsin(sqrt(SinSq2Theta12))
        theta13 = 0.5*arcsin(sqrt(SinSq2Theta13))
        _SinSqTheta12 = sin(theta12)**2  # sin²θ₁₂
        _CosSqTheta12 = cos(theta12)**2  # cos²θ₁₂
        _CosQuTheta13 = (cos(theta13)**2)**2 # cos^4 θ₁₃
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
        save=f"output/{testname}_plot.pdf",
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

    DeltaMSq32 *= 0.9
    in_Dm32.outputs[0].set(DeltaMSq32)
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

    savegraph(graph, f"output/{testname}.png")
