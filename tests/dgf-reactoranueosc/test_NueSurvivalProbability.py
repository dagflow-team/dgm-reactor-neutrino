#!/usr/bin/env python

from matplotlib.pyplot import subplots
from numpy import allclose, arcsin, cos, finfo, geomspace, sin, sqrt
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.plot import plot_auto
from dgf_reactoranueosc.NueSurvivalProbability import (NueSurvivalProbability,
                                                       _oscprobArgConversion)


@mark.parametrize("nmo", (1, -1))  # mass ordering
@mark.parametrize("L", (2, 52, 180))  # km
@mark.parametrize(
    "conversionFactor",
    (None, _oscprobArgConversion, 0.9 * _oscprobArgConversion),
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
        oscprob = NueSurvivalProbability("P(ee)")
        Array("E", E) >> oscprob("E")
        Array("L", [L]) >> oscprob("L")
        Array("nmo", [nmo]) >> oscprob("nmo")
        Array("DeltaMSq21", [DeltaMSq21]) >> oscprob("DeltaMSq21")
        Array("DeltaMSq32", [DeltaMSq32]) >> oscprob("DeltaMSq32")
        Array("SinSq2Theta12", [SinSq2Theta12]) >> oscprob("SinSq2Theta12")
        Array("SinSq2Theta13", [SinSq2Theta13]) >> oscprob("SinSq2Theta13")
        if conversionFactor is not None:
            Array("oscprobArgConversion", [conversionFactor]) >> oscprob(
                "oscprobArgConversion"
            )
    if conversionFactor is None:
        conversionFactor = _oscprobArgConversion

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

    atol = finfo("d").resolution * 2
    assert oscprob.tainted is True
    assert allclose(oscprob.outputs[0].data, res, rtol=0, atol=atol)
    assert oscprob.tainted is False

    subplots(1, 1)
    plot_auto(
        oscprob,
        filter_kw={"masked_value": 0},
        show=False,
        close=True,
        save=f"output/{testname}_plot.pdf",
    )

    savegraph(graph, f"output/{testname}.png")
