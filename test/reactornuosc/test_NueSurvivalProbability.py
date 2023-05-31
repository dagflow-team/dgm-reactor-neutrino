#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from numpy import allclose, finfo, linspace, pi, sin, sqrt
from pytest import mark
from scipy.constants import value

from reactornueosc.NueSurvivalProbability import NueSurvivalProbability

_conversionFactor = (
    pi * 2e-3 * value("electron volt-inverse meter relationship")
)


@mark.parametrize("alpha", (1, -1))  # mass ordering
@mark.parametrize("L", (2, 52, 180))  # km
@mark.parametrize(
    "conversionFactor", (None, _conversionFactor, 0.9 * _conversionFactor)
)
def test_NueSurvivalProbability_01(
    debug_graph, testname, L, alpha, conversionFactor
):
    E = linspace(1, 10, 101)  # MeV
    DeltaMSq21 = 7.39 * 1e-5  # eV^2
    DeltaMSq32 = 2.45 * 1e-3  # eV^2
    sinSq2Theta12 = 3.1 * 1e-1  # [-]
    sinSq2Theta13 = 2.241 * 1e-2  # [-]

    with Graph(close=True, debug=debug_graph) as graph:
        oscprob = NueSurvivalProbability("P(ee)")
        Array("E", E) >> oscprob("E")
        Array("L", [L]) >> oscprob("L")
        Array("alpha", [alpha]) >> oscprob("alpha")
        Array("DeltaMSq21", [DeltaMSq21]) >> oscprob("DeltaMSq21")
        Array("DeltaMSq32", [DeltaMSq32]) >> oscprob("DeltaMSq32")
        Array("sinSq2Theta12", [sinSq2Theta12]) >> oscprob("sinSq2Theta12")
        Array("sinSq2Theta13", [sinSq2Theta13]) >> oscprob("sinSq2Theta13")
        if conversionFactor is not None:
            Array("oscprobArgConversion", [conversionFactor]) >> oscprob(
                "oscprobArgConversion"
            )
    if conversionFactor is None:
        conversionFactor = _conversionFactor

    tmp = L * conversionFactor / 4.0 / E
    _DeltaMSq32 = alpha * DeltaMSq32  # Δm²₃₂ = α*|Δm²₃₂|
    _DeltaMSq31 = alpha * DeltaMSq32 + DeltaMSq21  # Δm²₃₁ = α*|Δm²₃₂| + Δm²₂₁
    _sinSqTheta12 = 0.5 * (1 - sqrt(1 - sinSq2Theta12))  # sin²θ₁₂
    _cosSqTheta12 = 1.0 - _sinSqTheta12  # cos²θ₁₂
    _cosQuTheta13 = (0.5 * (1 - sqrt(1 - sinSq2Theta13))) ** 2  # cos^4 θ₁₃
    res = (
        1
        - sinSq2Theta13
        * (
            _sinSqTheta12 * sin(_DeltaMSq32 * tmp) ** 2
            + _cosSqTheta12 * sin(_DeltaMSq31 * tmp) ** 2
        )
        - sinSq2Theta12 * _cosQuTheta13 * sin(DeltaMSq21 * tmp) ** 2
    )

    atol = finfo("d").precision*2
    assert oscprob.tainted is True
    assert allclose(oscprob.outputs[0].data, res, rtol=0, atol=atol)
    assert oscprob.tainted is False

    savegraph(graph, f"output/{testname}.png")
