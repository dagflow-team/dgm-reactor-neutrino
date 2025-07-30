#!/usr/bin/env python

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from numpy import allclose, finfo, linspace, pi
from pytest import mark

from dgm_reactor_neutrino import InverseSquareLaw

_scales = {"km_to_cm": 1e5, "m_to_cm": 1e2, None: 1}


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("scalename", tuple(_scales))
def test_InverseSquareLaw_01(debug_graph, test_name, dtype, scalename):
    arrays_in = tuple(linspace(1, 10, 10, dtype=dtype) * i for i in (1, 2, 3))

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        arrays = tuple(Array("test", array_in, mode="fill") for array_in in arrays_in)
        isl = InverseSquareLaw("InvSqLaw", scale=scalename)
        arrays >> isl

    scale = _scales[scalename]
    res_all = tuple(0.25 / pi / (a * scale) ** 2 for a in arrays_in)

    atol = finfo(dtype).resolution * 2
    assert isl.tainted is True
    assert all(output.dd.dtype == dtype for output in isl.outputs)
    assert all(
        allclose(output.data, res, rtol=0, atol=atol)
        for output, res in zip(isl.outputs, res_all)
    )
    assert isl.tainted is False

    savegraph(graph, f"output/{test_name}.png")
