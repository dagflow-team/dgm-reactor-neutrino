#!/usr/bin/env python
from dag_modelling.bundles.load_parameters import load_parameters
from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.plot.plot import plot_auto
from dgm_reactor_neurino.IBDXsecVBO1Group import IBDXsecVBO1Group
from matplotlib.pyplot import subplots
from numpy import linspace, meshgrid


def test_IBDXsecVBO1Group(debug_graph, testname):
    data = {
        "format": "value",
        "state": "fixed",
        "parameters": {
            "NeutronLifeTime": 879.4,  # s,   page 165
            "NeutronMass": 939.565413,  # MeV, page 165
            "ProtonMass": 938.272081,  # MeV, page 163
            "ElectronMass": 0.5109989461,  # MeV, page 16
            "PhaseSpaceFactor": 1.71465,
            "g": 1.2701,
            "f": 1.0,
            "f2": 3.706,
        },
        "labels": {
            "NeutronLifeTime": "neutron lifetime, s (PDG2014)",
            "NeutronMass": "neutron mass, MeV (PDG2012)",
            "ProtonMass": "proton mass, MeV (PDG2012)",
            "ElectronMass": "electron mass, MeV (PDG2012)",
            "PhaseSpaceFactor": "IBD phase space factor",
            "f": "vector coupling constant f",
            "g": "axial-vector coupling constant g",
            "f2": "anomalous nucleon isovector magnetic moment f₂",
        },
    }

    enu1 = linspace(1, 12.0, 111)
    ee1 = enu1.copy()
    ctheta1 = linspace(-1, 1, 5)
    enu2, ctheta2 = meshgrid(enu1, ctheta1, indexing="ij")
    ee2, _ = meshgrid(ee1, ctheta1, indexing="ij")

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        storage = load_parameters(data)

        ee = Array("ee", ee2, label={"axis": r"$E_{\rm pos}$, MeV"})
        ctheta = Array("ctheta", ctheta2, label={"axis": r"$\cos\theta$"})

        ibdxsec = IBDXsecVBO1Group()

        ibdxsec << storage("parameters.constant")
        ee >> ibdxsec.inputs["ee"]
        ctheta >> ibdxsec.inputs["costheta"]
        ibdxsec.print(recursive=True)

    csc_ee = ibdxsec.get_data()

    show = False
    close = not show
    from mpl_toolkits.mplot3d import axes3d  # accessed implicitly in `subplots()`

    subplots(1, 1, subplot_kw={"projection": "3d"})
    plot_auto(
        ibdxsec,
        plotoptions={"method": "surface"},
        cmap=True,
        colorbar=True,
        show=False,
        close=close,
        save=f"output/{testname}_xsec_surf.pdf",
    )

    subplots(1, 1, subplot_kw={"projection": "3d"})
    plot_auto(
        ibdxsec,
        plotoptions={"method": "wireframe"},
        cmap=True,
        colorbar=True,
        show=False,
        close=close,
        save=f"output/{testname}_xsec_wirec.pdf",
    )

    subplots(1, 1, subplot_kw={"projection": "3d"})
    plot_auto(
        ibdxsec,
        plotoptions={"method": "wireframe"},
        show=False,
        close=close,
        save=f"output/{testname}_xsec_wire.pdf",
    )

    subplots(1, 1)
    plot_auto(
        ibdxsec,
        plotoptions={"method": "pcolormesh"},
        colorbar=True,
        show=False,
        close=close,
        save=f"output/{testname}_xsec_mesh.pdf",
    )

    subplots(1, 1)
    plot_auto(
        ibdxsec.outputs["enu"],
        plotoptions={"method": "pcolormesh"},
        colorbar=True,
        show=False,
        close=close,
        save=f"output/{testname}_enu_mesh.pdf",
    )

    subplots(1, 1, subplot_kw={"projection": "3d"})
    plot_auto(
        ibdxsec.outputs["enu"],
        plotoptions={"method": "surface"},
        cmap=True,
        colorbar=True,
        show=False,
        close=close,
        save=f"output/{testname}_enu_surf.pdf",
    )

    subplots(1, 1)
    plot_auto(
        ibdxsec.outputs["jacobian"],
        plotoptions={"method": "pcolormesh"},
        colorbar=True,
        show=False,
        close=close,
        save=f"output/{testname}_jac_mesh.pdf",
    )

    subplots(1, 1, subplot_kw={"projection": "3d"})
    plot_auto(
        ibdxsec.outputs["jacobian"],
        plotoptions={"method": "surface"},
        cmap=True,
        colorbar=True,
        show=show,
        close=True,
        save=f"output/{testname}_jac_surf.pdf",
    )

    savegraph(graph, f"output/{testname}.pdf")
