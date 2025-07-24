#!/usr/bin/env python

from dag_modelling.bundles.load_parameters import load_parameters
from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.plot.plot import plot_auto
from dgm_reactor_neurino.EeToEnu import EeToEnu
from dgm_reactor_neurino.IBDXsecVBO1 import IBDXsecVBO1
from dgm_reactor_neurino.Jacobian_dEnu_dEe import Jacobian_dEnu_dEe
from matplotlib.pyplot import subplots
from numpy import linspace, meshgrid


def test_IBDXsecVBO1(debug_graph, testname):
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

    enu1 = linspace(0, 12.0, 121)
    ee1 = enu1.copy()
    ctheta1 = linspace(-1, 1, 5)
    enu2, ctheta2 = meshgrid(enu1, ctheta1, indexing="ij")
    ee2, _ = meshgrid(ee1, ctheta1, indexing="ij")

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        storage = load_parameters(data)

        enu = Array("enu", enu2, mode="fill")
        ee = Array("ee", ee2, mode="fill")
        ctheta = Array("ctheta", ctheta2, mode="fill")

        ibdxsec_enu = IBDXsecVBO1("ibd_Eν")
        ibdxsec_ee = IBDXsecVBO1("ibd_Ee")
        eetoenu = EeToEnu("Enu")
        jacobian = Jacobian_dEnu_dEe("dEν/dEe")

        ibdxsec_enu << storage("parameters.constant")
        ibdxsec_ee << storage("parameters.constant")
        eetoenu << storage("parameters.constant")
        jacobian << storage("parameters.constant")

        (enu, ctheta) >> ibdxsec_enu
        (ee, ctheta) >> eetoenu
        (eetoenu, ee, ctheta) >> jacobian
        (eetoenu, ctheta) >> ibdxsec_ee

    csc_enu = ibdxsec_enu.get_data()
    csc_ee = ibdxsec_ee.get_data()
    enu = eetoenu.get_data()
    jac = jacobian.get_data()

    subplots(1, 1)
    plot_auto(
        ibdxsec_enu,
        plotoptions={"method": "pcolormesh"},
        colorbar=True,
        filter_kw={"masked_value": 0},
        show=False,
        close=True,
        save=f"output/{testname}_plot.pdf",
    )

    savegraph(graph, f"output/{testname}.pdf")
