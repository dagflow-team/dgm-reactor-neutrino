from dag_modelling.bundles.load_parameters import load_parameters
from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.plot.plot import plot_auto
from matplotlib.pyplot import subplots
from numpy import linspace, meshgrid

from dgm_reactor_neutrino import EeToEnu, Jacobian_dEnu_dEe
from dgm_reactor_neutrino.ibd_xsec_sv import IBDXsecSV


def test_IBDXsecSV(debug_graph, test_name: str, output_path: str):
    data = {
        "format": "value",
        "state": "fixed",
        "parameters": {
            "NeutronMass": 939.565413,
            "ProtonMass": 938.272081,
            "ElectronMass": 0.5109989461,
            "PionMass": 134.97,
            "CosOfCab": 0.9746,
            "xi": 3.706,
            "g1_0": -1.270,
            "MAsq": 1.0 * 10 ** 6,
            "MVsq": 0.71 * 10 ** 6,
            "MZ": 80.385 * 1.0e3,
        },
        "labels": {
            "NeutronMass": "neutron mass, MeV",
            "ProtonMass": "proton mass, MeV",
            "ElectronMass": "electron mass, MeV",
            "PionMass": "pion mass, MeV",
            "CosOfCab": "the cosine of Cabbibo`s angle",
            "xi": "the difference between the proton and neutron anomalous magnetic moments in units of the nuclear magneton",
            "g1_0": "axial-vector coupling constant g at param t equal zero",
            "MAsq": "axial mass in square, MeV^2",
            "MVsq": "vector mass in square, MeV^2",
            "MZ": "Z-boson mass, MeV",
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

        ibdxsec_enu = IBDXsecSV("ibd_Eν")
        ibdxsec_ee = IBDXsecSV("ibd_Ee")
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
        save=f"{output_path}/{test_name}_plot.pdf",
    )

    savegraph(graph, f"{output_path}/{test_name}.dot")
