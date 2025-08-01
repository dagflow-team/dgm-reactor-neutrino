import numpy as np
from dgm_reactor_neutrino.ibd_xsec_sv import IBDXsecSVO1




def test_IBDXsecSV(debug_graph, test_name: str, output_path: str):
    # enu = np.linspace(0, 12.0, 121, dtype="d")
    # ctheta = np.linspace(-1, 1, 5, dtype="d")
    # enu2, ctheta2 = np.meshgrid(enu, ctheta, indexing="ij")
    # result = np.array([])
    data = {
        "format": "value",
        "state": "fixed",
        "parameters":{
            "NeutronMass": 939.565413,
            "ProtonMass": 938.272081,
            "ElectronMass": 0.5109989461,
            "PionMass": 134.97,
            "CosOfCab": 0.9746,
            "xi": 3.706,
            "g1_0": -1.270,
            "MAsq": 1.0 * 10 ** 6,
            "MVsq": 0.71 * 10 ** 6,
            "M_Z": 80.385 * 1.0e3
        },
        "labels":{

        }
    }





