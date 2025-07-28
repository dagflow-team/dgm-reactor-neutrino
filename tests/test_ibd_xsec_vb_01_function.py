#!/usr/bin/env python

from matplotlib import pyplot as plt
from numpy import linspace, meshgrid, zeros_like

from dgm_reactor_neutrino.ibd_xsec_vb_o1 import _ibdxsecO1

plt.rcParams.update(
    {
        "axes.grid": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.formatter.use_mathtext": True,
    }
)


def test_IBDXsecVBO1_function(testname, show=False):
    NeutronLifeTime = 879.4
    NeutronMass = 939.565413
    ProtonMass = 938.272081
    ElectronMass = 0.5109989461
    const_fps = 1.71465
    const_g = 1.2701
    const_f = 1.0
    const_f2 = 3.706

    enu1 = linspace(0, 12.0, 121, dtype="d")
    ctheta1 = linspace(-1, 1, 3, dtype="d")
    enu2, ctheta2 = meshgrid(enu1, ctheta1, indexing="ij")
    result = zeros_like(enu2)

    _ibdxsecO1(
        enu2,
        ctheta2,
        result,
        ElectronMass,
        ProtonMass,
        NeutronMass,
        NeutronLifeTime,
        const_fps,
        const_g,
        const_f,
        const_f2,
    )

    plt.subplots(1, 1)
    ax = plt.gca()
    for ictheta, ctheta in enumerate(ctheta1):
        ax.plot(enu2[:, ictheta], result[:, ictheta], label=f"cosθ={ctheta}")

    ax.set_title("IBD cross section")
    ax.set_xlabel("Eν, MeV")
    ax.set_ylabel("dσ/dE/dcosθ")
    ax.legend()

    if show:
        plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("input")

    test_IBDXsecVBO1_function(testname="test_IBDXsecVBO1_function", show=True)
