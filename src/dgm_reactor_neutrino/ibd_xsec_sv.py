from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
from numpy import log, pi, power, sqrt
from scipy.constants import value as constant

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from numpy import double
    from dag_modelling.core.input import Input
    from dag_modelling.core.output import Output

from dag_modelling.core.input_strategy import AddNewInputAddNewOutput
from dag_modelling.core.node import Node
from dag_modelling.core.type_functions import (
    assign_axes_from_inputs_to_outputs,
    check_dimension_of_inputs,
    check_dtype_of_inputs,
    check_inputs_equivalence,
    copy_from_inputs_to_outputs,
)


class IBDXsecSV(Node):
    """Inverse beta decay cross
    section by Strumia and Vissani."""

    __slots__ = (
        "_enu",
        "_ctheta",
        "_result",
        "_const_cos_cab",
        "_const_mn",
        "_const_mp",
        "_const_me",
        "_const_mpi",
        "_const_xi",
        "_const_g1_0",
        "_const_masq",
        "_const_mvsq",
        "_const_mz",
    )

    _enu: Input
    _ctheta: Input
    _result: Output
    _const_cos_cab: Input
    _const_mn: Input
    _const_mp: Input
    _const_me: Input
    _const_mpi: Input
    _const_xi: Input
    _const_g1_0: Input
    _const_masq: Input
    _const_mvsq: Input
    _const_mz: Input

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs, input_strategy=AddNewInputAddNewOutput())
        self.labels.setdefaults(
            {
                "text": r"IBD cross section σ(Eν,cosθ), cm⁻²",
                "plot_title": r"IBD cross section $\sigma(E_{\nu}, \cos\theta)$, cm$^{-2}$",
                "latex": r"IBD cross section $\sigma(E_{\nu}, \cos\theta)$, cm$^{-2}$",
                "axis": r"$\sigma(E_{\nu}, \cos\theta)$, cm$^{-2}$",
            }
        )

        self._enu = self._add_input("enu", positional=True, keyword=True)
        self._ctheta = self._add_input("costheta", positional=True, keyword=True)
        self._result = self._add_output("result", positional=True, keyword=True)
        self._const_cos_cab = self._add_input("CosOfCab", positional=False, keyword=True)
        self._const_mn = self._add_input("NeutronMass", positional=False, keyword=True)
        self._const_mp = self._add_input("ProtonMass", positional=False, keyword=True)
        self._const_me = self._add_input("ElectronMass", positional=False, keyword=True)
        self._const_mpi = self._add_input("PionMass", positional=False, keyword=True)
        self._const_xi = self._add_input("xi", positional=False, keyword=True)
        self._const_g1_0 = self._add_input("g1_0", positional=False, keyword=True)
        self._const_masq = self._add_input("MAsq", positional=False, keyword=True)
        self._const_mvsq = self._add_input("MVsq", positional=False, keyword=True)
        self._const_mz = self._add_input("MZ", positional=False, keyword=True)

    def _function(self):
        _ibdxsec(
            EnuIn=self._enu.data,
            CosThetaIn=self._ctheta.data,
            Result=self._result._data,
            CosOfCab=self._const_cos_cab.data[0],
            NeutronMass=self._const_mn.data[0],
            ProtonMass=self._const_mp.data[0],
            ElectronMass=self._const_me.data[0],
            PionMass=self._const_mpi.data[0],
            xi=self._const_xi.data[0],
            g1_0=self._const_g1_0.data[0],
            MAsq=self._const_masq.data[0],
            MVsq=self._const_mvsq.data[0],
            MZ=self._const_mz.data[0],
        )

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_dtype_of_inputs(self, slice(None), dtype="d")
        check_dimension_of_inputs(self, slice(0, 1), 2)
        check_inputs_equivalence(self, slice(0, 1))
        copy_from_inputs_to_outputs(self, "enu", "result", edges=False, meshes=False)
        assign_axes_from_inputs_to_outputs(
            self,
            ("enu", "costheta"),
            "result",
            assign_meshes=True,
            merge_input_axes=True,
        )


_constant_hbar = constant("reduced Planck constant")
_constant_qe = constant("elementary charge")
_constant_c = constant("speed of light in vacuum")
_constant_aplha = constant("fine-structure constant")
_constant_g_Fermi = constant("Fermi coupling constant")


@njit(cache=True)
def _ibdxsec(
    EnuIn: NDArray[double],
    CosThetaIn: NDArray[double],
    Result: NDArray[double],
    NeutronMass: double,
    ProtonMass: double,
    ElectronMass: double,
    PionMass: double,
    CosOfCab: double,
    xi: double,
    g1_0: double,
    MAsq: double,
    MVsq: double,
    MZ: double,
):
    NeutronMass2 = NeutronMass * NeutronMass
    ProtonMass2 = ProtonMass * ProtonMass
    ElectronMass2 = ElectronMass * ElectronMass
    PionMass2 = PionMass * PionMass

    EnuThreshold = ((NeutronMass + ElectronMass) ** 2 - ProtonMass2) / (2 * ProtonMass)

    CosOfCab2 = CosOfCab * CosOfCab
    G_F2 = _constant_g_Fermi * _constant_g_Fermi

    delta = (NeutronMass2 - ProtonMass2 - ElectronMass2) / (2 * ProtonMass)

    MeV2J = 1.0e6 * _constant_qe
    J2MeV = 1.0 / MeV2J
    MeV2cm = power(_constant_hbar * _constant_c * J2MeV, 2) * 1.0e4

    DeltaIn = _constant_aplha / pi * (2 * log(MZ / ProtonMass) + 0.55)
    DeltaIn1 = DeltaIn + 1

    result = Result.ravel()
    for i, (Enu, CosTheta) in enumerate(zip(EnuIn.ravel(), CosThetaIn.ravel())):
        if Enu < EnuThreshold:
            result[i] = 0.0
            continue

        if CosTheta < -1.0:
            CosTheta = -1.0

        if CosTheta > 1.0:
            CosTheta = 1.0

        eps = Enu / ProtonMass

        eps1 = 1.0 + eps

        kappa = eps1*eps1 - (eps * CosTheta) ** 2

        EnuDelta = Enu - delta
        EnuDelta2 = EnuDelta * EnuDelta

        if EnuDelta2 - ElectronMass2 * kappa <= 0.0:
            result[i] = 0.0
            continue

        Ee = (
            EnuDelta * eps1 + eps * CosTheta * sqrt(EnuDelta2 - ElectronMass2 * kappa)
        ) / kappa

        s = ProtonMass2 + 2 * ProtonMass * Enu
        u = s - 2 * ProtonMass * (Enu + Ee) + ElectronMass2

        su = s - u
        A, B, C = __coeff_A_B_C(
            Enu=Enu,
            Ee=Ee,
            NeutronMass=NeutronMass,
            ProtonMass=ProtonMass,
            ElectronMass2=ElectronMass2,
            PionMass2=PionMass2,
            xi=xi,
            g1_0=g1_0,
            MAsq=MAsq,
            MVsq=MVsq,
        )

        Matrix_elem = A - su * B + C * su * su

        dsigma_dt = ((G_F2 * CosOfCab2) / (2 * pi * (s - ProtonMass2) ** 2)) * Matrix_elem

        dsigma_dE = 2 * ProtonMass * dsigma_dt

        if Ee <= ElectronMass:
            result[i] = 0.0
            continue

        pe = sqrt(Ee * Ee - ElectronMass2)

        dsigma_dcos = (
            eps * pe / (1 + eps * (1 - (Ee / pe) * CosTheta)) * dsigma_dE
        ) * MeV2cm  # converting from Mev^-2 to cm^2

        # Delta = _constant_aplha / pi * (6 + 1.5 * log(ProtonMass/(2 * Ee)) + 1.2 * (ProtonMass/ Ee) ** 1.5)

        result[i] = dsigma_dcos * DeltaIn1


@njit(cache=True)
def __coeff_A_B_C(
    Enu: double,
    Ee: double,
    NeutronMass: double,
    ProtonMass: double,
    ElectronMass2: double,
    PionMass2: double,
    xi: double,
    g1_0: double,
    MAsq: double,
    MVsq: double,
) -> tuple [double, double, double]:

    NeutronMass2 = NeutronMass * NeutronMass
    ProtonMass2 = ProtonMass * ProtonMass

    NucleonsMass = 0.5 * (NeutronMass + ProtonMass)
    DeltaNP = NeutronMass - ProtonMass

    NucleonsMass2 = NucleonsMass * NucleonsMass
    DeltaNP2 = DeltaNP * DeltaNP

    t = NeutronMass2 - ProtonMass2 - 2 * ProtonMass * (Enu - Ee)

    f1 = (4 - (1 + xi) * t / NucleonsMass2) / ((4 - t / NucleonsMass2) * (1 - t / MVsq) ** 2)
    f2 = xi / ((1 - t / (4 * NucleonsMass2)) * (1 - t / MVsq) ** 2)
    g1 = g1_0 / ((1 - t / MAsq) ** 2)
    g2 = (2 * NucleonsMass2 * g1) / (PionMass2 - t)

    f1sq = f1 * f1
    f2sq = f2 * f2
    g1sq = g1 * g1
    g2sq = g2 * g2

    f12 = f1 * f2
    g12 = g1 * g2

    A = (1 / 16) * (
        (t - ElectronMass2)
        * (
            4 * f1sq * (4 * NucleonsMass2 + t + ElectronMass2)
            + 4 * g1sq * (-4 * NucleonsMass2 + t + ElectronMass2)
            + f2sq * (t * t / NucleonsMass2 + 4 * t + 4 * ElectronMass2)
            + 4 * ElectronMass2 * t * g2sq / NucleonsMass2
            + 8 * f12 * (2 * t + ElectronMass2)
            + 16 * ElectronMass2 * g12
        )
        - DeltaNP2
        * (
            (4 * f1sq + t * f2sq / NucleonsMass2) * (4 * NucleonsMass2 + t - ElectronMass2)
            + 4 * g1sq * (4 * NucleonsMass2 - t + ElectronMass2)
            + 4 * ElectronMass2 * g2sq * (t - ElectronMass2) / NucleonsMass2
            + 8 * f12 * (2 * t - ElectronMass2)
            + 16 * ElectronMass2 * g12
        )
        - 32 * ElectronMass2 * NucleonsMass * DeltaNP * g1 * (f1 + f2)
    )

    B = (1 / 16) * (
        16 * t * g1 * (f1 + f2)
        + 4 * ElectronMass2 * DeltaNP * (f2sq + f12 + 2 * g12) / NucleonsMass
    )

    C = (1 / 16) * (4 * (f1sq + g1sq) - t * f2sq / NucleonsMass2)

    return A, B, C
