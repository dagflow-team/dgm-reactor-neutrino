from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from numba import njit
from numpy import sqrt

from dagflow.core.input_handler import MissingInputAddPair
from dagflow.core.node import Node
from dagflow.core.type_functions import (
    assign_axes_from_inputs_to_outputs,
    check_dimension_of_inputs,
    check_inputs_equivalence,
    copy_from_inputs_to_outputs,
)

if TYPE_CHECKING:
    from numpy import double
    from numpy.typing import NDArray

    from dagflow.core.input import Input
    from dagflow.core.output import Output


class Jacobian_dEnu_dEe(Node):
    """Enu(Ee, cosθ)"""

    __slots__ = (
        "_enu",
        "_e_input",
        "_ctheta",
        "_result",
        "_const_me",
        "_const_mp",
        "_const_mn",
        "_input_energy_type",
        "_use_edep",
    )

    _enu: Input
    _e_input: Input
    _ctheta: Input
    _result: Output

    _const_me: Input
    _const_mp: Input
    _const_mn: Input

    _input_energy_type: Literal["ee", "edep"]
    _use_edep: bool

    def __init__(self, name, *args, input_energy: Literal["ee", "edep"] = "ee", **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddPair())
        super().__init__(name, *args, **kwargs)

        self._input_energy_type = input_energy
        match input_energy:
            case "ee":
                self._use_edep = False
                self.labels.setdefaults(
                    {
                        "text": r"Energy conversion Jacobian dEν/dEe",
                        "plottitle": r"Energy conversion Jacobian $dE_{\nu}/dE_{e}$",
                        "latex": r"$dE_{\nu}/dE_{e}$",
                        "axis": r"$dE_{\nu}/dE_{e}$",
                    }
                )
            case "edep":
                self._use_edep = True
                self.labels.setdefaults(
                    {
                        "text": r"Energy conversion Jacobian dEν/dEdep",
                        "plottitle": r"Energy conversion Jacobian $dE_{\nu}/dE_{\rm dep}$",
                        "latex": r"$dE_{\nu}/dE_{\rm dep}$",
                        "axis": r"$dE_{\nu}/dE_{\rm dep}$",
                    }
                )
            case _:
                raise ValueError(f"Invalid `input_energy` {input_energy}")

        self._enu = self._add_input("enu", positional=True, keyword=True)
        self._e_input = self._add_input(input_energy, positional=True, keyword=True)
        self._ctheta = self._add_input("costheta", positional=True, keyword=True)
        self._result = self._add_output("result", positional=True, keyword=True)

        self._const_me = self._add_input("ElectronMass", positional=False, keyword=True)
        self._const_mp = self._add_input("ProtonMass", positional=False, keyword=True)

    def _function(self):
        _jacobian_dEnu_dEe(
            self._enu.data.ravel(),
            self._e_input.data.ravel(),
            self._ctheta.data.ravel(),
            self._result.data.ravel(),
            self._const_me.data[0],
            self._const_mp.data[0],
            self._use_edep,
        )

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_dimension_of_inputs(self, slice(0, 3), 2)
        check_inputs_equivalence(self, slice(0, 3))
        copy_from_inputs_to_outputs(self, self._input_energy_type, "result", edges=False, meshes=False)
        assign_axes_from_inputs_to_outputs(
            self,
            (self._input_energy_type, "costheta"),
            "result",
            assign_meshes=True,
            merge_input_axes=True,
        )


@njit(cache=True)
def _jacobian_dEnu_dEe(
    EnuIn: NDArray[double],
    EeIn: NDArray[double],
    CosThetaIn: NDArray[double],
    Result: NDArray[double],
    ElectronMass: float,
    ProtonMass: float,
    use_edep: bool,
):
    ElectronMass2 = ElectronMass * ElectronMass

    for i, (Enu, Ee, ctheta) in enumerate(zip(EnuIn, EeIn, CosThetaIn)):
        if use_edep:
            Ee -= ElectronMass
        if Ee <= ElectronMass:
            Result[i] = 0.0
            continue
        Ve = sqrt(1.0 - ElectronMass2 / (Ee * Ee))
        nominator = ProtonMass + Enu * (1.0 - ctheta / Ve)
        denominator = ProtonMass - Ee * (1 - Ve * ctheta)
        if denominator <= 0:
            Result[i] = 0.0
            continue
        Result[i] = nominator / denominator
