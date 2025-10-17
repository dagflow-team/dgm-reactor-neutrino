from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dag_modelling.core.global_parameters import NUMBA_CACHE_ENABLE
from dag_modelling.core.input_strategy import AddNewInputAddNewOutput
from dag_modelling.core.node import Node
from dag_modelling.core.type_functions import (
    assign_axes_from_inputs_to_outputs,
    check_dimension_of_inputs,
    check_inputs_equivalence,
    copy_from_inputs_to_outputs,
)
from numba import njit
from numpy import sqrt

if TYPE_CHECKING:
    from dag_modelling.core.input import Input
    from dag_modelling.core.output import Output
    from numpy import double
    from numpy.typing import NDArray


class EeToEnu(Node):
    """Enu(Ee, cosθ)"""

    __slots__ = (
        "_e_input",
        "_ctheta",
        "_result",
        "_const_me",
        "_const_mp",
        "_const_mn",
        "_input_energy_type",
        "_use_edep",
    )

    _e_input: Input
    _ctheta: Input
    _result: Output

    _const_me: Input
    _const_mp: Input
    _const_mn: Input

    _input_energy_type: Literal["ee", "edep"]
    _use_edep: bool

    def __init__(self, name, *args, input_energy: Literal["ee", "edep"] = "ee", **kwargs):
        super().__init__(name, *args, **kwargs, input_strategy=AddNewInputAddNewOutput())
        self.labels.setdefaults(
            {
                "text": r"Neutrino energy Eν, MeV",
                "plot_title": r"Neutrino energy $E_{\nu}$, MeV",
                "latex": r"$E_{\nu}$, MeV",
                "axis": r"$E_{\nu}$, MeV",
            }
        )

        self._input_energy_type = input_energy
        match input_energy:
            case "ee":
                self._use_edep = False
            case "edep":
                self._use_edep = True
            case _:
                raise ValueError(f"Invalid `input_energy` {input_energy}")

        self._e_input = self._add_input(input_energy, positional=True, keyword=True)
        self._ctheta = self._add_input("costheta", positional=True, keyword=True)
        self._result = self._add_output("result", positional=True, keyword=True)

        self._const_me = self._add_input("ElectronMass", positional=False, keyword=True)
        self._const_mp = self._add_input("ProtonMass", positional=False, keyword=True)
        self._const_mn = self._add_input("NeutronMass", positional=False, keyword=True)

    def _function(self):
        _enu(
            self._e_input.data.ravel(),
            self._ctheta.data.ravel(),
            self._result._data.ravel(),
            self._const_me.data[0],
            self._const_mp.data[0],
            self._const_mn.data[0],
            self._use_edep,
        )

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_dimension_of_inputs(self, slice(0, 2), 2)
        check_inputs_equivalence(self, slice(0, 2))
        copy_from_inputs_to_outputs(
            self, self._input_energy_type, "result", edges=False, meshes=False
        )
        assign_axes_from_inputs_to_outputs(
            self,
            (self._input_energy_type, "costheta"),
            "result",
            assign_meshes=True,
            merge_input_axes=True,
        )


@njit(cache=NUMBA_CACHE_ENABLE)
def _enu(
    EeIn: NDArray[double],
    CosThetaIn: NDArray[double],
    Result: NDArray[double],
    ElectronMass: float,
    ProtonMass: float,
    NeutronMass: float,
    use_edep: bool,
):
    ElectronMass2 = ElectronMass * ElectronMass
    NeutronMass2 = NeutronMass * NeutronMass
    ProtonMass2 = ProtonMass * ProtonMass

    delta = 0.5 * (NeutronMass2 - ProtonMass2 - ElectronMass2) / ProtonMass

    for i, (Ee, ctheta) in enumerate(zip(EeIn, CosThetaIn)):
        if use_edep:
            Ee -= ElectronMass
        Ve = sqrt(1.0 - ElectronMass2 / (Ee * Ee)) if Ee > ElectronMass else 0.0
        epsilon_e = Ee / ProtonMass
        Ee0 = Ee + delta
        corr = 1.0 - epsilon_e * (1.0 - Ve * ctheta)
        Result[i] = Ee0 / corr
