from __future__ import annotations

from typing import TYPE_CHECKING

from dagflow.core.meta_node import MetaNode
from multikeydict.typing import KeyLike, strkey

from .EeToEnu import EeToEnu
from .IBDXsecVBO1 import IBDXsecVBO1
from .Jacobian_dEnu_dEe import Jacobian_dEnu_dEe

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal

    from dagflow.core import NodeStorage


class IBDXsecVBO1Group(MetaNode):
    __slots__ = ("_input_energy_type",)
    _input_energy_type: str

    def __init__(
        self,
        name_ibd: str = "ibd",
        name_enu: str = "enu",
        name_jacobian: str = "jacobian",
        *,
        input_energy: Literal["ee", "edep"] = "ee",
        labels: dict = {},
    ):
        super().__init__(strategy="Disable")

        ibdxsec = IBDXsecVBO1(name_ibd, label=labels.get("xsec", {}))
        eetoenu = EeToEnu(
            name_enu, input_energy=input_energy, label=labels.get("enu", {})
        )
        jacobian = Jacobian_dEnu_dEe(
            name_jacobian, input_energy=input_energy, label=labels.get("jacobian", {})
        )

        eetoenu.outputs["result"] >> (jacobian.inputs["enu"], ibdxsec.inputs["enu"])

        self._input_energy_type = input_energy
        inputs_common = ["ElectronMass", "ProtonMass", "NeutronMass"]
        inputs_ibd = inputs_common + [
            "NeutronLifeTime",
            "PhaseSpaceFactor",
            "g",
            "f",
            "f2",
        ]
        merge_inputs = [self._input_energy_type, "costheta"] + inputs_common
        self._add_node(
            ibdxsec,
            kw_inputs=["costheta"] + inputs_ibd,
            merge_inputs=merge_inputs,
            outputs_pos=True,
        )
        self._add_node(
            eetoenu,
            kw_inputs=[self._input_energy_type, "costheta"] + inputs_common,
            merge_inputs=merge_inputs,
            kw_outputs={"result": "enu"},
        )
        self._add_node(
            jacobian,
            kw_inputs=["enu", self._input_energy_type, "costheta"] + inputs_common[:-1],
            merge_inputs=merge_inputs[:-1],
            kw_outputs={"result": "jacobian"},
        )
        self.inputs.make_positionals(self._input_energy_type, "costheta")

    @classmethod
    def replicate(
        cls,
        *args,
        names: Mapping[str, str] = {
            "ibd": "crosssection",
            "enu": "enu",
            "jacobian": "jacobian",
        },
        path: KeyLike = "ibd",
        verbose: bool = False,
        **kwargs,
    ) -> tuple[IBDXsecVBO1Group, NodeStorage]:
        from dagflow.core import NodeStorage

        path = strkey(path)
        name_ibd = strkey((path, names.get("ibd", "crosssection")))
        name_enu = strkey((path, names.get("enu", "enu")))
        name_jacobian = strkey((path, names.get("jacobian", "jacobian")))

        storage = NodeStorage(default_containers=True)
        nodes = storage.create_child("nodes")
        inputs = storage.create_child("inputs")
        outputs = storage.create_child("outputs")

        ibd = cls(name_ibd, name_enu, name_jacobian, *args, **kwargs)

        nodes[name_ibd] = ibd
        inputs[name_ibd, ibd._input_energy_type] = ibd.inputs[ibd._input_energy_type]
        inputs[name_ibd, "costheta"] = ibd.inputs["costheta"]
        outputs[name_ibd] = ibd.outputs["result"]
        outputs[name_enu] = ibd.outputs["enu"]
        outputs[name_jacobian] = ibd.outputs["jacobian"]

        NodeStorage.update_current(storage, strict=True, verbose=verbose)

        return ibd, storage
