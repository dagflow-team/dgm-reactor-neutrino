from __future__ import annotations

from typing import TYPE_CHECKING

from dagflow.core.meta_node import MetaNode
from multikeydict.typing import KeyLike, strkey

from .EeToEnu import EeToEnu
from .IBDXsecVBO1 import IBDXsecVBO1
from .Jacobian_dEnu_dEe import Jacobian_dEnu_dEe

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dagflow.core.storage import NodeStorage


class IBDXsecVBO1Group(MetaNode):
    __slots__ = ("_eename",)
    _eename: str

    def __init__(
        self,
        name_ibd: str = "ibd",
        name_enu: str = "enu",
        name_jacobian: str = "jacobian",
        *,
        use_edep: bool = False,
        labels: dict = {},
    ):
        super().__init__(strategy="Disable")

        ibdxsec = IBDXsecVBO1(name_ibd, label=labels.get("xsec", {}))
        eetoenu = EeToEnu(name_enu, use_edep=use_edep, label=labels.get("enu", {}))
        jacobian = Jacobian_dEnu_dEe(
            name_jacobian, use_edep=use_edep, label=labels.get("jacobian", {})
        )

        eetoenu.outputs["result"] >> (jacobian.inputs["enu"], ibdxsec.inputs["enu"])

        self._eename = "edep" if use_edep else "ee"
        inputs_common = ["ElectronMass", "ProtonMass", "NeutronMass"]
        inputs_ibd = inputs_common + [
            "NeutronLifeTime",
            "PhaseSpaceFactor",
            "g",
            "f",
            "f2",
        ]
        merge_inputs = [self._eename, "costheta"] + inputs_common
        self._add_node(
            ibdxsec,
            kw_inputs=["costheta"] + inputs_ibd,
            merge_inputs=merge_inputs,
            outputs_pos=True,
        )
        self._add_node(
            eetoenu,
            kw_inputs=[self._eename, "costheta"] + inputs_common,
            merge_inputs=merge_inputs,
            kw_outputs={"result": "enu"},
        )
        self._add_node(
            jacobian,
            kw_inputs=["enu", self._eename, "costheta"] + inputs_common[:-1],
            merge_inputs=merge_inputs[:-1],
            kw_outputs={"result": "jacobian"},
        )
        self.inputs.make_positionals(self._eename, "costheta")

    @classmethod
    def replicate(
        cls,
        names: Mapping[str, str] = {
            "ibd": "crosssection",
            "enu": "enu",
            "jacobian": "jacobian",
        },
        path: KeyLike = "ibd",
        *args,
        **kwargs,
    ) -> tuple[IBDXsecVBO1Group, NodeStorage]:
        from dagflow.storage import NodeStorage

        path = strkey(path)
        name_ibd = strkey((path, names.get("ibd", "crosssection")))
        name_enu = strkey((path, names.get("enu", "enu")))
        name_jacobian = strkey((path, names.get("jacobian", "jacobian")))

        storage = NodeStorage(default_containers=True)
        nodes = storage.child("nodes")
        inputs = storage.child("inputs")
        outputs = storage.child("outputs")

        ibd = cls(name_ibd, name_enu, name_jacobian, *args, **kwargs)

        nodes[name_ibd] = ibd
        inputs[name_ibd, ibd._eename] = ibd.inputs[ibd._eename]
        inputs[name_ibd, "costheta"] = ibd.inputs["costheta"]
        outputs[name_ibd] = ibd.outputs["result"]
        outputs[name_enu] = ibd.outputs["enu"]
        outputs[name_jacobian] = ibd.outputs["jacobian"]

        NodeStorage.update_current(storage, strict=True)

        return ibd, storage
