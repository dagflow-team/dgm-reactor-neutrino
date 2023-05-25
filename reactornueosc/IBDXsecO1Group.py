from .IBDXsecO1 import IBDXsecO1
from .EeToEnu import EeToEnu
from .Jacobian_dEnu_dEe import Jacobian_dEnu_dEe

from dagflow.meta_node import MetaNode

def IBDXsecO1Group(*, use_edep: bool=False, labels: dict={}):
    ibdxsec = IBDXsecO1('ibd', label=labels.get('xsec', {}))
    eetoenu = EeToEnu('Enu', use_edep=use_edep, label=labels.get('enu', {}))
    jacobian = Jacobian_dEnu_dEe('dEν/dEe', use_edep=use_edep, label=labels.get('jacobian', {}))

    eetoenu.outputs['result'] >> (jacobian.inputs['enu'], ibdxsec.inputs['enu'])

    eename = use_edep and 'edep' or 'ee'
    inputs_common = ['ElectronMass', 'ProtonMass', 'NeutronMass']
    inputs_ibd = inputs_common+[ 'NeutronLifeTime', 'PhaseSpaceFactor', 'g', 'f', 'f2' ]
    merge_inputs = [eename, 'costheta']+inputs_common
    ibd = MetaNode()
    ibd._add_node(
        ibdxsec,
        kw_inputs=['costheta']+inputs_ibd,
        merge_inputs=merge_inputs,
        outputs_pos=True
    )
    ibd._add_node(
        eetoenu,
        kw_inputs=[eename, 'costheta']+inputs_common,
        merge_inputs=merge_inputs,
        kw_outputs={'result': 'enu'}
    )
    ibd._add_node(
        jacobian,
        kw_inputs=['enu', eename, 'costheta']+inputs_common[:-1],
        merge_inputs=merge_inputs[:-1],
        kw_outputs={'result': 'jacobian'}
    )
    ibd.inputs.make_positionals(eename, 'costheta')

    return ibd

