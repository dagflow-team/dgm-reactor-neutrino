from __future__ import annotations

from typing import TYPE_CHECKING, override

from dag_modelling.core.global_parameters import NUMBA_CACHE_ENABLE
from dag_modelling.core.node import Node
from dag_modelling.core.storage import NodeStorage
from dag_modelling.core.type_functions import (
    assign_axes_from_inputs_to_outputs,
    check_shape_of_inputs,
    copy_from_inputs_to_outputs,
)
from numba import njit
from numpy import array, cos, exp, pi, sin, sqrt
from scipy.constants import value

if TYPE_CHECKING:
    from typing import Literal

    from dag_modelling.core.node import Node
    from dag_modelling.core.output import Output
    from nested_mapping.typing import KeyLike
    from numpy import double
    from numpy.typing import NDArray

_surprobArgConversion = pi * 2e-3 * value("electron volt-inverse meter relationship")

# Damping model codes (numba-friendly integers)
_DAMPING_NONE = 0
_DAMPING_DECAY = 1
_DAMPING_WPD1 = 2
_DAMPING_WPD2 = 3

_DAMPING_MODEL_MAP = {
    "none": _DAMPING_NONE,
    "decay": _DAMPING_DECAY,
    "wpd1": _DAMPING_WPD1,
    "wpd2": _DAMPING_WPD2,
}


# ---------------------------------------------------------------------------
# Numba kernel
#
# Standard three-flavor P(ν̄ₑ → ν̄ₑ) with damping factors applied to the
# interference terms:
#
#   P = 1 − ½[cos⁴θ₁₃ sin²2θ₁₂ + sin²2θ₁₃]
#       + ½ cos⁴θ₁₃ sin²2θ₁₂ · D₂₁ · cos(2Δ₂₁)
#       + ½ sin²2θ₁₃ · [D₃₁ cos²θ₁₂ cos(2Δ₃₁) + D₃₂ sin²θ₁₂ cos(2Δ₃₂)]
#       + sin⁴θ₁₃ (e^{−αL/E} − 1)          [ν₃ decay only]
#
#   ν₃ decay : D₃₁ = D₃₂ = exp(−αL/2E),  D₂₁ = 1,  rate-suppressing
#   WPD I    : D_ij = exp(−α (Δm²_ij)² L² / E⁴),   rate-preserving
#   WPD II   : D_ij = exp(−α (Δm²_ij)² L² / E²),   rate-preserving
# ---------------------------------------------------------------------------
@njit(cache=NUMBA_CACHE_ENABLE)
def _sur_prob(
    out: NDArray[double],
    E: NDArray[double],
    L: NDArray[double],
    L_scale: float,
    SinSq2Theta12: NDArray[double],
    SinSq2Theta13: NDArray[double],
    DeltaMSq21: NDArray[double],
    DeltaMSq3lAbs: NDArray[double],
    is_dm32_leading: float,
    nmo: NDArray[double],
    surprobArgConversion: NDArray[double],
    damping_model: NDArray[double],
    alpha_decay: NDArray[double],
    alpha_wpd1: NDArray[double],
    alpha_wpd2: NDArray[double],
) -> None:
    _DeltaMSq21 = DeltaMSq21[0]
    if is_dm32_leading:
        _DeltaMSq32 = nmo[0] * DeltaMSq3lAbs[0]
        _DeltaMSq31 = _DeltaMSq32 + _DeltaMSq21
    else:
        _DeltaMSq31 = nmo[0] * DeltaMSq3lAbs[0]
        _DeltaMSq32 = _DeltaMSq31 - _DeltaMSq21

    # structure factors from central angles
    _SinSq2Theta13 = SinSq2Theta13[0]
    _SinSq2Theta12 = SinSq2Theta12[0]
    _SinSqTheta12 = 0.5 * (1 - sqrt(1 - _SinSq2Theta12))
    _CosSqTheta12 = 1.0 - _SinSqTheta12
    _CosSqTheta13 = 1 - 0.5 * (1 - sqrt(1 - _SinSq2Theta13))
    _CosQuTheta13 = _CosSqTheta13 * _CosSqTheta13
    _SinSqTheta13 = 0.5 * (1 - sqrt(1 - _SinSq2Theta13))
    _SinQuTheta13 = _SinSqTheta13 * _SinSqTheta13

    _model = int(damping_model[0])
    _alpha_decay = alpha_decay[0]
    _alpha_wpd1 = alpha_wpd1[0]
    _alpha_wpd2 = alpha_wpd2[0]

    sinCommonArg = surprobArgConversion[0] * L[0] * 0.25 * L_scale
    L_km = L[0] * L_scale
    L_m = L_km * 1000.0

    for i in range(len(out)):
        L4E = sinCommonArg / E[i]

        Cos2d21 = cos(2.0 * _DeltaMSq21 * L4E)
        Cos2d31 = cos(2.0 * _DeltaMSq31 * L4E)
        Cos2d32 = cos(2.0 * _DeltaMSq32 * L4E)

        D21 = 1.0
        D31 = 1.0
        D32 = 1.0
        correction = 0.0

        if _model == _DAMPING_DECAY:
            _term1 = _alpha_decay * L_m / (2.0 * E[i])
            _term2 = _alpha_decay * L_m / E[i]
            D31 = exp(-_term1)
            D32 = exp(-_term1)
            correction = _SinQuTheta13 * (exp(-_term2) - 1.0)

        elif _model == _DAMPING_WPD1:
            E4 = E[i] * E[i] * E[i] * E[i]
            L2 = L_km * L_km
            D21 = exp(-_alpha_wpd1 * _DeltaMSq21 * _DeltaMSq21 * L2 / E4)
            D31 = exp(-_alpha_wpd1 * _DeltaMSq31 * _DeltaMSq31 * L2 / E4)
            D32 = exp(-_alpha_wpd1 * _DeltaMSq32 * _DeltaMSq32 * L2 / E4)

        elif _model == _DAMPING_WPD2:
            E2 = E[i] * E[i]
            L2 = L_km * L_km
            D21 = exp(-_alpha_wpd2 * _DeltaMSq21 * _DeltaMSq21 * L2 / E2)
            D31 = exp(-_alpha_wpd2 * _DeltaMSq31 * _DeltaMSq31 * L2 / E2)
            D32 = exp(-_alpha_wpd2 * _DeltaMSq32 * _DeltaMSq32 * L2 / E2)

        out[i] = (
            1.0
            - 0.5 * (_CosQuTheta13 * _SinSq2Theta12 + _SinSq2Theta13)
            + 0.5 * _CosQuTheta13 * _SinSq2Theta12 * D21 * Cos2d21
            + 0.5 * _SinSq2Theta13
            * (D31 * _CosSqTheta12 * Cos2d31 + D32 * _SinSqTheta12 * Cos2d32)
            + correction
        )


class NueSurvivalProbabilityDamping(Node):
    """Electron antineutrino survival probability in vacuum with damping.

    Implements the standard three-flavor P(ν̄ₑ → ν̄ₑ) modulated by one of three
    damping signatures from the JUNO sensitivity paper (JHEP 06 (2022) 062):

      * ``"decay"`` : ν₃ invisible decay (rate suppression)
      * ``"wpd1"``  : Wave Packet Decoherence I (rate preserving)
      * ``"wpd2"``  : Wave Packet Decoherence II (rate preserving)

    inputs:
        `E`: array of the energies [MeV]
        `L`: the distance [km] (or [m] with distance_unit="m")
        `SinSq2Theta12`: sin²2θ₁₂ (central value)
        `SinSq2Theta13`: sin²2θ₁₃ (central value)
        `DeltaMSq21`: Δm²₂₁
        `DeltaMSq32` or `DeltaMSq31`: |Δm²₃₂| or |Δm²₃₁| depending on
            `leading_mass_splitting_3l_name`
        `nmo`: +1 (NO) or −1 (IO) mass-ordering constant

    optional inputs:
        `surprobArgConversion`: convert Δm²[eV²]·L[km]/E[MeV] to natural units.
            Default: 2π·10⁻³·(eV ↔ m⁻¹).
        `damping_model`: 0=none, 1=decay, 2=wpd1, 3=wpd2.  Default: 0.
        `alpha_decay`: ν₃ decay α = m₃/τ₃ [MeV/m].  Default: 0.
        `alpha_wpd1` : WPD I α = (4√2 σ_x)⁻² [MeV²].  Default: 0.
        `alpha_wpd2` : WPD II dimensionless α.  Default: 0.

    outputs:
        `0` or `result`: array of probabilities

    Notes
    -----
    ``damping_model = 0`` (or ``alpha_* = 0``) reproduces the standard vacuum
    survival probability to machine precision.
    """

    __slots__ = (
        "_baseline_scale", "_result", "_E", "_L",
        "_SinSq2Theta12", "_SinSq2Theta13",
        "_DeltaMSq21", "_DeltaMSq3lAbs",
        "_is_dm32_leading", "_nmo",
        "_surprob_arg_conversion_factor",
        "_damping_model", "_alpha_decay", "_alpha_wpd1", "_alpha_wpd2",
    )

    _baseline_scale: float
    _E: NDArray
    _L: NDArray
    _nmo: NDArray
    _DeltaMSq21: NDArray
    _DeltaMSq3lAbs: NDArray
    _SinSq2Theta12: NDArray
    _SinSq2Theta13: NDArray
    _surprob_arg_conversion_factor: NDArray
    _damping_model: NDArray
    _alpha_decay: NDArray
    _alpha_wpd1: NDArray
    _alpha_wpd2: NDArray
    _result: NDArray

    def __init__(
        self,
        *args,
        leading_mass_splitting_3l_name: Literal["DeltaMSq31", "DeltaMSq32"],
        distance_unit: Literal["km", "m"] = "km",
        **kwargs,
    ):
        if leading_mass_splitting_3l_name not in {"DeltaMSq31", "DeltaMSq32"}:
            raise RuntimeError(
                f"Do not support switch: {leading_mass_splitting_3l_name=}"
            )

        super().__init__(
            *args, **kwargs,
            allowed_kw_inputs=(
                "E", "L", "SinSq2Theta13", "SinSq2Theta12",
                leading_mass_splitting_3l_name,
                "DeltaMSq21", "nmo",
                "surprobArgConversion",
                "damping_model", "alpha_decay", "alpha_wpd1", "alpha_wpd2",
            ),
        )
        self.needs_post_allocate = True
        self._is_dm32_leading = leading_mass_splitting_3l_name == "DeltaMSq32"
        self._labels.setdefault("mark", "P(ee)")

        self._add_pair("E", "result")
        self._add_input("L", positional=False)
        self._add_input("SinSq2Theta12", positional=False)
        self._add_input("SinSq2Theta13", positional=False)
        self._add_input("DeltaMSq21", positional=False)
        self._add_input(leading_mass_splitting_3l_name, positional=False)
        self._add_input("nmo", positional=False)

        try:
            self._baseline_scale = {"km": 1, "m": 1.0e-3}[distance_unit]
        except KeyError as e:
            raise RuntimeError(f"Invalid distance unit {distance_unit}") from e

    @override
    def _type_function(self) -> None:
        check_shape_of_inputs(
            self,
            (
                "L", "SinSq2Theta12", "SinSq2Theta13",
                "DeltaMSq21",
                "DeltaMSq32" if self._is_dm32_leading else "DeltaMSq31",
                "nmo",
            ),
            (1,),
        )
        for opt_name in (
            "surprobArgConversion",
            "damping_model", "alpha_decay", "alpha_wpd1", "alpha_wpd2",
        ):
            if self.inputs.get(opt_name) is not None:
                check_shape_of_inputs(self, (opt_name,), (1,))
        copy_from_inputs_to_outputs(self, "E", "result")
        assign_axes_from_inputs_to_outputs(
            self, "E", "result", assign_meshes=True, overwrite_assigned=True
        )

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        _sur_prob(
            self._result,
            self._E,
            self._L,
            self._baseline_scale,
            self._SinSq2Theta12,
            self._SinSq2Theta13,
            self._DeltaMSq21,
            self._DeltaMSq3lAbs,
            self._is_dm32_leading,
            self._nmo,
            self._surprob_arg_conversion_factor,
            self._damping_model,
            self._alpha_decay,
            self._alpha_wpd1,
            self._alpha_wpd2,
        )

    @override
    def _post_allocate(self):
        super()._post_allocate()

        self._result = self.outputs["result"]._data.ravel()
        self._E = self.inputs["E"]._data.ravel()
        self._L = self.inputs["L"]._data
        self._SinSq2Theta12 = self.inputs["SinSq2Theta12"]._data
        self._SinSq2Theta13 = self.inputs["SinSq2Theta13"]._data
        self._DeltaMSq21 = self.inputs["DeltaMSq21"]._data
        self._DeltaMSq3lAbs = self.inputs[
            "DeltaMSq32" if self._is_dm32_leading else "DeltaMSq31"
        ]._data
        self._nmo = self.inputs["nmo"]._data

        self._surprob_arg_conversion_factor = (
            cin._data if (cin := self.inputs.get("surprobArgConversion"))
            else array([_surprobArgConversion])
        )
        self._damping_model = (
            cin._data if (cin := self.inputs.get("damping_model"))
            else array([float(_DAMPING_NONE)])
        )
        self._alpha_decay = (
            cin._data if (cin := self.inputs.get("alpha_decay"))
            else array([0.0])
        )
        self._alpha_wpd1 = (
            cin._data if (cin := self.inputs.get("alpha_wpd1"))
            else array([0.0])
        )
        self._alpha_wpd2 = (
            cin._data if (cin := self.inputs.get("alpha_wpd2"))
            else array([0.0])
        )

    @classmethod
    def replicate(
        cls, *args, name: str,
        replicate_outputs: tuple[KeyLike, ...] = ((),),
        surprobArgConversion: Output | Literal[True] | None = None,
        verbose: bool = False, **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        storage = NodeStorage()
        nodes = storage.create_child("nodes")
        inputs = storage.create_child("inputs")
        outputs = storage.create_child("outputs")

        nametuple = tuple(name.split("."))
        for key in replicate_outputs:
            ckey = nametuple + (key,) if isinstance(key, str) else nametuple + key
            cname = ".".join(ckey)
            surprob = cls(cname, *args, **kwargs)
            nodes[ckey] = surprob
            inputs[nametuple + ("enu",) + key] = surprob.inputs[0]
            inputs[nametuple + ("L",) + key] = surprob.inputs["L"]
            outputs[ckey] = surprob.outputs[0]

            if surprobArgConversion:
                if surprobArgConversion is True:
                    inputs[nametuple + ("surprobArgConversion",) + key] = surprob(
                        "surprobArgConversion"
                    )
                else:
                    surprobArgConversion >> surprob("surprobArgConversion")

        NodeStorage.update_current(storage, strict=True, verbose=verbose)
        return None, storage