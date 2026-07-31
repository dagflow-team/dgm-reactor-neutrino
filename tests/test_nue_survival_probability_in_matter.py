from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.plot.plot import plot_auto
from matplotlib.pyplot import subplots
from numpy import allclose, arcsin, cos, finfo, geomspace, sin, sqrt
from scipy.constants import value
from pytest import mark

from dgm_reactor_neutrino import NueSurvivalProbabilityMatter
from dgm_reactor_neutrino.nue_survival_probability import _surprobArgConversion

aem_g = value('atomic mass constant') * 1000
Gf = value('Fermi coupling constant')

@mark.parametrize("nmo", (1, -1))  # mass ordering
@mark.parametrize(
    "L,leading_mass_splitting_3l_name", ((2, "DeltaMSq32"), (52, "DeltaMSq31"), (180, "DeltaMSq32"))
)  # km
@mark.parametrize(
    "conversionFactor",
    (None, _surprobArgConversion, 0.9 * _surprobArgConversion),
)
@mark.parametrize("density", (0, 2.5))
def test_NueSurvivalProbabilityMatter(
    debug_graph,
    test_name,
    L,
    leading_mass_splitting_3l_name,
    nmo,
    conversionFactor,
    density,
    output_path: str,
) -> None:
    E = geomspace(1, 100, 1000)  # MeV
    DeltaMSq21 = 7.39 * 1e-5  # eV^2
    DeltaMSq3lAbs = 2.45 * 1e-3  # eV^2
    SinSq2Theta12 = 3.1 * 1e-1  # [-]
    SinSq2Theta13 = 2.241 * 1e-2  # [-]
    ElectronFraction = 0.5
    is_dm32_leading = leading_mass_splitting_3l_name == "DeltaMSq32"
    is_dm31_leading = leading_mass_splitting_3l_name == "DeltaMSq31"

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        surprob = NueSurvivalProbabilityMatter(
            "P(ee)", leading_mass_splitting_3l_name=leading_mass_splitting_3l_name,
        )
        (in_E := Array("E", E, mode="fill")) >> surprob("E")
        (in_L := Array("L", [L], mode="fill")) >> surprob("L")
        (in_nmo := Array("nmo", [nmo], mode="fill")) >> surprob("nmo")
        (in_Dm21 := Array("DeltaMSq21", [DeltaMSq21], mode="fill")) >> surprob("DeltaMSq21")
        (in_Dm3l := Array(leading_mass_splitting_3l_name, [DeltaMSq3lAbs], mode="fill")) >> surprob(
            leading_mass_splitting_3l_name
        )
        (in_t12 := Array("SinSq2Theta12", [SinSq2Theta12], mode="fill")) >> surprob("SinSq2Theta12")
        (in_t13 := Array("SinSq2Theta13", [SinSq2Theta13], mode="fill")) >> surprob("SinSq2Theta13")
        (in_ef := Array("ElectronFraction", [ElectronFraction], mode="fill")) >> surprob("ElectronFraction")
        (in_dens := Array("density", [density], mode="fill")) >> surprob("density")
        if conversionFactor is not None:
            (
                in_conversion := Array("surprobArgConversion", [conversionFactor], mode="fill")
            ) >> surprob("surprobArgConversion")
        else:
            in_conversion = None

    if conversionFactor is None:
        conversionFactor = _surprobArgConversion

    def surprob_fcn() -> float:
        tmp = L * conversionFactor / 4.0 / E
        _DeltaMSq31 = (
                nmo * DeltaMSq3lAbs + DeltaMSq21 * is_dm32_leading
        )
        _DeltaMSq32 = (
                nmo * DeltaMSq3lAbs - DeltaMSq21 * is_dm31_leading
        )
        _DeltaMSq31 = (
                nmo * DeltaMSq3lAbs + DeltaMSq21 * is_dm32_leading
        )  # Δm²₃₁ = α*|Δm²₃ₗ| + β*Δm²₂₁
        _DeltaMSq32 = (
                nmo * DeltaMSq3lAbs - DeltaMSq21 * is_dm31_leading
        )  # Δm²₃₂ = α*|Δm²₃ₗ| - β*Δm²₂₁
        theta12 = 0.5 * arcsin(sqrt(SinSq2Theta12))
        theta13 = 0.5 * arcsin(sqrt(SinSq2Theta13))
        _SinSqTheta12 = sin(theta12) ** 2  # sin²θ₁₂
        _CosSqTheta12 = cos(theta12) ** 2  # cos²θ₁₂
        _CosSqTheta13 = cos(theta13) ** 2  # cos^2 θ₁₃
        _SinSqTheta13 = 1.0 - _CosSqTheta13  # sin^2 θ₁₃
        _CosQuTheta13 = _CosSqTheta13 ** 2  # cos^4 θ₁₃
        _scSq12 = _SinSqTheta12 * _CosSqTheta12
        _cosDouble12 = _CosSqTheta12 - _SinSqTheta12

        _DeltaMSqee = _CosSqTheta12 * _DeltaMSq31 + _SinSqTheta12 * _DeltaMSq32

        _m_density_const = ElectronFraction / aem_g * (-2.0) * sqrt(2.0) * Gf
        _msw_factor = _m_density_const * density
        _m_a = _msw_factor * E

        m_common_21 = (_CosSqTheta13 / DeltaMSq21) * _m_a
        m_common_21_squared = m_common_21 * m_common_21

        m_sinSq13_mod = _SinSqTheta13 + (2.0 * _SinSqTheta13 * _CosSqTheta13 / _DeltaMSqee) * _m_a

        m_sinSq12_mod = (_SinSqTheta12
                         + 2.0 * _scSq12 * m_common_21
                         + 3.0 * _scSq12 * _cosDouble12 * m_common_21_squared)

        m_dm21_mod = (DeltaMSq21
                      - DeltaMSq21 * _cosDouble12 * m_common_21
                      + 2.0 * DeltaMSq21 * _scSq12 * m_common_21_squared)

        m_common_3 = _scSq12 * _CosSqTheta13 * m_common_21

        m_dm31_mod = (_DeltaMSq31
                      - _m_a * ((_CosSqTheta12 * _CosSqTheta13 - _SinSqTheta13) - m_common_3))

        m_dm32_mod = (_DeltaMSq32
                      - _m_a * ((_SinSqTheta12 * _CosSqTheta13 - _SinSqTheta13) + m_common_3))

        sinArg21 = tmp * (m_dm21_mod / E)
        sinArg31 = tmp * (m_dm31_mod / E)
        sinArg32 = tmp * (m_dm32_mod / E)

        res = (1.0
                - 4.0 * ((1.0 - m_sinSq13_mod) ** 2
                         * m_sinSq12_mod * (1.0 - m_sinSq12_mod)
                         * (sin(sinArg21) ** 2))
                - 4.0 * (m_sinSq13_mod * (1.0 - m_sinSq13_mod)
                         * ((1.0 - m_sinSq12_mod) * (sin(sinArg31) ** 2)
                            + m_sinSq12_mod * (sin(sinArg32) ** 2))))
        return res