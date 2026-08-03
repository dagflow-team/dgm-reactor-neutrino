from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.plot.plot import plot_auto
from matplotlib.pyplot import subplots
from numpy import allclose, arcsin, cos, finfo, geomspace, sin, sqrt
from scipy.constants import value
from pytest import mark

from dgm_reactor_neutrino import NueSurvivalProbabilityKhan
from dgm_reactor_neutrino.nue_survival_probability import _surprobArgConversion

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

    with Graph(close_on_exit=True) as graph:
        surprob = NueSurvivalProbabilityKhan(
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
        aem_g = value('atomic mass constant') * 1000
        Gf = value('Fermi coupling constant')
        hbar = value('reduced Planck constant')
        c = value('speed of light in vacuum')
        e = value('elementary charge')
        hbar_c_eV_cm = (hbar * c / e) * 100
        SqGeV_to_SqMeV = 1e-18
        GF_CONVERSION = SqGeV_to_SqMeV * (hbar_c_eV_cm ** 3)
        MeV_to_eV = 1e-6

        Lcq = L * conversionFactor / 4.0
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

        # if is_dm32_leading:
        #     _DeltaMSq32 = nmo * DeltaMSq3lAbs  # Δm²₃₂ = α*|Δm²₃ₗ|
        #     _DeltaMSq31 = _DeltaMSq32 + DeltaMSq21  # Δm²₃₁ = Δm²₃₂ + Δm²₂₁
        # else:
        #     _DeltaMSq31 = nmo * DeltaMSq3lAbs  # Δm²₃₁ = α*|Δm²₃ₗ|
        #     _DeltaMSq32 = _DeltaMSq31 - DeltaMSq21  # Δm²₃₂ = Δm²₃₁ - Δm²₂₁

        _SinSqTheta12 = 0.5 * (1.0 - sqrt(1.0 - SinSq2Theta12))
        _CosSqTheta12 = 1.0 - _SinSqTheta12
        _SinSqTheta13 = 0.5 * (1.0 - sqrt(1.0 - SinSq2Theta13))
        _CosSqTheta13 = 1.0 - _SinSqTheta13
        _CosQuTheta13 = _CosSqTheta13 ** 2  # cos^4 θ₁₃
        _scSq12 = _SinSqTheta12 * _CosSqTheta12
        _cosDouble12 = _CosSqTheta12 - _SinSqTheta12

        _DeltaMSqee = _CosSqTheta12 * _DeltaMSq31 + _SinSqTheta12 * _DeltaMSq32

        _density_const = (ElectronFraction / aem_g * (-2.0) * sqrt(2.0) * Gf *
                          GF_CONVERSION / MeV_to_eV)
        _msw_factor = _density_const * density
        _a = _msw_factor * E

        common_21 = (_CosSqTheta13 / DeltaMSq21) * _a
        common_21_squared = common_21 * common_21

        sinSq13_mod = _SinSqTheta13 + (2.0 * _SinSqTheta13 * _CosSqTheta13 / _DeltaMSqee) * _a

        sinSq12_mod = (_SinSqTheta12
                         + 2.0 * _scSq12 * common_21
                         + 3.0 * _scSq12 * _cosDouble12 * common_21_squared)

        dm21_mod = (DeltaMSq21
                      - DeltaMSq21 * _cosDouble12 * common_21
                      + 2.0 * DeltaMSq21 * _scSq12 * common_21_squared)

        common_3 = _scSq12 * _CosSqTheta13 * common_21

        dm31_mod = (_DeltaMSq31
                      - _a * ((_CosSqTheta12 * _CosSqTheta13 - _SinSqTheta13) - common_3))

        dm32_mod = (_DeltaMSq32
                      - _a * ((_SinSqTheta12 * _CosSqTheta13 - _SinSqTheta13) + common_3))

        sinArg21 = Lcq * (dm21_mod / E)
        sinArg31 = Lcq * (dm31_mod / E)
        sinArg32 = Lcq * (dm32_mod / E)

        res = (1.0
                - 4.0 * ((1.0 - sinSq13_mod) ** 2
                         * sinSq12_mod * (1.0 - sinSq12_mod)
                         * (sin(sinArg21) ** 2))
                - 4.0 * (sinSq13_mod * (1.0 - sinSq13_mod)
                         * ((1.0 - sinSq12_mod) * (sin(sinArg31) ** 2)
                            + sinSq12_mod * (sin(sinArg32) ** 2))))
        return res

    atol = finfo("d").resolution * 2
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    subplots(1, 1)
    plot_auto(
        surprob,
        filter_kw={"masked_value": 0},
        show=False,
        close=True,
        save=f"{output_path}/{test_name}_plot.pdf",
    )