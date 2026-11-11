from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.plot.plot import plot_auto
from matplotlib.pyplot import subplots
from numpy import allclose, finfo, geomspace, sin, sqrt
from pytest import mark
from scipy.constants import value

from dgm_reactor_neutrino.nue_survival_probability_khan import (
    NueSurvivalProbabilityKhan,
    _surprobArgConversion,
)


@mark.parametrize("nmo", (1, -1))  # mass ordering
@mark.parametrize(
    "L,leading_mass_splitting_3l_name", ((2, "DeltaMSq32"), (52, "DeltaMSq31"))
)  # km
@mark.parametrize(
    "conversionFactor",
    (None, _surprobArgConversion),
)
@mark.parametrize("density", (0, 2.5))
@mark.parametrize("mode", ("expansion", "main"))
def test_NueSurvivalProbabilityKhan(
    debug_graph,
    test_name,
    L,
    leading_mass_splitting_3l_name,
    nmo,
    conversionFactor,
    density,
    mode,
    output_path: str,
) -> None:
    E = geomspace(1, 10, 1000)  # MeV
    DeltaMSq21 = 7.39 * 1e-5  # eV^2
    DeltaMSq3lAbs = 2.45 * 1e-3  # eV^2
    SinSq2Theta12 = 3.1 * 1e-1  # [-]
    SinSq2Theta13 = 2.241 * 1e-2  # [-]
    ElectronToNucleonFraction = 0.5
    is_dm32_leading = leading_mass_splitting_3l_name == "DeltaMSq32"
    is_dm31_leading = leading_mass_splitting_3l_name == "DeltaMSq31"

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        surprob = NueSurvivalProbabilityKhan(
            "P(ee)",
            leading_mass_splitting_3l_name=leading_mass_splitting_3l_name,
            mode=mode,
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
        Array("ElectronToNucleonFraction", [ElectronToNucleonFraction], mode="fill") >> surprob(
            "ElectronToNucleonFraction"
        )
        Array("matter_density", [density], mode="fill") >> surprob("matter_density")
        if conversionFactor is not None:
            (
                in_conversion := Array("surprobArgConversion", [conversionFactor], mode="fill")
            ) >> surprob("surprobArgConversion")
        else:
            in_conversion = None

    if conversionFactor is None:
        conversionFactor = _surprobArgConversion

    def surprob_fcn() -> float:
        aem_g = value("atomic mass constant") * 1000
        Gf = value("Fermi coupling constant")
        hbar = value("reduced Planck constant")
        c = value("speed of light in vacuum")
        e = value("elementary charge")
        hbar_c_eV_cm = (hbar * c / e) * 100
        SqGeV_to_SqMeV = 1e-18
        GF_CONVERSION = SqGeV_to_SqMeV * (hbar_c_eV_cm**3)
        MeV_to_eV = 1e-6

        Lcq = L * conversionFactor / 4.0
        _DeltaMSq31 = nmo * DeltaMSq3lAbs + DeltaMSq21 * is_dm32_leading
        _DeltaMSq32 = nmo * DeltaMSq3lAbs - DeltaMSq21 * is_dm31_leading
        _DeltaMSq31 = (
            nmo * DeltaMSq3lAbs + DeltaMSq21 * is_dm32_leading
        )  # Δm²₃₁ = α*|Δm²₃ₗ| + β*Δm²₂₁
        _DeltaMSq32 = (
            nmo * DeltaMSq3lAbs - DeltaMSq21 * is_dm31_leading
        )  # Δm²₃₂ = α*|Δm²₃ₗ| - β*Δm²₂₁

        _SinSqTheta12 = 0.5 * (1.0 - sqrt(1.0 - SinSq2Theta12))
        _CosSqTheta12 = 1.0 - _SinSqTheta12
        _SinSqTheta13 = 0.5 * (1.0 - sqrt(1.0 - SinSq2Theta13))
        _CosSqTheta13 = 1.0 - _SinSqTheta13

        _DeltaMSqee = _CosSqTheta12 * _DeltaMSq31 + _SinSqTheta12 * _DeltaMSq32

        _density_const = (
            ElectronToNucleonFraction / aem_g * (-2.0) * sqrt(2.0) * Gf * GF_CONVERSION / MeV_to_eV
        )
        _msw_factor = _density_const * density
        _a = _msw_factor * E
        _Cos2Theta12 = sqrt(1.0 - SinSq2Theta12)
        if mode == "expansion":
            a_dm21 = _a / DeltaMSq21
            a_dmee = _a / _DeltaMSqee

            common_21 = _CosSqTheta13 * a_dm21
            common_21_squared = common_21 * common_21

            SinSq13_mod = _SinSqTheta13 + (2.0 * _SinSqTheta13 * _CosSqTheta13) * a_dmee
            CosSq13_mod = 1.0 - SinSq13_mod
            CosQu13_mod = CosSq13_mod * CosSq13_mod

            SinSq12_mod = (
                _SinSqTheta12
                + 0.5 * SinSq2Theta12 * common_21
                + 0.75 * SinSq2Theta12 * _Cos2Theta12 * common_21_squared
            )

            CosSq12_mod = 1.0 - SinSq12_mod

            DeltaMSq21_mod = DeltaMSq21 * (
                1.0 - _Cos2Theta12 * common_21 + 0.5 * SinSq2Theta12 * common_21_squared
            )

            common_3 = 0.25 * SinSq2Theta12 * _CosSqTheta13 * common_21

            DeltaMSq31_mod = _DeltaMSq31 - _a * (
                (_CosSqTheta12 * _CosSqTheta13 - _SinSqTheta13) - common_3
            )

            DeltaMSq32_mod = _DeltaMSq32 - _a * (
                (_SinSqTheta12 * _CosSqTheta13 - _SinSqTheta13) + common_3
            )

            SinArg21 = Lcq * (DeltaMSq21_mod / E)
            SinArg31 = Lcq * (DeltaMSq31_mod / E)
            SinArg32 = Lcq * (DeltaMSq32_mod / E)

            SinSqDelta31_mod = sin(SinArg31) ** 2
            SinSqDelta21_mod = sin(SinArg21) ** 2
            SinSqDelta32_mod = sin(SinArg32) ** 2

            res = (
                1.0
                - 4.0 * (CosQu13_mod * SinSq12_mod * CosSq12_mod * SinSqDelta21_mod)
                - 4.0
                * (
                    SinSq13_mod
                    * CosSq13_mod
                    * (CosSq12_mod * SinSqDelta31_mod + SinSq12_mod * SinSqDelta32_mod)
                )
            )
        elif mode == "main":
            _Cos2Theta13 = sqrt(1.0 - SinSq2Theta13)

            a_dmee = _a / _DeltaMSqee
            _shifted_Cos2Theta13 = _Cos2Theta13 - a_dmee
            _DeltaMSqee_mod = _DeltaMSqee * sqrt(
                _shifted_Cos2Theta13 * _shifted_Cos2Theta13 + SinSq2Theta13
            )

            _Cos2Theta13_mod = (_DeltaMSqee * _Cos2Theta13 - _a) / _DeltaMSqee_mod
            _CosSq2Theta13_mod = _Cos2Theta13_mod * _Cos2Theta13_mod
            _SinSq2Theta13_mod = 1.0 - _CosSq2Theta13_mod
            _CosSqTheta13_mod = (1.0 + _Cos2Theta13_mod) * 0.5
            _CosQuTheta13_mod = _CosSqTheta13_mod * _CosSqTheta13_mod

            a_prime = (_a + _DeltaMSqee - _DeltaMSqee_mod) * 0.5
            ap_dm21 = a_prime / DeltaMSq21
            _shifted_Cos2Theta12 = _Cos2Theta12 - ap_dm21

            CosSqDiff_Theta13Mod_Theta13 = (
                0.5 * (_DeltaMSqee_mod + _DeltaMSqee - _a * _Cos2Theta13) / _DeltaMSqee_mod
            )
            _DeltaMSq21_mod = DeltaMSq21 * sqrt(
                _shifted_Cos2Theta12 * _shifted_Cos2Theta12
                + CosSqDiff_Theta13Mod_Theta13 * SinSq2Theta12
            )

            _Cos2Theta12_mod = (DeltaMSq21 * _Cos2Theta12 - a_prime) / _DeltaMSq21_mod
            _CosSq2Theta12_mod = _Cos2Theta12_mod * _Cos2Theta12_mod
            _SinSq2Theta12_mod = 1.0 - _CosSq2Theta12_mod
            _SinSqTheta12_mod = (1.0 - _Cos2Theta12_mod) * 0.5
            _CosSqTheta12_mod = 1.0 - _SinSqTheta12_mod

            _DeltaMSq31_mod = _DeltaMSqee_mod + _SinSqTheta12_mod * _DeltaMSq21_mod
            _DeltaMSq32_mod = _DeltaMSq31_mod - _DeltaMSq21_mod

            SinArg21 = Lcq * (_DeltaMSq21_mod / E)
            SinArg31 = Lcq * (_DeltaMSq31_mod / E)
            SinArg32 = Lcq * (_DeltaMSq32_mod / E)

            SinSqDelta31_mod = sin(SinArg31) ** 2
            SinSqDelta21_mod = sin(SinArg21) ** 2
            SinSqDelta32_mod = sin(SinArg32) ** 2

            res = (
                1.0
                - _CosQuTheta13_mod * _SinSq2Theta12_mod * SinSqDelta21_mod
                - _SinSq2Theta13_mod
                * (_CosSqTheta12_mod * SinSqDelta31_mod + _SinSqTheta12_mod * SinSqDelta32_mod)
            )
        else:
            raise ValueError("Not corrected mode")
        return res

    atol = finfo("d").resolution * 200
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

    nmo *= -1
    in_nmo.outputs[0].set(nmo)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    DeltaMSq21 *= 1.1
    in_Dm21.outputs[0].set(DeltaMSq21)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    DeltaMSq3lAbs *= 0.9
    in_Dm3l.outputs[0].set(DeltaMSq3lAbs)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    SinSq2Theta12 *= 1.2
    in_t12.outputs[0].set(SinSq2Theta12)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    SinSq2Theta13 += 0.1
    in_t13.outputs[0].set(SinSq2Theta13)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    L *= 10
    in_L.outputs[0].set(L)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    E *= 15
    in_E.outputs[0].set(E)
    assert surprob.tainted is True
    res = surprob_fcn()
    assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
    assert surprob.tainted is False

    if in_conversion is not None:
        conversionFactor *= 1.01
        in_conversion.outputs[0].set(conversionFactor)
        assert surprob.tainted is True
        res = surprob_fcn()
        assert allclose(surprob.outputs[0].data, res, rtol=0, atol=atol)
        assert surprob.tainted is False

    savegraph(graph, f"{output_path}/{test_name}.png")


@mark.parametrize(
    "nmo, L,leading_mass_splitting_3l_name", ((-1, 2, "DeltaMSq32"), (-1, 52, "DeltaMSq31"), (1, 180, "DeltaMSq32"))
)  # km
@mark.parametrize("mode", ("expansion", "main"))
def test_crosscheck_NueSurvivalProbability_NueSurvivalProbabilityKhan(
    debug_graph,
    test_name,
    L,
    leading_mass_splitting_3l_name,
    nmo,
    mode,
    output_path: str,
):
    E = geomspace(1, 100, 1000)  # MeV
    DeltaMSq21 = 7.39 * 1e-5  # eV^2
    DeltaMSq3lAbs = 2.45 * 1e-3  # eV^2
    SinSq2Theta12 = 3.1 * 1e-1  # [-]
    SinSq2Theta13 = 2.241 * 1e-2  # [-]
    ElectronToNucleonFraction = 0.5
    density = 0.0
    conversionFactor = _surprobArgConversion

    with Graph(close_on_exit=True, debug=debug_graph):
        surprobKhan = NueSurvivalProbabilityKhan(
            "P(ee)", leading_mass_splitting_3l_name=leading_mass_splitting_3l_name, mode=mode
        )
        Array("E", E, mode="fill") >> surprobKhan("E")
        Array("L", [L], mode="fill") >> surprobKhan("L")
        Array("nmo", [nmo], mode="fill") >> surprobKhan("nmo")
        Array("DeltaMSq21", [DeltaMSq21], mode="fill") >> surprobKhan("DeltaMSq21")
        Array(leading_mass_splitting_3l_name, [DeltaMSq3lAbs], mode="fill") >> surprobKhan(
            leading_mass_splitting_3l_name
        )
        Array("SinSq2Theta12", [SinSq2Theta12], mode="fill") >> surprobKhan("SinSq2Theta12")
        Array("SinSq2Theta13", [SinSq2Theta13], mode="fill") >> surprobKhan("SinSq2Theta13")
        Array("ElectronToNucleonFraction", [ElectronToNucleonFraction], mode="fill") >> surprobKhan(
            "ElectronToNucleonFraction"
        )
        Array("matter_density", [density], mode="fill") >> surprobKhan("matter_density")
        if conversionFactor is not None:
            (Array("surprobArgConversion", [conversionFactor], mode="fill")) >> surprobKhan(
                "surprobArgConversion"
            )

    from dgm_reactor_neutrino.nue_survival_probability import NueSurvivalProbability

    with Graph(close_on_exit=True, debug=debug_graph):
        surprob = NueSurvivalProbability(
            "P(ee)",
            leading_mass_splitting_3l_name=leading_mass_splitting_3l_name,
        )
        Array("E", E, mode="fill") >> surprob("E")
        Array("L", [L], mode="fill") >> surprob("L")
        Array("nmo", [nmo], mode="fill") >> surprob("nmo")
        Array("DeltaMSq21", [DeltaMSq21], mode="fill") >> surprob("DeltaMSq21")
        Array(leading_mass_splitting_3l_name, [DeltaMSq3lAbs], mode="fill") >> surprob(
            leading_mass_splitting_3l_name
        )
        Array("SinSq2Theta12", [SinSq2Theta12], mode="fill") >> surprob("SinSq2Theta12")
        Array("SinSq2Theta13", [SinSq2Theta13], mode="fill") >> surprob("SinSq2Theta13")
        if conversionFactor is not None:
            (Array("surprobArgConversion", [conversionFactor], mode="fill")) >> surprob(
                "surprobArgConversion"
            )

    atol = finfo("d").resolution * 4
    assert allclose(surprob.outputs[0].data, surprobKhan.outputs[0].data, rtol=0, atol=atol)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, height_ratios=[2, 1, 1], sharex=True)

    axs[0].plot(E, surprob.outputs[0].data, label="common")
    axs[0].plot(E, surprobKhan.outputs[0].data, label="Khan")
    axs[1].plot(E, surprobKhan.outputs[0].data / surprob.outputs[0].data - 1)
    axs[2].plot(E, surprobKhan.outputs[0].data - surprob.outputs[0].data)
    axs[0].set_ylabel("Entries")
    axs[1].set_xlabel("E, MeV")
    axs[1].set_ylabel("ratio - 1")
    axs[1].set_xlabel("E, MeV")
    axs[2].set_ylabel("difference")
    axs[0].legend(loc="upper right")
    plt.subplots_adjust(hspace=0)
    plt.savefig(f"{output_path}/{test_name}.pdf")
    plt.close(fig)


@mark.parametrize(
    "nmo, L,leading_mass_splitting_3l_name", ((-1, 2, "DeltaMSq32"), (-1, 52, "DeltaMSq31"), (1, 180, "DeltaMSq32"))
)  # km
@mark.parametrize(
    "conversionFactor",
    (None, _surprobArgConversion),
)
def test_crosscheck_NueSurvivalProbabilityKhan(
    debug_graph,
    test_name,
    L,
    leading_mass_splitting_3l_name,
    nmo,
    conversionFactor,
    output_path: str,
):
    E = geomspace(1, 12, 120)  # MeV
    DeltaMSq21 = 7.39 * 1e-5  # eV^2
    DeltaMSq3lAbs = 2.45 * 1e-3  # eV^2
    SinSq2Theta12 = 3.1 * 1e-1  # [-]
    SinSq2Theta13 = 2.241 * 1e-2  # [-]
    ElectronToNucleonFraction = 0.5
    density = 2.5

    with Graph(close_on_exit=True, debug=debug_graph):
        surprobMain = NueSurvivalProbabilityKhan(
            "P(ee)",
            leading_mass_splitting_3l_name=leading_mass_splitting_3l_name,
            mode="main",
        )
        Array("E", E, mode="fill") >> surprobMain("E")
        Array("L", [L], mode="fill") >> surprobMain("L")
        Array("nmo", [nmo], mode="fill") >> surprobMain("nmo")
        Array("DeltaMSq21", [DeltaMSq21], mode="fill") >> surprobMain("DeltaMSq21")
        Array(leading_mass_splitting_3l_name, [DeltaMSq3lAbs], mode="fill") >> surprobMain(
            leading_mass_splitting_3l_name
        )
        Array("SinSq2Theta12", [SinSq2Theta12], mode="fill") >> surprobMain("SinSq2Theta12")
        Array("SinSq2Theta13", [SinSq2Theta13], mode="fill") >> surprobMain("SinSq2Theta13")
        Array("ElectronToNucleonFraction", [ElectronToNucleonFraction], mode="fill") >> surprobMain(
            "ElectronToNucleonFraction"
        )
        Array("matter_density", [density], mode="fill") >> surprobMain("matter_density")
        if conversionFactor is not None:
            (Array("surprobArgConversion", [conversionFactor], mode="fill")) >> surprobMain(
                "surprobArgConversion"
            )

    with Graph(close_on_exit=True, debug=debug_graph):
        surprobExpansion = NueSurvivalProbabilityKhan(
            "P(ee)",
            leading_mass_splitting_3l_name=leading_mass_splitting_3l_name,
            mode="expansion",
        )
        Array("E", E, mode="fill") >> surprobExpansion("E")
        Array("L", [L], mode="fill") >> surprobExpansion("L")
        Array("nmo", [nmo], mode="fill") >> surprobExpansion("nmo")
        Array("DeltaMSq21", [DeltaMSq21], mode="fill") >> surprobExpansion("DeltaMSq21")
        Array(leading_mass_splitting_3l_name, [DeltaMSq3lAbs], mode="fill") >> surprobExpansion(
            leading_mass_splitting_3l_name
        )
        Array("SinSq2Theta12", [SinSq2Theta12], mode="fill") >> surprobExpansion("SinSq2Theta12")
        Array("SinSq2Theta13", [SinSq2Theta13], mode="fill") >> surprobExpansion("SinSq2Theta13")
        Array(
            "ElectronToNucleonFraction", [ElectronToNucleonFraction], mode="fill"
        ) >> surprobExpansion("ElectronToNucleonFraction")
        Array("matter_density", [density], mode="fill") >> surprobExpansion("matter_density")
        if conversionFactor is not None:
            (Array("surprobArgConversion", [conversionFactor], mode="fill")) >> surprobExpansion(
                "surprobArgConversion"
            )

    atol = finfo("d").resolution
    rtol = 1e-4
    assert allclose(
        surprobExpansion.outputs[0].data, surprobMain.outputs[0].data, rtol=rtol, atol=atol
    )

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, height_ratios=[2, 1, 1], sharex=True)

    axs[0].plot(E, surprobExpansion.outputs[0].data, label="expansion")
    axs[0].plot(E, surprobMain.outputs[0].data, label="exact")
    axs[1].plot(E, surprobMain.outputs[0].data / surprobExpansion.outputs[0].data - 1)
    axs[2].plot(E, surprobMain.outputs[0].data - surprobExpansion.outputs[0].data)
    axs[0].set_ylabel("Entries")
    axs[1].set_xlabel("E, MeV")
    axs[1].set_ylabel("ratio - 1")
    axs[1].set_xlabel("E, MeV")
    axs[2].set_ylabel("difference")
    axs[0].legend(loc="upper right")
    plt.subplots_adjust(hspace=0)
    plt.savefig(f"{output_path}/{test_name}.pdf")
    plt.close(fig)
