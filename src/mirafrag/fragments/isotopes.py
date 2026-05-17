from __future__ import annotations

from mirafrag.adducts import PERIODIC_TABLE

_ISOTOPE_MASS_NUMBERS = {
    1: (1, 2),
    6: (12, 13),
    7: (14, 15),
    8: (16, 17, 18),
    9: (19,),
    11: (23,),
    15: (31,),
    16: (32, 33, 34, 36),
    17: (35, 37),
    19: (39, 41),
    35: (79, 81),
    53: (127,),
}
_ELEMENT_ISOTOPE_CACHE: dict[str, list[tuple[float, float]]] = {}
_FORMULA_ISOTOPE_CACHE: dict[
    tuple[tuple[tuple[str, int], ...], float, int],
    list[tuple[float, float]],
] = {}


def _fragment_element_counts(
    mol,
    atom_indices: tuple[int, ...],
    atom_hs: list[int],
    h_shift: int,
) -> dict[str, int]:
    """
    Count elements for a fragment formula after a hydrogen shift.
    """
    counts: dict[str, int] = {}
    hydrogen_count = int(h_shift)
    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
        hydrogen_count += int(atom_hs[int(atom_idx)])
    if hydrogen_count < 0:
        return {}
    if hydrogen_count > 0:
        counts['H'] = counts.get('H', 0) + hydrogen_count
    return {symbol: count for symbol, count in counts.items() if int(count) > 0}


def _formula_isotope_peaks(
    element_counts: dict[str, int],
    *,
    include_isotopes: bool,
    threshold: float,
    max_peaks: int,
) -> list[tuple[float, float]]:
    """
    Return isotope mass/probability peaks for an element-count formula.

    When isotope expansion is disabled, only the monoisotopic mass is returned. Otherwise a truncated convolution of element isotope distributions is cached and normalized.
    """
    if not include_isotopes:
        mass = sum(
            float(count) * float(PERIODIC_TABLE.GetMostCommonIsotopeMass(symbol))
            for symbol, count in element_counts.items()
        )
        return [(mass, 1.0)]

    key = (
        tuple(
            sorted(
                (str(symbol), int(count)) for symbol, count in element_counts.items()
            )
        ),
        float(threshold),
        int(max_peaks),
    )
    cached = _FORMULA_ISOTOPE_CACHE.get(key)
    if cached is not None:
        return cached

    distribution: list[tuple[float, float]] = [(0.0, 1.0)]
    max_states = max(16, int(max_peaks) * 8)
    for symbol, count in sorted(element_counts.items()):
        isotopes = _element_isotopes(symbol)
        for _ in range(int(count)):
            distribution = _convolve_isotope_distribution(
                distribution,
                isotopes,
                threshold=max(float(threshold) * 0.01, 1e-12),
                max_states=max_states,
            )

    peaks = [
        (mass, prob)
        for mass, prob in sorted(distribution, key=lambda item: item[0])
        if float(prob) >= float(threshold)
    ]
    if not peaks:
        peaks = sorted(distribution, key=lambda item: item[1], reverse=True)[:1]
    peaks = sorted(peaks, key=lambda item: item[1], reverse=True)[: int(max_peaks)]
    peaks = sorted(peaks, key=lambda item: item[0])
    total = sum(float(prob) for _mass, prob in peaks)
    if total <= 0.0:
        peaks = [(float(peaks[0][0]), 1.0)]
    else:
        peaks = [(float(mass), float(prob) / total) for mass, prob in peaks]
    _FORMULA_ISOTOPE_CACHE[key] = peaks
    return peaks


def _element_isotopes(symbol: str) -> list[tuple[float, float]]:
    """
    Return normalized isotope masses and probabilities for one element symbol.
    """
    cached = _ELEMENT_ISOTOPE_CACHE.get(symbol)
    if cached is not None:
        return cached

    atomic_number = int(PERIODIC_TABLE.GetAtomicNumber(symbol))
    isotopes: list[tuple[float, float]] = []
    for mass_number in _ISOTOPE_MASS_NUMBERS.get(atomic_number, ()):
        abundance = float(
            PERIODIC_TABLE.GetAbundanceForIsotope(atomic_number, mass_number)
        )
        if abundance <= 0.0:
            continue
        isotopes.append(
            (
                float(PERIODIC_TABLE.GetMassForIsotope(atomic_number, mass_number)),
                abundance / 100.0,
            )
        )
    if not isotopes:
        isotopes = [(float(PERIODIC_TABLE.GetMostCommonIsotopeMass(symbol)), 1.0)]
    total = sum(prob for _mass, prob in isotopes)
    isotopes = [(mass, prob / total) for mass, prob in isotopes]
    _ELEMENT_ISOTOPE_CACHE[symbol] = isotopes
    return isotopes


def _convolve_isotope_distribution(
    distribution: list[tuple[float, float]],
    isotopes: list[tuple[float, float]],
    *,
    threshold: float,
    max_states: int,
) -> list[tuple[float, float]]:
    """
    Convolve a running isotope distribution with one element isotope distribution.
    """
    merged: dict[float, float] = {}
    for base_mass, base_prob in distribution:
        for isotope_mass, isotope_prob in isotopes:
            prob = float(base_prob) * float(isotope_prob)
            if prob < float(threshold):
                continue
            mass_key = round(float(base_mass) + float(isotope_mass), 8)
            merged[mass_key] = merged.get(mass_key, 0.0) + prob
    if not merged:
        base_mass, base_prob = max(distribution, key=lambda item: item[1])
        isotope_mass, isotope_prob = max(isotopes, key=lambda item: item[1])
        merged[round(float(base_mass) + float(isotope_mass), 8)] = float(
            base_prob
        ) * float(isotope_prob)
    out = sorted(merged.items(), key=lambda item: item[1], reverse=True)
    return [(float(mass), float(prob)) for mass, prob in out[: int(max_states)]]
