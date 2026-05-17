from __future__ import annotations

import re
from dataclasses import dataclass

from rdkit import Chem

_ADDUCT_CHARGE_PATTERN = re.compile(r'\](\d*)([+-])$')
ELECTRON_MASS = 0.00054858
PERIODIC_TABLE = Chem.GetPeriodicTable()
HYDROGEN_MASS = float(PERIODIC_TABLE.GetMostCommonIsotopeMass('H'))
PROTON_MASS = HYDROGEN_MASS - ELECTRON_MASS
SODIUM_ADDUCT_MASS = (
    float(PERIODIC_TABLE.GetMostCommonIsotopeMass('Na')) - ELECTRON_MASS
)
POTASSIUM_ADDUCT_MASS = (
    float(PERIODIC_TABLE.GetMostCommonIsotopeMass('K')) - ELECTRON_MASS
)
AMMONIUM_ADDUCT_MASS = (
    float(PERIODIC_TABLE.GetMostCommonIsotopeMass('N'))
    + 4.0 * HYDROGEN_MASS
    - ELECTRON_MASS
)
ADDUCT_FORMULA_ALIASES = {
    'ACN': 'C2H3N',
    'FA': 'CH2O2',
    'HFA': 'CH2O2',
    'FORMATE': 'CHO2',
    'HCOO': 'CHO2',
    'HCOOH': 'CH2O2',
    'AC': 'C2H4O2',
    'HAC': 'C2H4O2',
}


@dataclass(frozen=True)
class Adduct:
    """
    Parsed ion adduct description.

    The object stores the bracket label, molecule multiplier, signed mass delta, and ion charge. It is used by fragmentation and metadata code to convert neutral fragment masses into observed m/z values.
    """

    label: str
    molecule_multiplier: int
    mass_delta: float
    charge: int

    def mz(self, neutral_mass: float) -> float:
        """
        Convert a neutral molecular or fragment mass to m/z.

        The calculation applies the molecule multiplier, adds the parsed adduct mass delta, and divides by the absolute ion charge. A zero charge is guarded as singly charged to keep malformed fallbacks from crashing downstream code.
        """
        charge_abs = max(abs(int(self.charge)), 1)
        return (
            float(self.molecule_multiplier) * float(neutral_mass)
            + float(self.mass_delta)
        ) / float(charge_abs)


def parse_adduct(adduct: str | None, *, default_mass: float = PROTON_MASS) -> Adduct:
    """
    Parse a bracketed mass-spectrometry adduct string.

    Supported inputs include common signed forms such as ``[M+H]+``, ``[M-H]-``, ``[M+Na]+``, ``[M+2H]2+``, and formula modifiers such as ``[M+FA-H]-``. Missing or NaN labels fall back to protonation so prediction-only inputs still produce candidate masses.
    """
    label = str(adduct or '').strip()
    if not label or label.lower() == 'nan':
        return Adduct('[M+H]+', 1, float(default_mass), 1)
    if label == '[M+H]+':
        return Adduct(label, 1, float(default_mass), 1)
    if not label.startswith('['):
        raise ValueError(f'Unsupported adduct {label!r}; expected bracketed form.')
    try:
        close_idx = label.rindex(']')
    except ValueError as exc:
        raise ValueError(
            f'Unsupported adduct {label!r}; missing closing bracket.'
        ) from exc

    body = label[1:close_idx]
    charge = parse_adduct_charge_suffix(label[close_idx + 1 :])
    if charge == 0:
        raise ValueError(f'Unsupported adduct {label!r}; missing ion charge.')

    pos = 0
    multiplier_text = ''
    while pos < len(body) and body[pos].isdigit():
        multiplier_text += body[pos]
        pos += 1
    if pos >= len(body) or body[pos] != 'M':
        raise ValueError(f'Unsupported adduct {label!r}; expected M in adduct body.')
    molecule_multiplier = int(multiplier_text) if multiplier_text else 1

    neutral_delta = _adduct_expression_mass(body[pos + 1 :])
    mass_delta = neutral_delta - float(charge) * ELECTRON_MASS
    return Adduct(
        label=label,
        molecule_multiplier=max(1, int(molecule_multiplier)),
        mass_delta=float(mass_delta),
        charge=int(charge),
    )


def parse_adduct_charge(adduct: str | None) -> int:
    """
    Return the integer ion charge encoded in an adduct label.

    This helper only reads the suffix after the closing bracket, for example ``+``, ``-``, ``2+``, or ``3-``. Invalid or missing suffixes return zero so callers can decide how strict they want to be.
    """
    text = str(adduct or '').strip()
    match = _ADDUCT_CHARGE_PATTERN.search(text)
    if match is None:
        return 0
    magnitude_text, sign_text = match.groups()
    return parse_adduct_charge_suffix(f'{magnitude_text}{sign_text}')


def parse_adduct_charge_suffix(suffix: str) -> int:
    """
    Parse a bare charge suffix into a signed integer.

    The suffix is expected to end in ``+`` or ``-`` with an optional leading magnitude. Empty strings and malformed suffixes return zero rather than raising.
    """
    if not suffix:
        return 0
    sign = suffix[-1]
    if sign not in '+-':
        return 0
    magnitude_text = suffix[:-1]
    magnitude = int(magnitude_text) if magnitude_text else 1
    return magnitude if sign == '+' else -magnitude


def adduct_mass_delta(
    adduct: str | None, *, default_mass: float = PROTON_MASS
) -> float:
    """
    Return only the signed mass delta for an adduct.

    This is a convenience wrapper around :func:`parse_adduct` for call sites that need ion mass adjustment but do not need the molecule multiplier or charge.
    """
    return parse_adduct(adduct, default_mass=default_mass).mass_delta


def _adduct_expression_mass(expression: str) -> float:
    """
    Evaluate the signed expression inside an adduct body.

    The expression is the part after ``M`` and is parsed as a sequence of ``+TERM`` or ``-TERM`` modifiers. Each term can be a formula or a supported alias such as ``FA`` or ``ACN``.
    """
    if not expression:
        return 0.0
    total = 0.0
    pos = 0
    while pos < len(expression):
        marker = expression[pos]
        if marker not in '+-':
            raise ValueError(f'Unsupported adduct expression {expression!r}.')
        sign = 1.0 if marker == '+' else -1.0
        pos += 1
        start = pos
        while pos < len(expression) and expression[pos] not in '+-':
            pos += 1
        term = expression[start:pos]
        if not term:
            raise ValueError(f'Unsupported adduct expression {expression!r}.')
        total += sign * _adduct_term_mass(term)
    return total


def _adduct_term_mass(term: str) -> float:
    """
    Convert one adduct modifier term to a neutral mass.

    Terms may have a leading multiplier and may use formula aliases. The returned value does not include electron-mass correction; that is applied once the net ion charge is known.
    """
    formula = ADDUCT_FORMULA_ALIASES.get(term.upper(), term)
    pos = 0
    multiplier_text = ''
    while pos < len(formula) and formula[pos].isdigit():
        multiplier_text += formula[pos]
        pos += 1
    multiplier = int(multiplier_text) if multiplier_text else 1
    return float(multiplier) * formula_mass(formula[pos:])


def formula_mass(formula: str) -> float:
    """
    Compute a monoisotopic mass for a simple chemical formula.

    The parser supports element symbols followed by optional integer counts. It intentionally rejects unknown elements and malformed formula strings so unsupported adducts do not silently produce wrong masses.
    """
    if not formula:
        raise ValueError('Empty adduct formula.')
    total = 0.0
    pos = 0
    while pos < len(formula):
        if not formula[pos].isupper():
            raise ValueError(f'Unsupported adduct formula {formula!r}.')
        symbol = formula[pos]
        pos += 1
        if pos < len(formula) and formula[pos].islower():
            symbol += formula[pos]
            pos += 1
        count_text = ''
        while pos < len(formula) and formula[pos].isdigit():
            count_text += formula[pos]
            pos += 1
        count = int(count_text) if count_text else 1
        atomic_number = int(PERIODIC_TABLE.GetAtomicNumber(symbol))
        if atomic_number <= 0:
            raise ValueError(f'Unsupported adduct element {symbol!r}.')
        total += float(count) * float(PERIODIC_TABLE.GetMostCommonIsotopeMass(symbol))
    return total
