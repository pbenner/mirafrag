from __future__ import annotations

import re

_ADDUCT_CHARGE_PATTERN = re.compile(r'\](\d*)([+-])$')


def parse_adduct_charge(adduct: str | None) -> int:
    text = str(adduct or '').strip()
    match = _ADDUCT_CHARGE_PATTERN.search(text)
    if match is None:
        return 0
    magnitude_text, sign_text = match.groups()
    return parse_adduct_charge_suffix(f'{magnitude_text}{sign_text}')


def parse_adduct_charge_suffix(suffix: str) -> int:
    if not suffix:
        return 0
    sign = suffix[-1]
    if sign not in '+-':
        return 0
    magnitude_text = suffix[:-1]
    magnitude = int(magnitude_text) if magnitude_text else 1
    return magnitude if sign == '+' else -magnitude
