"""Column-name normalisation and alias resolution.

CIC-DDoS2019 and InSDN describe the same CICFlowMeter quantities under different
spellings: InSDN uses the V4 abbreviations ("Fwd Pkt Len Mean") while CIC-DDoS2019
uses long names, often with a stray leading space (" Fwd Packet Length Mean").
Table 1 and Table 2 of the paper show the two styles side by side without ever
saying they are the same features.

Everything downstream keys off the canonical snake_case name produced here, so
this module is the single place where header spelling matters.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# Raw header fragments that carry no information and only differ between releases.
_PUNCT_TO_UNDERSCORE = re.compile(r"[^0-9a-zA-Z]+")
_MULTI_UNDERSCORE = re.compile(r"_+")

# Two boundaries, applied together in one pass:
#   lower/digit -> upper   splits "FwdPacket"  -> "Fwd Packet"
#   upper -> upper+lower   splits "IATTotal"   -> "IAT Total"
# The second rule is what keeps concatenated acronyms (FwdIATTotal, URGFlagCount)
# from collapsing into fwd_iattotal / urg_flagcount.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Applied after the generic normaliser so that "Flow Byts/s" and "Flow Bytes/s"
# both land on "..._per_s" rather than a bare trailing "_s". The lookbehind stops
# an already-canonical "flow_bytes_per_s" from growing a second "_per".
_RATE_SUFFIX = re.compile(r"(?<!_per)_s$")


def normalize_column_name(raw: str) -> str:
    """Fold one raw header into canonical snake_case.

    strip -> unicode NFKC -> split camelCase -> punctuation to underscore ->
    collapse underscores -> lowercase.

    camelCase is only split when the header contains no whitespace, because a
    spaced header like "Fwd IAT Total" must not become "fwd_i_a_t_total".
    """
    name = unicodedata.normalize("NFKC", raw).replace("﻿", "").strip()
    if not name:
        return ""
    if not re.search(r"\s", name):
        name = _CAMEL_BOUNDARY.sub(" ", name)
    name = _PUNCT_TO_UNDERSCORE.sub("_", name)
    name = _MULTI_UNDERSCORE.sub("_", name).strip("_").lower()
    return name


def _rate_variants(name: str) -> List[str]:
    """Both spellings of a per-second rate column, so aliases match either."""
    variants = [name]
    if _RATE_SUFFIX.search(name):
        variants.append(_RATE_SUFFIX.sub("_per_s", name))
    return variants


def build_alias_index(schema_aliases: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    """Invert {canonical: [variants]} into {normalised_variant: canonical}.

    Raises on a variant claimed by two different canonical names, since that
    would make column resolution order-dependent.
    """
    index: Dict[str, str] = {}
    for canonical, variants in schema_aliases.items():
        canonical_norm = normalize_column_name(canonical)
        for variant in list(variants) + [canonical]:
            for spelling in _rate_variants(normalize_column_name(variant)):
                existing = index.get(spelling)
                if existing is not None and existing != canonical_norm:
                    raise ValueError(
                        f"alias {spelling!r} is claimed by both {existing!r} and "
                        f"{canonical_norm!r}; fix schema_aliases"
                    )
                index[spelling] = canonical_norm
    return index


def resolve_column(raw: str, alias_index: Mapping[str, str]) -> str:
    """Canonical name for one raw header, falling back to the generic normaliser."""
    normalised = normalize_column_name(raw)
    for spelling in _rate_variants(normalised):
        if spelling in alias_index:
            return alias_index[spelling]
    return normalised


def build_column_mapping(
    raw_columns: Iterable[str], schema_aliases: Mapping[str, Sequence[str]]
) -> Tuple[Dict[str, str], List[str]]:
    """Map every raw header to its canonical name.

    Returns the mapping plus the list of canonical names produced by more than one
    raw column. Collisions are reported rather than raised: some releases genuinely
    ship a duplicated column, and the caller decides whether that is fatal.
    """
    alias_index = build_alias_index(schema_aliases)
    mapping: Dict[str, str] = {}
    seen: Dict[str, str] = {}
    collisions: List[str] = []
    for raw in raw_columns:
        canonical = resolve_column(raw, alias_index)
        mapping[raw] = canonical
        if canonical in seen and canonical not in collisions:
            collisions.append(canonical)
        seen.setdefault(canonical, raw)
    return mapping, collisions


def find_label_column(
    raw_columns: Sequence[str], candidates: Sequence[str]
) -> str:
    """Locate the label column by normalised name, tolerating the leading space."""
    wanted = {normalize_column_name(c) for c in candidates}
    for raw in raw_columns:
        if normalize_column_name(raw) in wanted:
            return raw
    raise KeyError(
        f"no label column among {list(candidates)!r}; headers were {list(raw_columns)!r}"
    )
