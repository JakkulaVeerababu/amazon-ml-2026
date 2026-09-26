"""
preprocess_utils.py — Shared normalization functions
(imported by blocking, feature engineering, and matching stages)
"""

import re
import unicodedata
from typing import Set

# ── Legal suffix normalization ──────────────────────────────────────────────
SUFFIX_MAP = [
    (r"\bcorporation\b", "corp"),
    (r"\bcorp\b", "corp"),
    (r"\bincorporated\b", "inc"),
    (r"\b(?<![\w])inc\b", "inc"),
    (r"\blimited\b", "ltd"),
    (r"\bltd\b", "ltd"),
    (r"\bprivate\b", "pvt"),
    (r"\bpvt\b", "pvt"),
    (r"\bllc\b", "llc"),
    (r"\bllp\b", "llp"),
    (r"\bcompany\b", "co"),
    (r"\benterprises\b", "ent"),
    (r"\benterprise\b", "ent"),
    (r"\bindustries\b", "ind"),
    (r"\bindustrial\b", "ind"),
    (r"\bservices\b", "svc"),
    (r"\bservice\b", "svc"),
    (r"\bsolutions\b", "sol"),
    (r"\bsolution\b", "sol"),
    (r"\bgroup\b", "grp"),
    (r"\bholdings\b", "hld"),
    (r"\bholding\b", "hld"),
    (r"\binternational\b", "intl"),
    (r"\bintl\b", "intl"),
    (r"\bassociates\b", "assoc"),
    (r"\bassociate\b", "assoc"),
    (r"\btrading\b", "trd"),
    (r"\bmanufacturing\b", "mfg"),
    (r"\bconsulting\b", "cons"),
    (r"\btechnologies\b", "tech"),
    (r"\btechnology\b", "tech"),
    (r"\btech\b", "tech"),
    (r"\bventures\b", "vent"),
    (r"\bventure\b", "vent"),
    (r"\bsystems\b", "sys"),
    (r"\bsystem\b", "sys"),
    (r"\bnetworks\b", "net"),
    (r"\bnetwork\b", "net"),
]

ADDR_MAP = [
    (r"\bstreet\b", "st"),
    (r"\broad\b", "rd"),
    (r"\bavenue\b", "ave"),
    (r"\bboulevard\b", "blvd"),
    (r"\bdrive\b", "dr"),
    (r"\blane\b", "ln"),
    (r"\bcourt\b", "ct"),
    (r"\bplace\b", "pl"),
    (r"\bsquare\b", "sq"),
    (r"\bhighway\b", "hwy"),
    (r"\bparkway\b", "pkwy"),
    (r"\bfloor\b", "fl"),
    (r"\bsuite\b", "ste"),
    (r"\bapartment\b", "apt"),
    (r"\bbuilding\b", "bldg"),
    (r"\bnorth\b", "n"),
    (r"\bsouth\b", "s"),
    (r"\beast\b", "e"),
    (r"\bwest\b", "w"),
]

NAME_STOP = {"the", "of", "and", "in", "at", "by", "for", "to", "a", "an", "is", "do"}
ADDR_STOP = {"the", "of", "and", "in", "at", "by", "for", "to", "a", "an", "is",
             "near", "opp", "next", "behind", "above", "below"}


def unicode_normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_name(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = unicode_normalize(text)
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    for pattern, replacement in SUFFIX_MAP:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = unicode_normalize(text)
    text = re.sub(r"[^\w\s]", " ", text)
    for pattern, replacement in ADDR_MAP:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_name_tokens(text: str) -> Set[str]:
    norm = normalize_name(text)
    return {t for t in norm.split() if len(t) >= 2 and t not in NAME_STOP}


def get_address_tokens(text: str) -> Set[str]:
    norm = normalize_address(text)
    return {t for t in norm.split() if len(t) >= 2 and t not in ADDR_STOP}


def make_ngrams(text: str, n: int = 3) -> Set[str]:
    text = f"_{text}_"
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def token_overlap(a: Set, b: Set) -> float:
    """Overlap coefficient (intersection / min)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    m = min(len(a), len(b))
    return len(a & b) / m if m else 0.0
