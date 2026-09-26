"""
02_preprocess.py — Text Normalization Utilities
Shared preprocessing used by blocking and feature stages.
"""

import re
import unicodedata

# ── Legal suffix normalization ──────────────────────────────────────────────
SUFFIX_MAP = {
    r"\bcorp(?:oration)?\b": "corp",
    r"\binc(?:orporated)?\b": "inc",
    r"\bltd\b": "ltd",
    r"\blimited\b": "ltd",
    r"\bllc\b": "llc",
    r"\bllp\b": "llp",
    r"\bpvt\b": "pvt",
    r"\bprivate\b": "pvt",
    r"\bco(?:mpany)?\b": "co",
    r"\benterprises?\b": "ent",
    r"\bindustri(?:es|al)?\b": "ind",
    r"\bservices?\b": "svc",
    r"\bsolutions?\b": "sol",
    r"\bgroup\b": "grp",
    r"\bholdings?\b": "hld",
    r"\bintl\b": "intl",
    r"\binternational\b": "intl",
    r"\bassociates?\b": "assoc",
    r"\btrading\b": "trd",
    r"\bmanufacturing\b": "mfg",
    r"\bconsulting\b": "cons",
    r"\btechnologi(?:es|y)?\b": "tech",
    r"\btech\b": "tech",
}

# ── Address abbreviation normalization ──────────────────────────────────────
ADDR_MAP = {
    r"\bst(?:reet)?\b": "st",
    r"\brd\b": "rd",
    r"\broad\b": "rd",
    r"\bave?(?:nue)?\b": "ave",
    r"\bblvd\b": "blvd",
    r"\bboulevard\b": "blvd",
    r"\bdr(?:ive)?\b": "dr",
    r"\bln\b": "ln",
    r"\blane\b": "ln",
    r"\bct\b": "ct",
    r"\bcourt\b": "ct",
    r"\bpl(?:ace)?\b": "pl",
    r"\bsq(?:uare)?\b": "sq",
    r"\bhwy\b": "hwy",
    r"\bhighway\b": "hwy",
    r"\bpkwy\b": "pkwy",
    r"\bparkway\b": "pkwy",
    r"\bflr\b": "fl",
    r"\bfloor\b": "fl",
    r"\bste\b": "ste",
    r"\bsuite\b": "ste",
    r"\bapt\b": "apt",
    r"\bapartment\b": "apt",
    r"\bbldg\b": "bldg",
    r"\bbuilding\b": "bldg",
    r"\bnorth\b": "n",
    r"\bsouth\b": "s",
    r"\beast\b": "e",
    r"\bwest\b": "w",
    r"\bnr\b": "near",
    r"\bopposite\b": "opp",
    r"\bopp\b": "opp",
}


def unicode_normalize(text: str) -> str:
    """Convert accented chars to ASCII equivalents."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_name(text: str) -> str:
    """Normalize a business name for comparison."""
    if not text:
        return ""
    text = text.lower()
    text = unicode_normalize(text)
    # Replace & with and
    text = re.sub(r"\s*&\s*", " and ", text)
    # Remove punctuation except spaces and alphanumeric
    text = re.sub(r"[^\w\s]", " ", text)
    # Apply suffix normalization
    for pattern, replacement in SUFFIX_MAP.items():
        text = re.sub(pattern, replacement, text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(text: str) -> str:
    """Normalize a business address for comparison."""
    if not text:
        return ""
    text = text.lower()
    text = unicode_normalize(text)
    text = re.sub(r"[^\w\s]", " ", text)
    for pattern, replacement in ADDR_MAP.items():
        text = re.sub(pattern, replacement, text)
    # Remove common noise words
    text = re.sub(r"\b(?:near|next to|opp|opposite|behind|above|below)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_name_tokens(text: str) -> set:
    """Get meaningful tokens from a normalized name (min 2 chars)."""
    norm = normalize_name(text)
    tokens = set(t for t in norm.split() if len(t) >= 2)
    # Remove very common stop words
    stop = {"the", "of", "and", "in", "at", "by", "for", "to", "a", "an", "is"}
    return tokens - stop


def get_address_tokens(text: str) -> set:
    """Get meaningful tokens from a normalized address."""
    norm = normalize_address(text)
    tokens = set(t for t in norm.split() if len(t) >= 2)
    stop = {"the", "of", "and", "in", "at", "by", "for", "to", "a", "an", "is", "near", "opp"}
    return tokens - stop


def make_ngrams(text: str, n: int = 3) -> set:
    """Character n-grams for fuzzy blocking."""
    text = f"_{text}_"
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    # Quick smoke test
    print(normalize_name("Amazon.com, Inc."))
    print(normalize_name("AMAZON CORPORATION"))
    print(normalize_address("123 Main Street, Suite 400"))
    print(normalize_address("123 Main St Ste 400"))
    print(get_name_tokens("Tech Solutions Pvt Ltd"))
