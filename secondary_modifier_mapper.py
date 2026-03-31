"""
secondary_modifier_mapper.py
----------------------------
Lightweight standalone mapper for AIR/RMS secondary modifier fields:
  - roof_cover    (codes 0-11, shared AIR/RMS)
  - wall_type     (codes 0-9,  shared AIR/RMS)
  - foundation_type (codes 0-12, shared AIR/RMS)
  - soft_story    (codes 0-2,  shared AIR/RMS)

Key fact: AIR (Touchstone) and RMS use IDENTICAL integer codes for all four
secondary modifiers. No cross-scheme translation needed.

Usage:
    from secondary_modifier_mapper import SecondaryModifierMapper

    mapper = SecondaryModifierMapper()

    # Single field
    code = mapper.map_roof_cover("clay tile")        # -> 3
    code = mapper.map_foundation_type("slab")        # -> 8
    code = mapper.map_soft_story("yes")              # -> 2

    # All fields at once from a row dict
    result = mapper.map_all({
        "roof_cover":      "TPO membrane",
        "wall_type":       "brick veneer",
        "foundation_type": "pile",
        "soft_story":      "no",
    })
    # -> {"roof_cover": 7, "wall_type": 7, "foundation_type": 9, "soft_story": 1,
    #     "roof_cover_desc": "Single ply membrane", ...}
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference data loading
# ---------------------------------------------------------------------------

_REF_PATH = Path(__file__).parent / "reference" / "secondary_modifiers.json"

def _load_ref() -> Dict[str, Any]:
    with open(_REF_PATH, encoding="utf-8") as f:
        return json.load(f)


_REF_DATA: Dict[str, Any] = {}  # loaded lazily on first use


def _ref() -> Dict[str, Any]:
    global _REF_DATA
    if not _REF_DATA:
        _REF_DATA = _load_ref()
    return _REF_DATA


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize(raw: str) -> str:
    """Lowercase, strip, collapse whitespace, remove punctuation except / and -."""
    s = raw.strip().lower()
    s = re.sub(r"[^\w\s/-]", " ", s)   # keep word chars, /, and -
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_valid_int(val: str, lo: int, hi: int) -> Optional[int]:
    """Return int if val is an integer string in [lo, hi], else None."""
    try:
        i = int(val.strip())
        if lo <= i <= hi:
            return i
    except (ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Core lookup logic (shared across all modifier types)
# ---------------------------------------------------------------------------

def _lookup(
    raw: str,
    field: str,
    use_llm: bool = True,
    llm_client=None,
) -> Dict[str, Any]:
    """
    3-stage lookup for a single modifier field.

    Stage 1: Integer pass-through — already a valid code, return immediately.
    Stage 2: Exact alias match — O(1) dict lookup after normalization.
    Stage 3: Keyword token scan — longest-match substring search.
    Stage 4: LLM fallback — only for roof_cover and foundation_type (most
             ambiguous). Returns 0 (Unknown) if LLM unavailable or disabled.

    Returns dict with keys: code (int), description (str), method (str),
                             confidence (float), original (str).
    """
    spec = _ref()[field]
    codes     = spec["codes"]
    lo, hi    = spec["valid_range"]
    aliases   = spec["aliases"]
    keywords  = spec["keyword_tokens"]

    original = raw
    normalized = _normalize(raw)

    # ── Stage 1: Integer pass-through ──────────────────────────────────────
    maybe_int = _is_valid_int(normalized, lo, hi)
    if maybe_int is not None:
        return {
            "code":        maybe_int,
            "description": codes[str(maybe_int)],
            "method":      "integer",
            "confidence":  1.0,
            "original":    original,
        }

    # ── Stage 2: Exact alias lookup ─────────────────────────────────────────
    if normalized in aliases:
        code = aliases[normalized]
        return {
            "code":        code,
            "description": codes[str(code)],
            "method":      "alias",
            "confidence":  0.97,
            "original":    original,
        }

    # ── Stage 3: Keyword token scan (longest match wins) ───────────────────
    best_code   = None
    best_len    = 0
    for token, code in keywords.items():
        if token in normalized and len(token) > best_len:
            best_code = code
            best_len  = len(token)

    if best_code is not None:
        return {
            "code":        best_code,
            "description": codes[str(best_code)],
            "method":      "keyword",
            "confidence":  0.85,
            "original":    original,
        }

    # ── Stage 4: LLM fallback (roof_cover and foundation_type only) ─────────
    if use_llm and field in {"roof_cover", "foundation_type"} and llm_client:
        llm_code = _llm_classify(normalized, field, codes, llm_client)
        if llm_code is not None:
            return {
                "code":        llm_code,
                "description": codes[str(llm_code)],
                "method":      "llm",
                "confidence":  0.80,
                "original":    original,
            }

    # ── Default: Unknown (0) ────────────────────────────────────────────────
    logger.debug(f"[{field}] No match for '{original}' — defaulting to 0 (Unknown)")
    return {
        "code":        0,
        "description": codes["0"],
        "method":      "default",
        "confidence":  0.0,
        "original":    original,
    }


def _llm_classify(
    normalized: str,
    field: str,
    codes: Dict[str, str],
    llm_client,
) -> Optional[int]:
    """
    Minimal LLM call — returns a single integer code or None on failure.
    Uses plain text output (not JSON) to minimise latency and hallucination.
    """
    code_list = "\n".join(f"  {k}: {v}" for k, v in codes.items())
    field_label = field.replace("_", " ")
    prompt = (
        f"You are an expert at classifying building characteristics.\n"
        f"Map the following {field_label} description to exactly one integer from the list below.\n\n"
        f"Valid codes:\n{code_list}\n\n"
        f"Input: \"{normalized}\"\n\n"
        f"Return ONLY a single integer. No text, no explanation."
    )
    try:
        response = llm_client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config={"temperature": 0, "max_output_tokens": 8},
        )
        text = response.text.strip()
        val = int(re.sub(r"\D", "", text))
        lo, hi = _ref()[field]["valid_range"]
        if lo <= val <= hi:
            return val
    except Exception as e:
        logger.warning(f"LLM fallback failed for [{field}] '{normalized}': {e}")
    return None


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class SecondaryModifierMapper:
    """
    Lightweight mapper for AIR/RMS secondary modifier fields.

    AIR and RMS share IDENTICAL integer codes for:
      roof_cover (0-11), wall_type (0-9), foundation_type (0-12), soft_story (0-2)

    The mapper normalises raw text → canonical integer in three stages:
      1. Integer pass-through   (fastest — already a code)
      2. Exact alias dict       (O(1) lookup, 200+ entries)
      3. Keyword token scan     (longest-match substring)
      4. LLM fallback           (roof_cover & foundation_type only, optional)

    Args:
        use_llm:  Enable LLM stage 4 (default True if GEMINI_API_KEY present).
                  Set False for offline/test usage.
        llm_client: Pre-built google.genai client. Auto-created if None and use_llm=True.
    """

    FIELDS = ("roof_cover", "wall_type", "foundation_type", "soft_story")

    def __init__(self, use_llm: bool = True, llm_client=None):
        # Warm the reference data cache
        _ref()

        self._llm_client = None
        if use_llm:
            if llm_client:
                self._llm_client = llm_client
            else:
                api_key = os.getenv("GEMINI_API_KEY", "")
                if api_key:
                    try:
                        import google.genai as genai
                        self._llm_client = genai.Client(api_key=api_key)
                        logger.info("SecondaryModifierMapper: LLM stage active.")
                    except ImportError:
                        logger.warning("google-genai not installed. LLM stage disabled.")
                else:
                    logger.debug("GEMINI_API_KEY not set. LLM stage disabled.")

    # ── Per-field convenience methods ────────────────────────────────────────

    def map_roof_cover(self, raw: str) -> int:
        """Map raw roof cover description → code 0-11."""
        return self._map("roof_cover", raw)["code"]

    def map_wall_type(self, raw: str) -> int:
        """Map raw wall type description → code 0-9."""
        return self._map("wall_type", raw)["code"]

    def map_foundation_type(self, raw: str) -> int:
        """Map raw foundation type description → code 0-12."""
        return self._map("foundation_type", raw)["code"]

    def map_soft_story(self, raw: str) -> int:
        """Map raw soft story indicator → code 0-2."""
        return self._map("soft_story", raw)["code"]

    # ── Full detail (code + description + confidence + method) ───────────────

    def map_roof_cover_detail(self, raw: str) -> Dict[str, Any]:
        return self._map("roof_cover", raw)

    def map_wall_type_detail(self, raw: str) -> Dict[str, Any]:
        return self._map("wall_type", raw)

    def map_foundation_type_detail(self, raw: str) -> Dict[str, Any]:
        return self._map("foundation_type", raw)

    def map_soft_story_detail(self, raw: str) -> Dict[str, Any]:
        return self._map("soft_story", raw)

    # ── Batch: map all 4 fields from a row dict ────────────────────────────

    def map_all(self, row: Dict[str, str]) -> Dict[str, Any]:
        """
        Map all secondary modifier fields from a row dict.

        Args:
            row: dict with any subset of keys:
                 "roof_cover", "wall_type", "foundation_type", "soft_story"
                 Values can be raw text or already integer strings.

        Returns:
            Flat dict with integer codes AND descriptions:
            {
              "roof_cover": 7,       "roof_cover_desc": "Single ply membrane",
              "wall_type":  2,       "wall_type_desc": "Masonry",
              "foundation_type": 8,  "foundation_type_desc": "Mat / slab",
              "soft_story": 1,       "soft_story_desc": "No",
              "_methods": {"roof_cover": "alias", ...},
              "_confidence": {"roof_cover": 0.97, ...},
            }
        """
        result: Dict[str, Any] = {}
        methods: Dict[str, str] = {}
        confidence: Dict[str, float] = {}

        for field in self.FIELDS:
            raw_val = str(row.get(field) or "").strip()
            if not raw_val:
                raw_val = "0"  # treat missing as Unknown

            detail = self._map(field, raw_val)
            result[field]             = detail["code"]
            result[f"{field}_desc"]   = detail["description"]
            methods[field]            = detail["method"]
            confidence[field]         = detail["confidence"]

        result["_methods"]    = methods
        result["_confidence"] = confidence
        return result

    # ── Code description lookups ─────────────────────────────────────────────

    def describe(self, field: str, code: int) -> str:
        """Return the human-readable description for a given field + code."""
        codes = _ref()[field]["codes"]
        return codes.get(str(code), f"Unknown code {code}")

    def valid_codes(self, field: str) -> Dict[int, str]:
        """Return all valid codes for a field as {int: description}."""
        codes = _ref()[field]["codes"]
        return {int(k): v for k, v in codes.items()}

    # ── Internal ─────────────────────────────────────────────────────────────

    def _map(self, field: str, raw: str) -> Dict[str, Any]:
        if field not in self.FIELDS:
            raise ValueError(f"Unknown field '{field}'. Valid: {self.FIELDS}")
        if not raw or not raw.strip():
            spec = _ref()[field]
            return {
                "code": 0,
                "description": spec["codes"]["0"],
                "method": "empty",
                "confidence": 0.0,
                "original": raw,
            }
        return _lookup(raw, field, use_llm=bool(self._llm_client), llm_client=self._llm_client)

    def __repr__(self) -> str:
        llm = "enabled" if self._llm_client else "disabled"
        return f"SecondaryModifierMapper(llm={llm})"


# ---------------------------------------------------------------------------
# Module-level convenience (singleton for quick scripting use)
# ---------------------------------------------------------------------------

_default_mapper: Optional[SecondaryModifierMapper] = None


def get_mapper(use_llm: bool = True) -> SecondaryModifierMapper:
    """Return a cached (module-level) SecondaryModifierMapper instance."""
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = SecondaryModifierMapper(use_llm=use_llm)
    return _default_mapper


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    mapper = SecondaryModifierMapper(use_llm=False)
    tests = [
        ("roof_cover",     "asphalt shingles"),
        ("roof_cover",     "clay tile"),
        ("roof_cover",     "TPO"),
        ("roof_cover",     "standing seam metal"),
        ("roof_cover",     "3"),        # integer pass-through
        ("wall_type",      "brick veneer"),
        ("wall_type",      "stucco"),
        ("wall_type",      "HardiePlank"),
        ("foundation_type","slab on grade"),
        ("foundation_type","pile foundation"),
        ("foundation_type","cripple wall"),
        ("soft_story",     "yes"),
        ("soft_story",     "no"),
        ("soft_story",     "unknown"),
    ]
    print("=== SecondaryModifierMapper smoke test ===\n")
    for field, raw in tests:
        detail = mapper._map(field, raw)
        print(f"{field:20s}  '{raw:30s}'  -> {detail['code']:2d}  "
              f"({detail['description'][:30]})  [{detail['method']}, {detail['confidence']:.2f}]")

    print("\n--- map_all() ---")
    row = {
        "roof_cover":       "TPO membrane",
        "wall_type":        "brick veneer",
        "foundation_type":  "slab",
        "soft_story":       "no",
    }
    result = mapper.map_all(row)
    for field in mapper.FIELDS:
        print(f"  {field}: {result[field]} ({result[field+'_desc']})  [{result['_methods'][field]}]")
