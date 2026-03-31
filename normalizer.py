"""
normalizer.py — All row-level normalization for the CAT pipeline.

Functions:
  normalize_all_rows(rows, rules_config) → (normalized_rows, new_flags)

Each sub-function focuses on one field group and returns
(updated_row, list_of_flag_dicts).
"""
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from rules import BusinessRulesConfig
from secondary_modifier_mapper import get_mapper

logger = logging.getLogger("normalizer")

# ── Lookup tables ──────────────────────────────────────────────────────────────

SPRINKLER_TRUE_VALUES = {"yes", "y", "1", "true", "sprinklered", "wet pipe", "dry pipe", "wet", "dry"}
SPRINKLER_FALSE_VALUES = {"no", "n", "0", "false", "none", "no sprinkler", "unsprinklered"}

WOOD_FRAME_CODES_AIR = {str(c) for c in range(101, 106)}  # 101-105
MASONRY_CODES_AIR = {str(c) for c in range(111, 122)}     # 111-121

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "dual": 2, "double": 2, "triple": 3, "ground": 1,
}

# ── Helper ─────────────────────────────────────────────────────────────────────

def _make_flag(row_index: int, field: str, issue: str, current_value: Any,
               message: str, confidence: Optional[float] = None,
               alternatives: Optional[List] = None) -> dict:
    return {
        "row_index": row_index,
        "field": field,
        "issue": issue,
        "current_value": current_value,
        "confidence": confidence,
        "alternatives": alternatives or [],
        "message": message,
    }


# ── Year built ─────────────────────────────────────────────────────────────────

def _normalize_year(row: dict, row_idx: int, rules: BusinessRulesConfig,
                    target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []
    year_key = "YearBuilt" if target == "AIR" else "YEARBUILT"
    retro_key = "YearRetrofitted" if target == "AIR" else "YEARUPGRAD"
    raw = row.get("YearBuilt") or row.get("YEARBUILT") or row.get(year_key)
    if raw is None or str(raw).strip() == "":
        return row, flags

    s = str(raw).strip()

    if re.match(r"^\s*-\s*\d{4}\s*$", s):
        flags.append(_make_flag(row_idx, year_key, "negative_year_built",
                                raw, f"Negative year '{raw}' — blanked."))
        row[year_key] = None
        row["Year_Built_Flag"] = "UNPARSEABLE"
        return row, flags

    years = [int(m) for m in re.findall(r"\b(18\d{2}|19\d{2}|20\d{2})\b", s)]
    
    if not years:
        for m in re.findall(r"\b(\d{2})\b", s):
            yy = int(m)
            year = 1900 + yy if yy >= 25 else 2000 + yy
            years.append(year)
            
    if not years:
        row[year_key] = None
        row["Year_Built_Flag"] = "UNPARSEABLE"
        flags.append(_make_flag(row_idx, year_key, "unparseable_year",
                                raw, f"Cannot parse year from '{raw}' — blanked."))
        return row, flags

    original_num_years = len(years)
    year = min(years)
    row["Year_Built_Flag"] = "VALID"

    if original_num_years > 1:
        flags.append(_make_flag(row_idx, year_key, "multiple_years_extracted",
                                raw, f"Multiple years found; selected earliest {year} from '{raw}'."))

    if not (rules.year_min <= year <= rules.year_max):
        row["Year_Built_Flag"] = "OUT_OF_RANGE"
        action = rules.invalid_year_action
        if action == "reset_year":
            row[year_key] = None
            row["Year_Built_Original"] = year
        elif action == "set_default":
            row[year_key] = rules.year_default
            row["Year_Built_Original"] = year
        else:
            row[year_key] = year

        flags.append(_make_flag(row_idx, year_key, "out_of_range_year",
                                year, f"Year {year} outside range [{rules.year_min}–{rules.year_max}]"))
    else:
        row[year_key] = year

    # Pre-1940 masonry check
        if year < 1940 and row.get("ConstructionCode") in MASONRY_CODES_AIR:
            method = row.get("Occupancy_Method", "")
            if method not in ("llm", "user_override"):
                flags.append(_make_flag(row_idx, "ConstructionCode", "pre1940_masonry",
                                        row.get("ConstructionCode"),
                                        "Pre-1940 masonry — verify ductility. Consider unreinforced masonry codes (111–114)."))

    return row, flags

# Extract first 4-digit year matching 18xx|19xx|20xx
    match = re.search(r"\b(18|19|20)\d{2}\b", str(raw))
    if not match:
        row[year_key] = None
        row["Year_Built_Flag"] = "UNPARSEABLE"
        if str(raw).strip():
            flags.append(_make_flag(row_idx, year_key, "unparseable_year",
                                    raw, f"Cannot parse year from '{raw}'"))
        return row, flags

    year = int(match.group())
    row["Year_Built_Flag"] = "VALID"

    if not (rules.year_min <= year <= rules.year_max):
        row["Year_Built_Flag"] = "OUT_OF_RANGE"
        action = rules.invalid_year_action
        if action == "reset_year":
            row[year_key] = None
            row["Year_Built_Original"] = year
        elif action == "set_default":
            row[year_key] = rules.year_default
            row["Year_Built_Original"] = year
        else:  # flag_review or none
            row[year_key] = year

        flags.append(_make_flag(row_idx, year_key, "out_of_range_year",
                                year, f"Year {year} outside range [{rules.year_min}–{rules.year_max}]"))
    else:
        row[year_key] = year

    # Pre-1940 masonry check
    if year < 1940 and row.get("ConstructionCode") in MASONRY_CODES_AIR:
        method = row.get("Occupancy_Method", "")
        if method not in ("llm", "user_override"):
            flags.append(_make_flag(row_idx, "ConstructionCode", "pre1940_masonry",
                                    row.get("ConstructionCode"),
                                    "Pre-1940 masonry — verify ductility. Consider unreinforced masonry codes (111–114)."))
    return row, flags



_EXCLUDE_UPGRADES = re.compile(
    r"\b(roof|electrical|plumb|hvac|mechanical|cosmetic|interior|tenant|appraisal)\b",
    re.IGNORECASE
)

_INCLUDE_UPGRADES = re.compile(
    r"\b(upgrade|renovat|retrofit|rehab|modernize|code|reconstruct|rebuilt|reno)\b",
    re.IGNORECASE
)

def _normalize_upgraded_year(row: dict, row_idx: int, rules: BusinessRulesConfig,
                             target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []
    upg_key = "YearRetrofitted" if target == "AIR" else "YEARUPGRAD"
    
    raw = row.get("YearRetrofitted") or row.get("YEARUPGRAD") or row.get(upg_key)
    
    if raw is None or str(raw).strip() == "":
        row[upg_key] = None
        return row, flags

    s = str(raw).strip()
    s_lower = s.lower()

    if _EXCLUDE_UPGRADES.search(s_lower) and not _INCLUDE_UPGRADES.search(s_lower):
        flags.append(_make_flag(row_idx, upg_key, "ignored_upgrade",
                                raw, f"Ignored non-structural upgrade '{raw}' — returning 9999."))
        row[upg_key] = 9999
        return row, flags

    years = [int(m) for m in re.findall(r"\b(18\d{2}|19\d{2}|20\d{2})\b", s)]
    
    if not years:
        row[upg_key] = None
        return row, flags

    # Read the year_built that was just resolved by _normalize_year
    yb_key = "YearBuilt" if target == "AIR" else "YEARBUILT"
    year_built = row.get(yb_key)

    if not year_built:
        flags.append(_make_flag(row_idx, upg_key, "upgrade_missing_built",
                                raw, f"Upgrade year '{raw}' ignored because YearBuilt is empty."))
        row[upg_key] = None
        return row, flags

    upgrade_years = [y for y in years if y != year_built]

    if not upgrade_years:
        flags.append(_make_flag(row_idx, upg_key, "upgrade_same_as_built",
                                raw, f"Upgrade year matches Built year — returning 9999."))
        row[upg_key] = 9999
        return row, flags

    year_upgraded = max(upgrade_years)
    
    if year_upgraded <= year_built:
        flags.append(_make_flag(row_idx, upg_key, "upgrade_before_built",
                                raw, f"Upgrade year {year_upgraded} <= Built {year_built} — returning 9999."))
        row[upg_key] = 9999
        return row, flags

    if year_upgraded > 2026:
        row[upg_key] = 9999
        return row, flags

    row[upg_key] = year_upgraded
    return row, flags


# ── Number of stories ──────────────────────────────────────────────────────────


_WORD_TO_NUM_STORIES = {
    "single": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

def _normalize_stories(row: dict, row_idx: int, rules: BusinessRulesConfig,
                        target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []
    stories_key = "NumberOfStories" if target == "AIR" else "NUMSTORIES"
    raw = row.get("NumberOfStories") or row.get("NUMSTORIES") or row.get(stories_key)
    if raw is None or str(raw).strip() == "":
        row[stories_key] = None
        return row, flags

    s = str(raw).strip()

    # Exact negative check (invalid)
    if re.match(r"^\s*-\s*\d+(?:\.\d+)?\s*$", s):
        flags.append(_make_flag(row_idx, stories_key, "negative_stories",
                                raw, f"Negative stories '{raw}' — blanked."))
        row[stories_key] = None
        return row, flags

    s_lower = s.lower()
    
    # 0. Evaluate explicit summations like "3 + 2" -> "5"
    while re.search(r'(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)', s_lower):
        def repl_add(m):
            return " " + str(float(m.group(1)) + float(m.group(2))) + " "
        s_lower = re.sub(r'(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)', repl_add, s_lower, count=1)
    
    # 1. Expand "G+X" / "Ground+X" -> (1+X)
    def repl_g(m):
        return " " + str(1 + int(m.group(1))) + " "
    if re.search(r'\bg(?:round)?\s*\+', s_lower):
        s_lower = re.sub(r'\bg(?:round)?\s*\+\s*(\d+)\b', repl_g, s_lower)
    
    # 2. Ignore Mezzanine / Basement
    if re.search(r'mezzanine|mezz|basement|bsmt', s_lower):
        s_lower = re.sub(r'(?:\+\s*)?\b(?:\d+\s+)?(?:mezzanine|mezz|basement|bsmt)\b', ' ', s_lower)
        flags.append(_make_flag(row_idx, stories_key, "ignored_mezzanine_basement",
                                raw, f"Ignored mezzanine/basement references in '{raw}'."))

    # 3. Ignore building counts (e.g. "5 Buildings - Various")
    if re.search(r'building|bldg|structure', s_lower):
        s_lower = re.sub(r'\b(\d+)\s+(?:buildings?|bldgs?|structures?)\b', ' ', s_lower)

    # 4. Convert words
    for w, n in _WORD_TO_NUM_STORIES.items():
        if w in s_lower:
            s_lower = re.sub(r'\b' + re.escape(w) + r'\b', str(n), s_lower)

    # 5. Extract all positive decimals
    s_clean = re.sub(r'[-\u2013/,(]+', ' ', s_lower)
    s_clean = re.sub(r'\bto\b', ' ', s_clean)
    
    nums = [float(m) for m in re.findall(r'\d+(?:\.\d+)?', s_clean)]
    
    if not nums:
        flags.append(_make_flag(row_idx, stories_key, "unparseable_stories",
                                raw, f"Cannot parse number of stories from '{raw}' — blanked."))
        row[stories_key] = None
        return row, flags
        
    import math
    stories = max(math.ceil(n) for n in nums)
    row[stories_key] = stories

    if '+' in str(raw) and len(nums) == 1 and str(raw).strip() != s_lower.strip():
        # Evaluated an addition
        flags.append(_make_flag(row_idx, stories_key, "summed_stories",
                                raw, f"Evaluated addition in '{raw}' yielding {stories}."))
    elif len(nums) > 1:
        flags.append(_make_flag(row_idx, stories_key, "multiple_story_values",
                                raw, f"Extracted maximum value {stories} from '{raw}'."))
    elif any("." in m for m in re.findall(r'\d+(?:\.\d+)?', s_clean)):
        flags.append(_make_flag(row_idx, stories_key, "decimal_stories",
                                raw, f"Decimal stories '{raw}' rounded up to {stories}."))

    # Business rule: wood frame story limit
    const_code = str(row.get("ConstructionCode", "") or row.get("BLDGCLASS", ""))
    if stories > rules.max_stories_wood_frame and const_code in WOOD_FRAME_CODES_AIR:
        action = rules.stories_exceeded_action
        const_field = "ConstructionCode" if target == "AIR" else "BLDGCLASS"
        if action == "reset_construction":
            row["Construction_Code_Original"] = const_code
            row[const_field] = "100"  # Unknown
            flags.append(_make_flag(row_idx, const_field, "wood_frame_stories_exceeded",
                                    const_code, f"{stories}-story building exceeds wood frame limit; construction reset to Unknown"))
        elif action == "reset_stories":
            row[stories_key] = None
            flags.append(_make_flag(row_idx, stories_key, "wood_frame_stories_exceeded",
                                    stories, f"Stories reset: {stories} exceeds max {rules.max_stories_wood_frame} for wood frame"))
        else:
            flags.append(_make_flag(row_idx, stories_key, "wood_frame_stories_exceeded",
                                    stories, f"Info: {stories} stories for wood frame (code {const_code}) exceeds typical limit of {rules.max_stories_wood_frame}"))

    return row, flags

# ── Building count ─────────────────────────────────────────────────────────────

# Named structures that count as +1 in additive strings if no number precedes them
_NAMED_STRUCTURES = frozenset({
    "clubhouse", "club house", "community center", "center", "office",
    "shed", "maintenance shed", "garage", "auxiliary", "aux",
    "main building", "reception", "gym", "pool house", "amenity center",
    "annex", "pavilion", "chapel", "library", "lobby",
})

_VAGUE_BLDG_PAT = re.compile(
    r"^(n/?a|various|multiple buildings?|multiple|tbd|unknown|"
    r"several|numerous|many|tba|none|nil|na|-)$",
    re.IGNORECASE,
)

_SINGLE_BLDG_PAT = re.compile(
    r"^(single building|single|one building|one)$", re.IGNORECASE
)

_WORD_TO_NUM: Dict[str, int] = {
    **WORD_NUMBERS,
    "a": 1, "an": 1,
    "twenty-one": 21, "twenty one": 21, "twenty-two": 22, "twenty two": 22,
    "twenty-three": 23, "twenty three": 23, "twenty-four": 24, "twenty four": 24,
    "twenty-five": 25, "twenty five": 25,
    "thirty": 30, "forty": 40, "fifty": 50,
}


def _word_to_int(token: str) -> Optional[int]:
    return _WORD_TO_NUM.get(token.strip().lower())


def _extract_all_ints(text: str) -> List[int]:
    return [int(m.replace(",", "")) for m in re.findall(r"\d[\d,]*", text)]


def _normalize_building_count(row: dict, row_idx: int,
                               target: str = "AIR") -> Tuple[dict, List[dict]]:
    """
    Normalize RiskCount (AIR) / NUMBLDGS (RMS) to a positive whole integer.

    Processing order:
      1  Blank/None -> None
      2  Vague/N/A text -> None + flag
      3  "Single Building" / "One" -> 1
      4  Word-number prefix (Eight Buildings -> 8)
      5  Units-vs-buildings ("X Units in Y Buildings" -> Y)
      6  Additive (+/&/and): sum all parts; named structure = +1 each
      7  Qualifier ("X (Including Y)") -> keep X
      8  Range -> take LARGEST
      9  Slash -> take LARGEST
     10  Decimal -> math.ceil
     11  Bare integer (or largest of multiple)
     12  Negative -> None + flag
     13  Unresolvable -> None + flag
    """
    flags: List[dict] = []
    bldg_key = "RiskCount" if target == "AIR" else "NUMBLDGS"

    raw = row.get(bldg_key) or row.get("RiskCount") or row.get("NUMBLDGS")
    if raw is None or str(raw).strip() == "":
        row[bldg_key] = None
        return row, flags

    s = str(raw).strip()

    # Negative value — detect BEFORE digit regex strips the minus sign
    if re.match(r"^-\s*\d", s):
        flags.append(_make_flag(row_idx, bldg_key, "negative_building_count",
                                raw, f"Negative building count '{raw}' — blanked."))
        row[bldg_key] = None
        return row, flags

    # 2. Vague / N/A
    if _VAGUE_BLDG_PAT.match(s):
        flags.append(_make_flag(row_idx, bldg_key, "vague_building_count",
                                raw, f"\'{raw}\' is vague/N/A — blanked."))
        row[bldg_key] = None
        return row, flags

    # 3. Single constant
    if _SINGLE_BLDG_PAT.match(s):
        row[bldg_key] = 1
        return row, flags

    s_lower = s.lower()

    # 4. Word-number prefix
    first_tok = s_lower.split()[0] if s_lower.split() else ""
    wn = _word_to_int(first_tok)
    if wn is not None:
        s_lower = s_lower.replace(first_tok, str(wn), 1)

    # 5a. "X Units in Y Buildings" -> Y
    m = re.search(r"(\d[\d,]*)\s+units?\s+in\s+(\d[\d,]*)\s+buildings?", s_lower)
    if m:
        result = int(m.group(2).replace(",", ""))
        flags.append(_make_flag(row_idx, bldg_key, "units_vs_buildings",
                                raw, f"Extracted building count ({result}) from 'units in buildings' pattern."))
        row[bldg_key] = result
        return row, flags

    # 5b. "X Buildings (Y Units Each)" -> X
    m = re.search(r"(\d[\d,]*)\s+buildings?\s*\(\s*\d+\s+units?", s_lower)
    if m:
        row[bldg_key] = int(m.group(1).replace(",", ""))
        return row, flags

    # 6. Additive
    if re.search(r"\+|&|\band\b", s_lower):
        parts = re.split(r"\+|&|\band\b", s_lower)
        total = 0
        valid_parse = True
        for part in parts:
            part = part.strip()
            if not part:
                continue
            nm = re.search(r"(\d[\d,]*)", part)
            if nm:
                total += int(nm.group(1).replace(",", ""))
            else:
                first = part.split()[0] if part.split() else ""
                wn2 = _word_to_int(first)
                if wn2 is not None:
                    total += wn2
                else:
                    part_clean = re.sub(
                        r"\b(buildings?|bldgs?|structures?)\b", "", part
                    ).strip()
                    if any(ns in part_clean for ns in _NAMED_STRUCTURES):
                        total += 1
                    elif part_clean:
                        valid_parse = False
                        break
        if valid_parse and total > 0:
            flags.append(_make_flag(row_idx, bldg_key, "additive_building_count",
                                    raw, f"Additive count: \'{raw}\' -> {total}."))
            row[bldg_key] = total
            return row, flags

    # 7. "X (Including Y)" qualifier -> keep X
    m = re.match(r"^(\d[\d,]*)\s*\(including\b", s_lower)
    if m:
        row[bldg_key] = int(m.group(1).replace(",", ""))
        return row, flags

    # Expand parenthetical content for remaining steps
    s_clean = re.sub(r"\(([^)]*)\)", lambda mo: " " + mo.group(1) + " ", s_lower)
    s_clean = s_clean.replace("(", " ").replace(")", " ").strip()

    # 8. Range -> take LARGEST
    m = re.search(r"(\d[\d,]*)\s*[-\u2013to]+\s*(\d[\d,]*)", s_clean)
    if m:
        a, b = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
        result = max(a, b)
        flags.append(_make_flag(row_idx, bldg_key, "building_count_range",
                                raw, f"Range \'{raw}\' — took largest ({result})."))
        row[bldg_key] = result
        return row, flags

    # 9. Slash -> take LARGEST
    m = re.search(r"(\d[\d,]*)\s*/\s*(\d[\d,]*)", s_clean)
    if m:
        a, b = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
        result = max(a, b)
        flags.append(_make_flag(row_idx, bldg_key, "building_count_slash",
                                raw, f"Slash \'{raw}\' — took largest ({result})."))
        row[bldg_key] = result
        return row, flags

    # 10. Decimal -> ceil
    float_m = re.search(r"(\d+\.\d+)", s_clean)
    if float_m:
        result = math.ceil(float(float_m.group(1)))
        flags.append(_make_flag(row_idx, bldg_key, "decimal_building_count",
                                raw, f"Decimal \'{raw}\' rounded up to {result}."))
        row[bldg_key] = result
        return row, flags

    # 11. Bare integers
    nums = _extract_all_ints(s_clean)
    if nums:
        result = max(nums)
        if len(nums) > 1:
            flags.append(_make_flag(row_idx, bldg_key, "multiple_integers_in_count",
                                    raw, f"Multiple integers in \'{raw}\' — kept largest ({result})."))
        # 12. Negative
        if result < 0:
            flags.append(_make_flag(row_idx, bldg_key, "negative_building_count",
                                    raw, f"Negative building count \'{raw}\' — blanked."))
            row[bldg_key] = None
            return row, flags
        row[bldg_key] = result
        return row, flags

    # 13. Unresolvable
    flags.append(_make_flag(row_idx, bldg_key, "unresolvable_building_count",
                            raw, f"Cannot parse building count from \'{raw}\' — blanked."))
    row[bldg_key] = None
    return row, flags


# ── Gross area ─────────────────────────────────────────────────────────────────

_VAGUE_AREA_PAT = re.compile(
    r"^(n/?a|various|varies|tbd|unknown|none|nil|included|-)$",
    re.IGNORECASE,
)

_INVALID_UNITS = re.compile(
    r"\b(acre|acres|ac|hectare|hectares|ha|sq\s*meters?|sqm|m2)\b",
    re.IGNORECASE,
)

def _normalize_area(row: dict, row_idx: int, rules: BusinessRulesConfig,
                    target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []
    area_key = "GrossArea" if target == "AIR" else "FLOORAREA"
    raw = row.get("GrossArea") or row.get("FLOORAREA") or row.get(area_key)
    if raw is None or str(raw).strip() == "":
        row[area_key] = None
        return row, flags

    s = str(raw).strip()

    if _INVALID_UNITS.search(s):
        flags.append(_make_flag(row_idx, area_key, "invalid_area_unit",
                                raw, f"Invalid unit in \'{raw}\' (Acres/Hectares/SqM) — blanked."))
        row[area_key] = None
        return row, flags

    if re.match(r"^\s*-\s*\d+(?:\.\d+)?\s*$", s):
        flags.append(_make_flag(row_idx, area_key, "negative_area",
                                raw, f"Negative area \'{raw}\' — blanked."))
        row[area_key] = None
        return row, flags

    s_lower = s.lower()

    if _VAGUE_AREA_PAT.match(s_lower):
        flags.append(_make_flag(row_idx, area_key, "vague_area",
                                raw, f"Vague/included area \'{raw}\' — blanked."))
        row[area_key] = None
        return row, flags

    def repl_km(m):
        val = float(m.group(1).replace(',', ''))
        mult = m.group(2)
        if mult == 'k': val *= 1000
        elif mult.startswith('m') or mult == 'mn': val *= 1000000
        return f" {int(val)} " if val.is_integer() else f" {val} "
    
    s_lower = re.sub(r'\b(\d[\d,\.]*)\s*(k|m|mn|million)[\bs]?\b', repl_km, s_lower)

    def repl_each(m):
        val1 = float(m.group(1).replace(',', ''))
        val2 = float(m.group(2).replace(',', ''))
        return f" {val1 * val2} "
        
    s_lower = re.sub(r'(\d[\d,\.]*)\s*\(each\)\s*(\d[\d,\.]*)', repl_each, s_lower)

    sqft_value = None

    if '+' in s_lower or '&' in s_lower:
        parts = re.split(r'\+|&', s_lower)
        total = 0.0
        for part in parts:
            nums = [float(n.replace(',', '')) for n in re.findall(r'(?<!-)\b\d[\d,\.]*', part.strip())]
            if nums:
                total += max(nums)
        if total > 0:
            sqft_value = total

    if sqft_value is None:
        s_clean = re.sub(r'[-\u2013/]', ' ', s_lower)
        nums = []
        for m in re.findall(r'\d[\d,\.]*', s_clean):
            val_str = m.replace(',', '')
            if val_str and val_str != '.':
                nums.append(float(val_str))
        
        if not nums:
            flags.append(_make_flag(row_idx, area_key, "unparseable_area",
                                    raw, f"Cannot parse area from \'{raw}\' — blanked."))
            row[area_key] = None
            return row, flags
        
        sqft_value = max(nums)

    row["Area_Converted"] = False

    if sqft_value < rules.min_area_sqft:
        if rules.invalid_area_action == "flag_review":
            flags.append(_make_flag(row_idx, area_key, "area_below_minimum",
                                    raw, f"Area {sqft_value:.0f} sqft below minimum {rules.min_area_sqft} sqft"))

    row[area_key] = round(sqft_value, 2) if not sqft_value.is_integer() else int(sqft_value)
    return row, flags


# ── Financial values ───────────────────────────────────────────────────────────

_CURRENCY_STRIP = re.compile(r"[$€£¥₹,\s]")
_SHORTHAND = re.compile(r"^([\d.]+)\s*([KMBkmb])$")

def _parse_value(raw: Any) -> float:
    if raw is None or str(raw).strip() == "":
        return 0
    s = _CURRENCY_STRIP.sub("", str(raw).strip())
    
    if s == '-' or s.lower() in ('n/a', 'na', 'none', 'nil'):
        return 0
        
    m = _SHORTHAND.match(s)
    if m:
        num, suffix = float(m.group(1)), m.group(2).upper()
        multiplier = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suffix, 1)
        val = num * multiplier
        return int(round(val))
    try:
        val = float(s)
        return int(round(val))
    except ValueError:
        return 0

VALUE_FIELDS_AIR = [
    ("BuildingValue",    "BuildingValue",    "max_building_value"),
    ("ContentsValue",    "ContentsValue",    "max_contents_value"),
    ("TimeElementValue", "TimeElementValue", "max_bi_value"),
]

VALUE_FIELDS_RMS = [
    ("BuildingValue",    "EQCV1VAL",  "max_building_value"),
    ("ContentsValue",    "EQCV2VAL",  "max_contents_value"),
    ("TimeElementValue", "EQCV3VAL",  "max_bi_value"),
]

BUILDING_VARIANTS = [f"BuildingValue{i}" for i in range(1, 6)]  # BuildingValue1..5


def _normalize_values(row: dict, row_idx: int, rules: BusinessRulesConfig,
                      target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []
    value_fields = VALUE_FIELDS_AIR if target == "AIR" else VALUE_FIELDS_RMS

    for src_field, dest_field, max_attr in value_fields:
        raw = row.get(src_field)
        val = _parse_value(raw)
        if val is None:
            flags.append(_make_flag(row_idx, dest_field, "unparseable_value",
                                    raw, f"Cannot parse numeric value from '{raw}'"))
            row[dest_field] = None
            continue
        if val < 0:
            flags.append(_make_flag(row_idx, dest_field, "negative_value",
                                    val, f"{src_field} is negative: {val}"))
        max_val = getattr(rules, max_attr)
        if val > max_val:
            action = rules.invalid_value_action
            if action == "reset_value":
                row[dest_field] = None
                flags.append(_make_flag(row_idx, dest_field, "value_exceeds_max",
                                        val, f"{src_field}={val:,.0f} exceeds max {max_val:,.0f}; reset to None"))
            else:
                row[dest_field] = val
                if action == "flag_review":
                    flags.append(_make_flag(row_idx, dest_field, "value_exceeds_max",
                                            val, f"{src_field}={val:,.0f} exceeds configured max {max_val:,.0f}"))
        else:
            row[dest_field] = val

    return row, flags


# ── Currency ───────────────────────────────────────────────────────────────────

def _normalize_currency(row: dict, row_idx: int, valid_currencies: set) -> Tuple[dict, List[dict]]:
    flags = []
    currency_cols = [
        "Currency", "EQCV3LCUR", "WSCV3LCUR", "TOCV3LCUR",
        "HUCV3LCUR", "FPCV3LCUR", "TRCV3LCUR",
    ]
    found_currencies = set()

    for col in currency_cols:
        raw = row.get(col)
        if not raw:
            continue
        # Extract last 3 chars if embedded in value string
        s = str(raw).strip()[-3:].upper()
        if s in valid_currencies:
            found_currencies.add(s)
        else:
            # Try the whole value
            whole = str(raw).strip().upper()
            if whole in valid_currencies:
                found_currencies.add(whole)
            else:
                row[col] = None
                flags.append(_make_flag(row_idx, col, "unrecognized_currency",
                                        raw, f"Unrecognized currency code '{raw}' in {col}"))

    if len(found_currencies) > 1:
        flags.append(_make_flag(row_idx, "Currency", "currency_conflict",
                                list(found_currencies),
                                f"Currency conflict: multiple currencies found: {found_currencies}"))
        row["Currency_Conflicts"] = True
    elif found_currencies:
        row["Currency"] = list(found_currencies)[0]

    return row, flags


# ── Sprinkler ──────────────────────────────────────────────────────────────────

def _normalize_sprinkler(row: dict, row_idx: int,
                          target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []
    spk_key = "SprinklerSystem" if target == "AIR" else "SPRINKLER"
    raw = row.get("SprinklerSystem") or row.get("SPRINKLER") or row.get(spk_key)
    if raw is None:
        return row, flags
    s = str(raw).strip().lower()
    if s in SPRINKLER_TRUE_VALUES:
        row[spk_key] = 1
    elif s in SPRINKLER_FALSE_VALUES:
        row[spk_key] = 0
    else:
        row[spk_key] = None
        flags.append(_make_flag(row_idx, spk_key, "unrecognized_sprinkler",
                                raw, f"Unrecognized sprinkler value '{raw}'. Expected yes/no."))
    return row, flags


# ── Roof / Wall / Foundation / Soft-Story (AIR modifier fields) ────────────────

def _normalize_modifiers(row: dict, row_idx: int,
                          target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []

    roof_key = "RoofGeometry" if target == "AIR" else "ROOFGEOM"
    wall_key = "WallSiding" if target == "AIR" else "CLADDING"
    found_key = "FoundationType" if target == "AIR" else "FOUNDATION"
    soft_key = "SoftStory" if target == "AIR" else "SOFTSTORY"
    walltype_key = "WallType" if target == "AIR" else "WALLTYPE"

    mapper = get_mapper() # Default LLM enabled from API key, handled gracefully

    sub_row = {
        "roof_cover": row.get(roof_key),
        "wall_type": row.get(wall_key) or row.get(walltype_key),
        "foundation_type": row.get(found_key),
        "soft_story": row.get(soft_key),
    }

    result = mapper.map_all(sub_row)
    methods = result.get("_methods", {})

    if row.get(roof_key) is not None:
        row[roof_key] = result["roof_cover"]
        if methods.get("roof_cover") not in ("empty", "integer"):
            flags.append(_make_flag(row_idx, roof_key, "mapped_modifier",
                                    row.get(roof_key), f"Mapped roof cover via {methods.get('roof_cover')} → {result['roof_cover_desc']}"))

    # WallSiding (AIR) or CLADDING (RMS)
    if row.get(wall_key) is not None:
        row[wall_key] = result["wall_type"]
        if methods.get("wall_type") not in ("empty", "integer"):
            flags.append(_make_flag(row_idx, wall_key, "mapped_modifier",
                                    row.get(wall_key), f"Mapped wall type via {methods.get('wall_type')} → {result['wall_type_desc']}"))
    
    # AIR specifically also has WallType (same integer code schema as WallSiding)
    if target == "AIR" and row.get(walltype_key) is not None:
        row[walltype_key] = result["wall_type"]
        if methods.get("wall_type") not in ("empty", "integer"):
            flags.append(_make_flag(row_idx, walltype_key, "mapped_modifier",
                                    row.get(walltype_key), f"Mapped wall type via {methods.get('wall_type')} → {result['wall_type_desc']}"))

    if row.get(found_key) is not None:
        row[found_key] = result["foundation_type"]
        if methods.get("foundation_type") not in ("empty", "integer"):
            flags.append(_make_flag(row_idx, found_key, "mapped_modifier",
                                    row.get(found_key), f"Mapped foundation via {methods.get('foundation_type')} → {result['foundation_type_desc']}"))

    if row.get(soft_key) is not None:
        row[soft_key] = result["soft_story"]
        if methods.get("soft_story") not in ("empty", "integer"):
            flags.append(_make_flag(row_idx, soft_key, "mapped_modifier",
                                    row.get(soft_key), f"Mapped soft story via {methods.get('soft_story')} → {result['soft_story_desc']}"))

    return row, flags



# ── Location Name ──────────────────────────────────────────────────────────────

def _normalize_location_name(row: dict, row_idx: int,
                              target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []
    loc_key = "LocationName" if target == "AIR" else "LOCNAME"
    raw = row.get("LocationName") or row.get("LOCNAME") or row.get(loc_key)
    
    # Rule 2: Fallback if None
    if raw is None or str(raw).strip() == "":
        contract_key = "ContractID" if target == "AIR" else "ACCNTNUM"
        fallback = (row.get(contract_key) or 
                    row.get("AccountName") or 
                    row.get("SubmissionName") or 
                    row.get("InsuredName"))
        if fallback and str(fallback).strip() != "":
            raw = fallback
            flags.append(_make_flag(row_idx, loc_key, "location_name_fallback",
                                    raw, f"Location name missing; used Account/Policy Name: '{raw}'"))
    
    # Rule 3: Missing
    if raw is None or str(raw).strip() == "":
        row[loc_key] = None
        flags.append(_make_flag(row_idx, loc_key, "missing_location_name",
                                None, "Location name missing and no fallback found."))
        return row, flags

    # Rule 4: Clean to varchar(40) (allowed: A-Za-z0-9 .,&'-()/)
    s = str(raw).strip()
    import re
    s = re.sub(r"[^A-Za-z0-9 .,&'\-()/]", "", s)
    s = s.strip()
    
    if len(s) > 40:
        s_truncated = s[:40].strip()
        flags.append(_make_flag(row_idx, loc_key, "location_name_truncated",
                                raw, f"Location name truncated to 40 chars: '{s_truncated}'"))
        s = s_truncated
    
    row[loc_key] = s
    return row, flags


# ── Identity fields (bulk pass over all rows) ─────────────────────────────────

def _normalize_identity_fields(
    rows: List[Dict],
    target_format: str,
) -> Tuple[List[Dict], List[dict]]:
    """
    Bulk post-pass that enforces two rules:

    1. ContractID (AIR) / ACCNTNUM (RMS) — uniform value across all rows.
       The column_mapper has already resolved source aliases (e.g. "Policy Number"
       → ContractID) before reaching here, so we simply scan for the first
       non-blank value in the canonical key and propagate it to every row.

    2. LocationID (AIR) / LOCNUM (RMS) — serial 1-based integer per policy group.
       Rows are grouped by ContractID/ACCNTNUM; within each group they receive
       sequential numbers 1, 2, 3 … in the order they appear.
       Single-policy SOV files get a simple 1-to-N across all rows.
    """
    from collections import defaultdict

    contract_key = "ContractID" if target_format == "AIR" else "ACCNTNUM"
    location_key = "LocationID" if target_format == "AIR" else "LOCNUM"
    flags: List[dict] = []

    # ── Step 1: resolve uniform policy ID ─────────────────────────────────────
    # Scan for first non-blank value already under the canonical key.
    policy_id: Optional[str] = None
    for row in rows:
        v = row.get(contract_key)
        if v and str(v).strip():
            policy_id = str(v).strip()
            break

    if not policy_id:
        for idx in range(len(rows)):
            flags.append(_make_flag(
                idx, contract_key, "missing_policy_id", None,
                f"{contract_key} is blank for all rows. "
                f"Map a source column to '{contract_key}' in the column mapping step.",
            ))
    else:
        for row in rows:
            row[contract_key] = policy_id

    # ── Step 2: serial location numbering ─────────────────────────────────────
    # Group row indexes by policy ID, then assign 1-based counters per group.
    groups: dict = defaultdict(list)
    for idx, row in enumerate(rows):
        group_key = str(row.get(contract_key) or "").strip() or "__no_policy__"
        groups[group_key].append(idx)

    for group_indices in groups.values():
        for seq_num, row_idx in enumerate(group_indices, start=1):
            rows[row_idx][location_key] = seq_num

    return rows, flags


# ── Main entry point ───────────────────────────────────────────────────────────


def normalize_all_rows(
    rows: List[Dict[str, Any]],
    rules_config: BusinessRulesConfig,
    valid_currencies: Optional[set] = None,
    target_format: str = "AIR",
) -> Tuple[List[Dict[str, Any]], List[Dict]]:
    """
    Run all normalization steps over every row.
    Returns (normalized_rows, all_new_flags).
    """
    if valid_currencies is None:
        valid_currencies = set()

    all_flags: List[dict] = []
    normalized: List[Dict] = []

    for idx, row in enumerate(rows):
        row = dict(row)  # work on a copy

        row, f = _normalize_year(row, idx, rules_config, target_format)
        all_flags.extend(f)

        row, f = _normalize_upgraded_year(row, idx, rules_config, target_format)
        all_flags.extend(f)

        row, f = _normalize_stories(row, idx, rules_config, target_format)
        all_flags.extend(f)

        row, f = _normalize_building_count(row, idx, target_format)
        all_flags.extend(f)

        row, f = _normalize_area(row, idx, rules_config, target_format)
        all_flags.extend(f)

        row, f = _normalize_values(row, idx, rules_config, target_format)
        all_flags.extend(f)

        row, f = _normalize_currency(row, idx, valid_currencies)
        all_flags.extend(f)

        row, f = _normalize_sprinkler(row, idx, target_format)
        all_flags.extend(f)

        row, f = _normalize_modifiers(row, idx, target_format)
        all_flags.extend(f)

        row, f = _normalize_location_name(row, idx, target_format)
        all_flags.extend(f)

        normalized.append(row)

    # ── Bulk identity pass (runs after all per-row normalization) ───────────
    normalized, id_flags = _normalize_identity_fields(normalized, target_format)
    all_flags.extend(id_flags)

    return normalized, all_flags
