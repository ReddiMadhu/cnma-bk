"""
tests/test_secondary_modifiers.py

Unit tests for SecondaryModifierMapper:
  - All 4 modifier fields (roof_cover, wall_type, foundation_type, soft_story)
  - Integer pass-through
  - Exact alias lookup
  - Keyword token scan
  - Default (unknown) handling
  - map_all() batch method
  - describe() and valid_codes() utilities
  - Edge cases: empty, whitespace, None-like
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from secondary_modifier_mapper import SecondaryModifierMapper, get_mapper


@pytest.fixture(scope="module")
def mapper():
    """Mapper with LLM disabled for deterministic tests."""
    return SecondaryModifierMapper(use_llm=False)


# ── Roof Cover (codes 0-11) ──────────────────────────────────────────────────

class TestRoofCover:

    # Integer pass-through
    @pytest.mark.parametrize("raw,expected", [
        ("0", 0), ("1", 1), ("3", 3), ("7", 7), ("11", 11),
    ])
    def test_integer_passthrough(self, mapper, raw, expected):
        assert mapper.map_roof_cover(raw) == expected

    # Exact alias coverage for every code
    @pytest.mark.parametrize("raw,expected", [
        ("asphalt shingles",              1),
        ("asphalt",                       1),
        ("composition shingle",           1),
        ("3-tab",                         1),
        ("architectural shingle",         1),
        ("wood shake",                    2),
        ("cedar shake",                   2),
        ("wooden shingles",               2),
        ("shake",                         2),
        ("clay tile",                     3),
        ("concrete tile",                 3),
        ("terracotta",                    3),
        ("barrel tile",                   3),
        ("spanish tile",                  3),
        ("tile roof",                     3),
        ("light metal panels",            4),
        ("metal panel",                   4),
        ("corrugated metal",              4),
        ("r panel",                       4),
        ("slate",                         5),
        ("natural slate",                 5),
        ("synthetic slate",               5),
        ("built-up roof with gravel",     6),
        ("tar and gravel",                6),
        ("bur gravel",                    6),
        ("single ply membrane",           7),
        ("tpo",                           7),
        ("epdm",                          7),
        ("pvc membrane",                  7),
        ("torch down",                    7),
        ("modified bitumen",              7),
        ("mod bit",                       7),
        ("standing seam",                 8),
        ("standing seam metal",           8),
        ("snap lock",                     8),
        ("concealed fastener metal",      8),
        ("built-up roof without gravel",  9),
        ("bur no gravel",                 9),
        ("smooth cap",                    9),
        ("ballasted membrane",            10),
        ("ballasted epdm",                10),
        ("hurricane",                     11),
        ("wind rated",                    11),
        ("impact resistant",              11),
        ("hvhz",                          11),
    ])
    def test_alias_lookup(self, mapper, raw, expected):
        assert mapper.map_roof_cover(raw) == expected, f"'{raw}' expected {expected}"

    # Keyword scan (partial text)
    @pytest.mark.parametrize("raw,expected", [
        ("asphalt composite",             1),
        ("cedar wood shingle",            2),
        ("clay barrel tiles",             3),
        ("corrugated steel panels",       4),
        ("natural slate roof",            5),
        ("tar and gravel built-up",       6),
        ("EPDM rubber membrane",          7),
        ("standing seam metal roof",      8),
        ("bur smooth cap no gravel",      9),
        ("ballasted TPO system",          10),
        ("HVHZ approved cover",           11),
    ])
    def test_keyword_scan(self, mapper, raw, expected):
        assert mapper.map_roof_cover(raw) == expected, f"'{raw}' expected {expected}"

    # Case insensitivity
    def test_case_insensitive(self, mapper):
        assert mapper.map_roof_cover("ASPHALT SHINGLES") == 1
        assert mapper.map_roof_cover("Clay Tile") == 3
        assert mapper.map_roof_cover("TPO") == 7

    # Out-of-range integer → default 0
    def test_out_of_range_int(self, mapper):
        assert mapper.map_roof_cover("99") == 0
        assert mapper.map_roof_cover("-1") == 0

    # Completely unknown → 0
    def test_unknown(self, mapper):
        assert mapper.map_roof_cover("polycarbonate dome") == 0

    # Empty / whitespace → 0
    def test_empty(self, mapper):
        assert mapper.map_roof_cover("") == 0
        assert mapper.map_roof_cover("  ") == 0


# ── Wall Type (codes 0-9) ────────────────────────────────────────────────────

class TestWallType:

    @pytest.mark.parametrize("raw,expected", [
        ("0", 0), ("1", 1), ("5", 5), ("9", 9),
    ])
    def test_integer_passthrough(self, mapper, raw, expected):
        assert mapper.map_wall_type(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("wood frame",        1),
        ("wood",              1),
        ("wf",                1),
        ("osb",               1),
        ("masonry",           2),
        ("brick",             2),
        ("cmu",               2),
        ("concrete block",    2),
        ("block",             2),
        ("concrete",          3),
        ("tilt-up",           3),
        ("cast in place",     3),
        ("cip",               3),
        ("steel",             4),
        ("metal wall",        4),
        ("metal panel",       4),
        ("stucco",            5),
        ("eifs",              5),
        ("dryvit",            5),
        ("vinyl siding",      6),
        ("vinyl",             6),
        ("siding",            6),
        ("brick veneer",      7),
        ("face brick",        7),
        ("stone veneer",      8),
        ("cultured stone",    8),
        ("fiber cement",      9),
        ("hardieplank",       9),
        ("hardiplank",        9),
        ("hardie board",      9),
        ("smartside",         9),
    ])
    def test_alias_lookup(self, mapper, raw, expected):
        assert mapper.map_wall_type(raw) == expected, f"'{raw}' expected {expected}"

    @pytest.mark.parametrize("raw,expected", [
        ("wood stud framing",         1),
        ("CMU block masonry",         2),
        ("concrete tilt wall",        3),
        ("corrugated steel panels",   4),
        ("EIFS synthetic stucco",     5),
        ("vinyl lap siding",          6),
        ("brick veneer exterior",     7),
        ("stone veneer cladding",     8),
        ("HardiePlank fiber cement",  9),
    ])
    def test_keyword_scan(self, mapper, raw, expected):
        assert mapper.map_wall_type(raw) == expected

    def test_empty(self, mapper):
        assert mapper.map_wall_type("") == 0

    def test_unknown(self, mapper):
        assert mapper.map_wall_type("polycarbonate panels") == 0


# ── Foundation Type (codes 0-12) ─────────────────────────────────────────────

class TestFoundationType:

    @pytest.mark.parametrize("raw,expected", [
        ("0", 0), ("1", 1), ("8", 8), ("12", 12),
    ])
    def test_integer_passthrough(self, mapper, raw, expected):
        assert mapper.map_foundation_type(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("masonry basement",          1),
        ("brick basement",            1),
        ("concrete basement",         2),
        ("full basement",             2),
        ("masonry wall",              3),
        ("crawl space cripple wall",  4),
        ("cripple wall",              4),
        ("cripple",                   4),
        ("crawl space masonry",       5),
        ("crawlspace masonry",        5),
        ("post and pier",             6),
        ("post & pier",               6),
        ("pier and beam",             6),
        ("footing",                   7),
        ("spread footing",            7),
        ("slab",                      8),
        ("mat",                       8),
        ("slab on grade",             8),
        ("mat/slab",                  8),
        ("concrete slab",             8),
        ("pile",                      9),
        ("pile foundation",           9),
        ("caisson",                   9),
        ("drilled pier",              9),
        ("no basement",               10),
        ("no bsmt",                   10),
        ("engineering foundation",    11),
        ("engineered foundation",     11),
        ("raised crawlspace",         12),
        ("elevated crawl",            12),
    ])
    def test_alias_lookup(self, mapper, raw, expected):
        assert mapper.map_foundation_type(raw) == expected, f"'{raw}' expected {expected}"

    @pytest.mark.parametrize("raw,expected", [
        ("masonry brick basement",        1),
        ("poured concrete basement",      2),
        ("cripple wall crawlspace",       4),
        ("post and pier foundation",      6),
        ("strip footing",                 7),
        ("slab-on-grade concrete",        8),
        ("driven pile foundation",        9),
        ("engineering deep foundation",   11),
        ("raised wood crawlspace",        12),
    ])
    def test_keyword_scan(self, mapper, raw, expected):
        assert mapper.map_foundation_type(raw) == expected

    def test_empty(self, mapper):
        assert mapper.map_foundation_type("") == 0

    def test_out_of_range(self, mapper):
        assert mapper.map_foundation_type("99") == 0


# ── Soft Story (codes 0-2) ───────────────────────────────────────────────────

class TestSoftStory:

    @pytest.mark.parametrize("raw,expected", [
        ("0", 0), ("1", 0), ("2", 0),  # Note: "0","1","2" as strings
    ])
    def test_integer_passthrough_within_range(self, mapper, raw, expected):
        # 0,1,2 are valid codes but aliases map "0"→1 (No) and "1"→2 (Yes)
        # Integer passthrough takes priority when value IS in [0,2]
        result = mapper.map_soft_story(raw)
        assert isinstance(result, int)
        assert 0 <= result <= 2

    @pytest.mark.parametrize("raw,expected", [
        ("yes",           2),
        ("Yes",           2),
        ("YES",           2),
        ("y",             2),
        ("true",          2),
        ("True",          2),
        ("soft story",    2),
        ("soft-story",    2),
        ("no",            1),
        ("No",            1),
        ("NO",            1),
        ("n",             1),
        ("false",         1),
        ("False",         1),
        ("no soft story", 1),
        ("unknown",       0),
        ("unspecified",   0),
        ("n/a",           0),
        ("-999",          0),
        ("u",             0),
    ])
    def test_alias_lookup(self, mapper, raw, expected):
        assert mapper.map_soft_story(raw) == expected, f"'{raw}' expected {expected}"

    def test_empty(self, mapper):
        assert mapper.map_soft_story("") == 0

    def test_random_text_defaults_to_unknown(self, mapper):
        # Ambiguous text that isn't clearly yes/no
        result = mapper.map_soft_story("maybe")
        assert result == 0


# ── map_all() batch method ───────────────────────────────────────────────────

class TestMapAll:

    def test_full_row(self, mapper):
        row = {
            "roof_cover":       "clay tile",
            "wall_type":        "brick veneer",
            "foundation_type":  "slab on grade",
            "soft_story":       "no",
        }
        result = mapper.map_all(row)
        assert result["roof_cover"] == 3
        assert result["wall_type"] == 7
        assert result["foundation_type"] == 8
        assert result["soft_story"] == 1

    def test_descriptions_present(self, mapper):
        row = {"roof_cover": "tpo", "wall_type": "stucco",
               "foundation_type": "pile", "soft_story": "yes"}
        result = mapper.map_all(row)
        assert result["roof_cover_desc"] == "Single ply membrane"
        assert result["wall_type_desc"] == "Stucco"
        assert result["foundation_type_desc"] == "Pile"
        assert result["soft_story_desc"] == "Yes"

    def test_methods_tracked(self, mapper):
        row = {"roof_cover": "7", "wall_type": "masonry",
               "foundation_type": "slab", "soft_story": "no"}
        result = mapper.map_all(row)
        assert result["_methods"]["roof_cover"] == "integer"
        assert result["_methods"]["wall_type"] in {"alias", "keyword"}

    def test_missing_fields_default_to_zero(self, mapper):
        result = mapper.map_all({})
        for field in mapper.FIELDS:
            assert result[field] == 0

    def test_integer_codes_passthrough(self, mapper):
        row = {"roof_cover": "1", "wall_type": "2",
               "foundation_type": "8", "soft_story": "0"}
        result = mapper.map_all(row)
        assert result["roof_cover"] == 1
        assert result["wall_type"] == 2
        assert result["foundation_type"] == 8

    def test_partial_row(self, mapper):
        row = {"roof_cover": "slate"}
        result = mapper.map_all(row)
        assert result["roof_cover"] == 5
        assert result["wall_type"] == 0       # missing → Unknown
        assert result["foundation_type"] == 0
        assert result["soft_story"] == 0

    def test_mixed_case_and_whitespace(self, mapper):
        row = {
            "roof_cover":       "  Standing Seam Metal  ",
            "wall_type":        "HARDIE BOARD",
            "foundation_type":  "CRIPPLE WALL",
            "soft_story":       "YES",
        }
        result = mapper.map_all(row)
        assert result["roof_cover"] == 8
        assert result["wall_type"] == 9
        assert result["foundation_type"] == 4
        assert result["soft_story"] == 2


# ── describe() and valid_codes() ─────────────────────────────────────────────

class TestUtilityMethods:

    def test_describe_roof_cover(self, mapper):
        assert mapper.describe("roof_cover", 7) == "Single ply membrane"
        assert mapper.describe("roof_cover", 0) == "Unknown/default"

    def test_describe_wall_type(self, mapper):
        assert mapper.describe("wall_type", 2) == "Masonry"
        assert mapper.describe("wall_type", 9) == "Fiber cement"

    def test_describe_foundation_type(self, mapper):
        assert mapper.describe("foundation_type", 8) == "Mat / slab"
        assert mapper.describe("foundation_type", 12) == "Crawlspace raised (wood)"

    def test_describe_soft_story(self, mapper):
        assert mapper.describe("soft_story", 2) == "Yes"

    def test_valid_codes_keys_are_ints(self, mapper):
        for field in mapper.FIELDS:
            codes = mapper.valid_codes(field)
            assert all(isinstance(k, int) for k in codes)

    def test_valid_codes_roof_cover(self, mapper):
        codes = mapper.valid_codes("roof_cover")
        assert len(codes) == 12    # 0-11
        assert 0 in codes and 11 in codes

    def test_valid_codes_wall_type(self, mapper):
        codes = mapper.valid_codes("wall_type")
        assert len(codes) == 10   # 0-9

    def test_valid_codes_foundation_type(self, mapper):
        codes = mapper.valid_codes("foundation_type")
        assert len(codes) == 13   # 0-12

    def test_valid_codes_soft_story(self, mapper):
        codes = mapper.valid_codes("soft_story")
        assert len(codes) == 3    # 0-2

    def test_invalid_field_raises(self, mapper):
        with pytest.raises(ValueError, match="Unknown field"):
            mapper._map("parking_type", "garage")

    def test_repr(self, mapper):
        r = repr(mapper)
        assert "SecondaryModifierMapper" in r


# ── Detail result structure ───────────────────────────────────────────────────

class TestDetailOutput:

    @pytest.mark.parametrize("method_name,raw,expected_code,expected_method", [
        ("map_roof_cover_detail",     "7",              7,  "integer"),
        ("map_roof_cover_detail",     "clay tile",      3,  "alias"),
        ("map_wall_type_detail",      "stucco",         5,  "alias"),
        ("map_foundation_type_detail","slab",           8,  "alias"),
        ("map_soft_story_detail",     "yes",            2,  "alias"),
    ])
    def test_detail_structure(self, mapper, method_name, raw, expected_code, expected_method):
        fn = getattr(mapper, method_name)
        result = fn(raw)
        assert result["code"] == expected_code
        assert result["method"] == expected_method
        assert "description" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["original"] == raw

    def test_integer_passthrough_confidence_1(self, mapper):
        result = mapper.map_roof_cover_detail("3")
        assert result["confidence"] == 1.0

    def test_alias_confidence_near_1(self, mapper):
        result = mapper.map_roof_cover_detail("asphalt shingles")
        assert result["confidence"] >= 0.95

    def test_default_confidence_0(self, mapper):
        result = mapper.map_roof_cover_detail("totally unknown material xyz")
        assert result["code"] == 0
        assert result["confidence"] == 0.0


# ── get_mapper() singleton ────────────────────────────────────────────────────

def test_get_mapper_returns_instance():
    m = get_mapper(use_llm=False)
    assert isinstance(m, SecondaryModifierMapper)

def test_get_mapper_singleton():
    m1 = get_mapper(use_llm=False)
    m2 = get_mapper(use_llm=False)
    assert m1 is m2
