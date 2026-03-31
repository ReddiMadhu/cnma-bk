"""
tests/test_abbreviation_expansion.py

Unit tests for abbreviation expansion — validates all new entries
expand correctly to their full forms.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import json
import pytest
import re

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"


@pytest.fixture(scope="module")
def abbreviations():
    return json.loads((REF_DIR / "abbreviations.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def expand():
    from code_mapper import expand_abbreviations, build_tfidf_indexes
    try:
        from code_mapper import _abbreviations
        if not _abbreviations:
            build_tfidf_indexes()
    except Exception:
        pass
    return expand_abbreviations


class TestAbbreviationTable:

    def test_all_new_entries_present(self, abbreviations):
        """Verify all 40+ new abbreviations are in the table."""
        required = [
            "r/c", "reinf conc", "cast in place", "cip",
            "pt conc", "pt slab", "flat slab", "waffle slab",
            "prestress conc", "tunnel conc", "fr", "fire res",
            "fireproof conc", "intumescent",
            "wd", "wood frm", "wod frm", "lt wt frm",
            "ht", "hvy timber", "glulam", "clt", "cross lam timber", "post & beam",
            "mnc", "metal bldg", "pemb", "lgs", "light gauge steel",
            "steel joist", "owsj", "curtain wall",
            "tilt up", "tilt panel", "tbd",
            "earthen", "rammed earth", "conc",
        ]
        missing = [k for k in required if k not in abbreviations]
        assert not missing, f"Missing abbreviations: {missing}"

    def test_no_empty_expansions(self, abbreviations):
        for abbrev, full in abbreviations.items():
            assert full.strip(), f"Abbreviation '{abbrev}' has empty expansion"

    def test_expansions_are_strings(self, abbreviations):
        for abbrev, full in abbreviations.items():
            assert isinstance(full, str), f"Expansion for '{abbrev}' is not a string"


class TestExpansionFunction:

    def test_pemb_expands(self, expand):
        result, was_expanded = expand("PEMB warehouse")
        assert was_expanded
        assert "pre-engineered metal building" in result.lower() or "metal building" in result.lower()

    def test_cmu_expands(self, expand):
        result, was_expanded = expand("CMU walls")
        assert was_expanded
        assert "concrete masonry unit" in result.lower()

    def test_lgs_expands(self, expand):
        result, was_expanded = expand("LGS framing")
        assert was_expanded
        assert "light gauge steel" in result.lower()

    def test_owsj_expands(self, expand):
        result, was_expanded = expand("OWSJ roof structure")
        assert was_expanded
        assert "steel joist" in result.lower()

    def test_clt_expands(self, expand):
        result, was_expanded = expand("CLT panels")
        assert was_expanded
        assert "laminated timber" in result.lower()

    def test_ht_expands(self, expand):
        result, was_expanded = expand("HT structure")
        assert was_expanded
        assert "timber" in result.lower()

    def test_mnc_expands(self, expand):
        result, was_expanded = expand("MNC building")
        assert was_expanded
        assert "masonry non-combustible" in result.lower()

    def test_rc_expands(self, expand):
        result, was_expanded = expand("RC frame building")
        assert was_expanded
        assert "reinforced concrete" in result.lower()

    def test_r_slash_c_expands(self, expand):
        result, was_expanded = expand("R/C construction")
        assert was_expanded
        assert "reinforced concrete" in result.lower()

    def test_reinf_conc_expands(self, expand):
        result, was_expanded = expand("REINF CONC frame")
        assert was_expanded
        assert "reinforced concrete" in result.lower()

    def test_flat_slab_expands(self, expand):
        result, was_expanded = expand("Flat Slab building")
        assert was_expanded
        assert "concrete" in result.lower()

    def test_fr_expands(self, expand):
        result, was_expanded = expand("FR construction")
        assert was_expanded
        assert "fire resistive" in result.lower()

    def test_tilt_up_expands(self, expand):
        result, was_expanded = expand("Tilt Up construction")
        assert was_expanded
        assert "tilt-up concrete" in result.lower() or "tilt" in result.lower()

    def test_tbd_expands(self, expand):
        result, was_expanded = expand("TBD construction")
        assert was_expanded
        assert "unknown" in result.lower()

    def test_was_expanded_false_for_clean_input(self, expand):
        _, was_expanded = expand("reinforced concrete frame")
        assert not was_expanded

    def test_no_double_expansion(self, expand):
        """Expanding already-expanded text should not change it further."""
        result1, _ = expand("PEMB warehouse")
        result2, _ = expand(result1)
        assert result1.lower() == result2.lower(), "Double expansion produced different text"

    def test_empty_string(self, expand):
        result, was_expanded = expand("")
        assert result == ""
        assert not was_expanded

    def test_case_insensitive(self, expand):
        r1, e1 = expand("pemb")
        r2, e2 = expand("PEMB")
        r3, e3 = expand("Pemb")
        assert e1 == e2 == e3 == True
        assert r1.lower() == r2.lower() == r3.lower()
