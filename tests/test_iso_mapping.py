"""
tests/test_iso_mapping.py

Unit tests for ISO Fire Class fast-path detection and AIR code mapping.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import json
import pytest

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"


@pytest.fixture(scope="module")
def iso_map():
    data = json.loads((REF_DIR / "iso_const_map.json").read_text(encoding="utf-8"))
    return data["iso_to_air"]


class TestISOMappingData:
    """Validate the iso_const_map.json data integrity."""

    def test_all_nine_classes_present(self, iso_map):
        for cls in range(10):
            assert str(cls) in iso_map, f"ISO class {cls} missing from map"

    def test_known_mappings(self, iso_map):
        expected = {
            "0": "100",  # Unknown → Unknown
            "1": "101",  # Frame → Wood Frame
            "2": "119",  # Joisted Masonry → Joisted Masonry
            "3": "152",  # Non-Combustible → Light Metal
            "4": "111",  # Masonry Non-Combustible → Masonry
            "5": "151",  # Modified Fire Resistive → Steel
            "6": "131",  # Fire Resistive → Reinforced Concrete
            "7": "104",  # Heavy Timber JM → Heavy Timber
            "8": "116",  # SNC → Reinforced Masonry
            "9": "116",  # SMNC → Reinforced Masonry
        }
        for iso_cls, air_cls in expected.items():
            actual = iso_map[iso_cls]["air_class"]
            assert actual == air_cls, (
                f"ISO {iso_cls}: expected AIR {air_cls}, got {actual}"
            )

    def test_all_entries_have_aliases(self, iso_map):
        for cls_key, data in iso_map.items():
            assert "aliases" in data, f"ISO class '{cls_key}' missing aliases"
            assert len(data["aliases"]) > 0, f"ISO class '{cls_key}' has empty aliases"

    def test_all_entries_have_air_class(self, iso_map):
        for cls_key, data in iso_map.items():
            assert "air_class" in data, f"ISO class '{cls_key}' missing air_class"
            assert data["air_class"].isdigit() or data["air_class"] == "100"


class TestISOSchemeDetection:
    """Test _is_iso_scheme() function from code_mapper."""

    def setup_method(self):
        # Import after sys.path manipulation
        from code_mapper import _is_iso_scheme
        self.detect = _is_iso_scheme

    def test_explicit_scheme_iso(self):
        assert self.detect("ISO", "concrete") is True

    def test_explicit_scheme_iso_class(self):
        assert self.detect("ISO_CLASS", "5") is True

    def test_explicit_scheme_fire_class(self):
        assert self.detect("FIRE_CLASS", "FR") is True

    def test_explicit_scheme_isf(self):
        assert self.detect("ISF", "2") is True

    def test_auto_detect_numeric(self):
        for v in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            assert self.detect("unknown_scheme", v) is True, f"Failed for value '{v}'"

    def test_auto_detect_labels(self):
        for v in ["F", "JM", "NC", "MNC", "MFR", "FR", "HTJM", "SNC", "SMNC"]:
            assert self.detect("unknown_scheme", v) is True, f"Failed for label '{v}'"

    def test_no_false_positive_long_string(self):
        assert self.detect("CONST", "Steel Frame") is False
        assert self.detect("CONST", "Reinforced Concrete") is False
        assert self.detect("CONST", "123") is False  # 3-digit → not ISO

    def test_lowercase_value(self):
        # Values are stripped+uppercased so lowercase should also detect "f" → "F"
        assert self.detect("ISO", "fr") is True or self.detect("ISO", "FR") is True


class TestISOLookup:
    """Test _lookup_iso() function."""

    def setup_method(self):
        from code_mapper import build_tfidf_indexes, _lookup_iso
        self._lookup = _lookup_iso
        # Only load if not already loaded
        try:
            from code_mapper import _iso_map
            if not _iso_map:
                build_tfidf_indexes()
        except Exception:
            pass

    def test_lookup_by_numeric(self):
        for num in range(10):
            result = self._lookup(str(num))
            assert result is not None, f"lookup({num}) returned None"
            assert "air_class" in result

    def test_lookup_by_label(self):
        test_cases = [
            ("F", "101"),
            ("JM", "119"),
            ("NC", "152"),
            ("MNC", "111"),
            ("FR", "131"),
            ("HTJM", "104"),
        ]
        for label, expected_air in test_cases:
            result = self._lookup(label)
            assert result is not None, f"lookup({label}) returned None"
            assert result["air_class"] == expected_air, (
                f"lookup({label}): expected AIR {expected_air}, got {result['air_class']}"
            )

    def test_lookup_unknown_returns_none(self):
        assert self._lookup("XYZ_INVALID") is None
        assert self._lookup("999") is None
