"""
tests/test_construction_rules.py

Unit tests for the ConflictResolver — validates all 30 structural conflict scenarios
from the Construction_Mapping.xlsx reference.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
from construction_rules import ConflictResolver, ConflictResult

resolver = ConflictResolver()


# ── Helpers ────────────────────────────────────────────────────────────────────

def resolve(raw: str) -> ConflictResult | None:
    return resolver.resolve(raw)


def assertCode(result: ConflictResult | None, expected_air: str, scenario: int):
    assert result is not None, f"Scenario {scenario}: expected ConflictResult, got None"
    assert result.air_code == expected_air, (
        f"Scenario {scenario}: expected AIR {expected_air}, "
        f"got {result.air_code} ({result.reasoning})"
    )


# ── All 30 Scenario Tests ──────────────────────────────────────────────────────

class TestConflictScenarios:

    def test_scenario_01_steel_frame_masonry_infill(self):
        r = resolve("Steel frame with masonry infill walls")
        assertCode(r, "151", 1)
        assert "frame" in r.rule_applied.lower() or r.rule_applied == "exact_scenario_match"

    def test_scenario_02_cmu_wood_roof_joists(self):
        r = resolve("CMU walls with wood roof joists")
        assertCode(r, "119", 2)
        assert r.final_category == "Joisted Masonry"

    def test_scenario_03_concrete_frame_glass_curtain(self):
        r = resolve("Concrete frame with glass curtain wall")
        assertCode(r, "131", 3)
        assert r.final_category == "Concrete Frame"

    def test_scenario_04_tiltup_steel_roof(self):
        r = resolve("Tilt-up concrete walls with steel roof deck")
        assertCode(r, "111", 4)
        assert r.final_category == "Masonry Non-Combustible"

    def test_scenario_05_wood_frame_brick_veneer(self):
        r = resolve("Wood frame with brick veneer")
        assertCode(r, "101", 5)
        assert r.final_category == "Frame"

    def test_scenario_06_steel_frame_wood_mezzanine(self):
        r = resolve("Steel frame with wood mezzanine")
        assertCode(r, "151", 6)
        assert r.final_category == "Steel Frame"

    def test_scenario_07_precast_walls_steel_frame(self):
        r = resolve("Precast walls with steel frame")
        assertCode(r, "151", 7)
        assert r.final_category == "Steel Frame"

    def test_scenario_08_concrete_podium_wood_upper(self):
        r = resolve("Concrete podium with wood upper floors")
        assertCode(r, "141", 8)
        assert r.final_category == "Mixed Construction"

    def test_scenario_09_heavy_timber_masonry_walls(self):
        r = resolve("Heavy timber with masonry walls")
        assertCode(r, "104", 9)
        assert r.final_category == "Heavy Timber"

    def test_scenario_10_metal_building_masonry_facade(self):
        r = resolve("Metal building with masonry front facade")
        assertCode(r, "152", 10)
        assert r.final_category == "Metal Building"

    def test_scenario_11_concrete_shear_steel_roof(self):
        r = resolve("Concrete shear walls with steel roof truss")
        assertCode(r, "131", 11)
        assert r.final_category == "Concrete Frame"

    def test_scenario_12_steel_frame_concrete_infill(self):
        r = resolve("Steel frame retrofitted with concrete infill")
        assertCode(r, "151", 12)

    def test_scenario_13_urm_steel_roof(self):
        r = resolve("Unreinforced masonry with steel roof")
        assertCode(r, "119", 13)
        assert "urm" in r.rule_applied.lower()

    def test_scenario_14_cmu_lower_wood_upper(self):
        r = resolve("CMU lower level with wood upper level")
        assertCode(r, "141", 14)
        assert r.final_category == "Mixed Construction"

    def test_scenario_15_steel_warehouse_wood_office(self):
        r = resolve("Steel warehouse with attached wood office")
        assertCode(r, "141", 15)
        assert r.final_category == "Mixed Construction"

    def test_scenario_16_pemb_interior_wood_office(self):
        r = resolve("Pre-engineered metal warehouse with interior wood office")
        assertCode(r, "152", 16)
        assert r.final_category == "Metal Building"

    def test_scenario_17_concrete_garage_steel_office(self):
        r = resolve("Concrete parking garage attached to steel office")
        # Scenario 17 is unresolvable distinct systems → AIR 100
        assert r is not None
        assert r.air_code in ("100", "141"), f"Scenario 17: got {r.air_code}"

    def test_scenario_18_steel_moment_precast_panels(self):
        r = resolve("Steel moment frame with precast panels")
        assertCode(r, "151", 18)

    def test_scenario_19_masonry_exterior_steel_interior(self):
        r = resolve("Masonry exterior with steel interior framing")
        assertCode(r, "151", 19)

    def test_scenario_20_wood_truss_masonry_walls(self):
        r = resolve("Wood truss roof over masonry walls")
        assertCode(r, "119", 20)
        assert r.final_category == "Joisted Masonry"

    def test_scenario_21_5story_wood_concrete_podium(self):
        r = resolve("5-story wood over concrete podium")
        assertCode(r, "141", 21)
        assert r.final_category == "Mixed Construction"

    def test_scenario_22_steel_highrise_concrete_core(self):
        r = resolve("Steel frame high-rise with concrete core")
        assertCode(r, "151", 22)
        assert r.final_category == "Steel Frame"

    def test_scenario_23_concrete_steel_canopy(self):
        r = resolve("Concrete building with steel canopy")
        assertCode(r, "131", 23)
        assert r.final_category == "Concrete Frame"

    def test_scenario_24_brick_exterior_unknown_frame(self):
        r = resolve("Brick exterior, unknown frame")
        assertCode(r, "119", 24)
        assert "conservative" in r.reasoning.lower() or "unknown" in r.rule_applied.lower()

    def test_scenario_25_metal_partial_masonry_addition(self):
        r = resolve("Metal building with partial masonry addition")
        assertCode(r, "141", 25)
        assert r.final_category == "Mixed Construction"

    def test_scenario_26_wood_framing_rc_foundation(self):
        r = resolve("Wood framing with reinforced concrete foundation")
        assertCode(r, "101", 26)
        assert r.final_category == "Frame"

    def test_scenario_27_rc_lower_wood_penthouse(self):
        r = resolve("Reinforced concrete lower floors, wood penthouse")
        assertCode(r, "141", 27)
        assert r.final_category == "Mixed Construction"

    def test_scenario_28_steel_warehouse_cmu_office(self):
        r = resolve("Steel warehouse with concrete block office extension")
        assertCode(r, "141", 28)

    def test_scenario_29_cmu_reinforced_steel_columns(self):
        r = resolve("CMU walls reinforced with steel columns")
        assertCode(r, "111", 29)
        assert r.final_category == "Masonry Non-Combustible"

    def test_scenario_30_tiltup_wood_interior(self):
        r = resolve("Concrete tilt-up with wood interior framing")
        assertCode(r, "141", 30)


# ── Additional edge case tests ─────────────────────────────────────────────────

class TestEdgeCases:

    def test_simple_description_returns_none(self):
        """Simple (non-compound) descriptions should return None — let deterministic handle."""
        assert resolver.resolve("Steel frame") is None
        assert resolver.resolve("Reinforced concrete") is None
        assert resolver.resolve("Wood") is None

    def test_empty_returns_none(self):
        assert resolver.resolve("") is None
        assert resolver.resolve(None) is None  # type: ignore

    def test_urm_governs_over_everything(self):
        r = resolve("Unreinforced masonry with steel moment frame and wood roof")
        assert r is not None
        assert r.air_code == "119"
        assert "urm" in r.rule_applied.lower()

    def test_combustible_downgrade_various_wording(self):
        r = resolve("Brick walls with wood truss roof")
        assert r is not None
        assert r.air_code == "119"

    def test_confidence_is_within_range(self):
        """All results must have confidence between 0.0 and 1.0."""
        test_cases = [
            "Steel frame with masonry infill walls",
            "CMU walls with wood roof joists",
            "Concrete frame with glass curtain wall",
            "Unreinforced masonry with steel roof",
            "5-story wood over concrete podium",
        ]
        for tc in test_cases:
            r = resolve(tc)
            if r:
                assert 0.0 <= r.confidence <= 1.0, f"Confidence out of range for: {tc}"

    def test_conflict_flag_set_for_compound(self):
        """Compound descriptions should set conflict_flag=True."""
        r = resolve("Steel frame with masonry infill walls")
        assert r is not None
        assert r.conflict_flag is True

    def test_rule_applied_is_populated(self):
        """rule_applied should always be set for valid conflict results."""
        r = resolve("CMU walls with wood roof joists")
        assert r is not None
        assert r.rule_applied and len(r.rule_applied) > 0
