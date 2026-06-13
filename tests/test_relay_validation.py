"""Tests for schema_validator.py inter-agent relay validation."""

import json
import subprocess

from conftest import SCRIPTS_DIR

from schema_validator import validate_relay, RELAY_SCHEMAS


class TestRelaySchemas:
    """Structural invariants every RELAY_SCHEMAS entry must satisfy."""

    def test_all_routes_have_required_key(self):
        """Every schema has a 'required' list."""
        for route, schema in RELAY_SCHEMAS.items():
            assert "required" in schema, f"{route} missing 'required'"
            assert isinstance(schema["required"], list)

    def test_all_routes_have_optional_key(self):
        """Every schema has an 'optional' list."""
        for route, schema in RELAY_SCHEMAS.items():
            assert "optional" in schema, f"{route} missing 'optional'"

    def test_types_reference_valid_fields(self):
        """Type constraints only reference fields in required or optional."""
        for route, schema in RELAY_SCHEMAS.items():
            all_fields = set(schema["required"]) | set(schema["optional"])
            for field in schema.get("types", {}):
                assert field in all_fields, f"{route}: type for '{field}' not in fields"


class TestValidateRelay:
    """validate_relay checks relay payloads against their route schema."""

    def test_valid_analyze_to_tuning(self):
        """A complete analyze_to_tuning payload validates with no errors."""
        result = validate_relay("analyze_to_tuning", {
            "recommendation": "continue",
            "batch_number": 3,
            "correlations": {"lr": 0.85}
        })
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_required_field(self):
        """Omitting a required field invalidates the payload and names it."""
        result = validate_relay("analyze_to_tuning", {
            "batch_number": 3
        })
        assert result["valid"] is False
        assert any("recommendation" in e for e in result["errors"])

    def test_type_mismatch(self):
        """A field of the wrong type invalidates the payload and names it."""
        result = validate_relay("analyze_to_tuning", {
            "recommendation": "continue",
            "batch_number": "not_an_int"
        })
        assert result["valid"] is False
        assert any("batch_number" in e for e in result["errors"])

    def test_unknown_route(self):
        """An unrecognized route name is invalid."""
        result = validate_relay("nonexistent_route", {"foo": "bar"})
        assert result["valid"] is False

    def test_extra_fields_produce_warnings(self):
        """Unexpected extra fields warn but do not invalidate the payload."""
        result = validate_relay("analyze_to_tuning", {
            "recommendation": "continue",
            "batch_number": 3,
            "unexpected_field": "surprise"
        })
        assert result["valid"] is True  # warnings don't invalidate
        assert len(result["warnings"]) > 0

    def test_empty_required_fields_valid(self):
        """Routes with no required fields accept empty data."""
        result = validate_relay("monitor_to_tuning", {})
        assert result["valid"] is True

    def test_all_routes_accept_valid_data(self):
        """Smoke test: each route accepts its required fields."""
        test_data = {
            "analyze_to_tuning": {"recommendation": "continue", "batch_number": 1},
            "analyze_to_research": {"pivot_reason": "diminishing returns"},
            "monitor_to_tuning": {},
            "research_to_implement": {"findings_path": "reports/research-findings.md"},
            "experiments_to_analyze": {"completed_ids": ["exp-001"]},
        }
        for route, data in test_data.items():
            result = validate_relay(route, data)
            assert result["valid"] is True, f"{route} failed: {result['errors']}"

    def test_numeric_type_accepts_int_and_float(self):
        """best_metric_value accepts both int and float."""
        for val in [0.95, 95]:
            result = validate_relay("analyze_to_tuning", {
                "recommendation": "continue",
                "batch_number": 1,
                "best_metric_value": val
            })
            assert result["valid"] is True


class TestRelayCliIntegration:
    """The relay CLI subcommand validates payloads and signals via exit code."""

    def test_valid_cli_output(self, tmp_path):
        """CLI relay command returns valid JSON with exit code 0."""
        script = str(SCRIPTS_DIR / "schema_validator.py")
        data = json.dumps({"recommendation": "continue", "batch_number": 1})
        result = subprocess.run(
            ['python3', script, 'relay', 'analyze_to_tuning', data],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["valid"] is True

    def test_invalid_cli_output(self, tmp_path):
        """CLI relay command returns exit code 1 for invalid data."""
        script = str(SCRIPTS_DIR / "schema_validator.py")
        data = json.dumps({"batch_number": 1})  # missing recommendation
        result = subprocess.run(
            ['python3', script, 'relay', 'analyze_to_tuning', data],
            capture_output=True, text=True
        )
        assert result.returncode == 1
