"""Tests for the reasoning budget."""

from types import SimpleNamespace

import pytest

from gemini_review.thinking import DISABLED, DYNAMIC, build_thinking_config, resolve_thinking_budget


class FakeTypes:
    """Stands in for google.genai.types so the tests need no client."""

    @staticmethod
    def ThinkingConfig(**kwargs):
        return SimpleNamespace(**kwargs)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)


class TestResolveThinkingBudget:
    def test_unset_leaves_the_api_default_alone(self):
        assert resolve_thinking_budget() is None
        assert resolve_thinking_budget({"thinking_budget": ""}) is None

    @pytest.mark.parametrize("value,expected", [("4096", 4096), ("-1", DYNAMIC), ("0", DISABLED), (2048, 2048)])
    def test_valid_values(self, value, expected):
        assert resolve_thinking_budget({"thinking_budget": value}) == expected

    def test_env_wins_over_config(self, monkeypatch):
        monkeypatch.setenv("GEMINI_THINKING_BUDGET", "512")
        assert resolve_thinking_budget({"thinking_budget": "8192"}) == 512

    @pytest.mark.parametrize("bad", ["lots", "4k", "-99"])
    def test_bad_values_fall_back_to_the_default_rather_than_raising(self, bad):
        """A typo in an optional knob must not fail a review."""
        assert resolve_thinking_budget({"thinking_budget": bad}) is None


class TestBuildThinkingConfig:
    def test_returns_none_when_unset(self):
        assert build_thinking_config(FakeTypes) is None

    def test_builds_a_config_with_the_budget(self):
        cfg = build_thinking_config(FakeTypes, {"thinking_budget": "4096"})
        assert cfg.thinking_budget == 4096

    def test_zero_is_an_explicit_disable_not_an_absence(self):
        cfg = build_thinking_config(FakeTypes, {"thinking_budget": "0"})
        assert cfg is not None and cfg.thinking_budget == 0
