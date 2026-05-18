"""Tests for WeightedRotator."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.stimuli.weighted_rotator import WeightedRotator


class TestWeightedRotatorInit:
    def test_uniform_weights(self):
        r = WeightedRotator(["a", "b", "c"])
        w = r.weights
        assert len(w) == 3
        assert all(abs(x - 1 / 3) < 1e-9 for x in w)

    def test_empty_items(self):
        r = WeightedRotator([])
        assert r.select() is None
        assert r.weights == []

    def test_single_item(self):
        r = WeightedRotator(["only"])
        assert r.select() == "only"
        assert r.weights == [1.0]

    def test_items_property(self):
        r = WeightedRotator(["a", "b"])
        assert r.items == ["a", "b"]


class TestWeightedRotatorSelect:
    def test_single_element_skips_decay(self):
        r = WeightedRotator(["only"])
        for _ in range(10):
            assert r.select() == "only"
        assert r.weights == [1.0]

    def test_select_returns_valid_item(self):
        items = ["a", "b", "c"]
        r = WeightedRotator(items)
        for _ in range(20):
            result = r.select()
            assert result in items

    def test_decay_changes_weights(self):
        r = WeightedRotator(["a", "b", "c"])
        initial = r.weights[:]
        r.select()
        assert r.weights != initial

    def test_weights_sum_to_one(self):
        r = WeightedRotator(["a", "b", "c", "d"])
        for _ in range(50):
            r.select()
            total = sum(r.weights)
            assert abs(total - 1.0) < 1e-9

    def test_distribution_not_dominated(self):
        """Over many selections, no single item should dominate >60%."""
        r = WeightedRotator(["a", "b", "c"])
        counts = {"a": 0, "b": 0, "c": 0}
        n = 300
        for _ in range(n):
            counts[r.select()] += 1
        for item, count in counts.items():
            assert count < n * 0.6, f"{item} dominated with {count}/{n}"

    def test_no_none_from_populated_list(self):
        r = WeightedRotator(["x", "y"])
        for _ in range(50):
            assert r.select() is not None


class TestWeightedRotatorReset:
    def test_reset_restores_uniform(self):
        r = WeightedRotator(["a", "b", "c"])
        for _ in range(20):
            r.select()
        r.reset()
        w = r.weights
        assert all(abs(x - 1 / 3) < 1e-9 for x in w)


class TestWeightedRotatorItemsSetter:
    def test_setter_resets_weights(self):
        r = WeightedRotator(["a", "b"])
        for _ in range(10):
            r.select()
        r.items = ["x", "y", "z"]
        assert r.items == ["x", "y", "z"]
        w = r.weights
        assert len(w) == 3
        assert all(abs(x - 1 / 3) < 1e-9 for x in w)

    def test_setter_empty_list(self):
        r = WeightedRotator(["a", "b"])
        r.items = []
        assert r.items == []
        assert r.select() is None

    def test_setter_single_item(self):
        r = WeightedRotator(["a", "b", "c"])
        r.items = ["only"]
        assert r.select() == "only"
