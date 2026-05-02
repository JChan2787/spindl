"""
Generic weighted rotation with decay.

Shared utility for template rotation (NANO-119/120) and model cycling
(NANO-121). Prevents repetition via weighted random selection with
exponential decay on the selected item.
"""

import random
from typing import Optional


class WeightedRotator:
    """Weighted random selector with decay-based anti-repetition.

    On each `select()`, the chosen item's weight is halved and the lost
    weight is redistributed evenly across all other items. Weights are
    normalized to sum=1.0 after every selection. Single-element lists
    skip the rotation machinery entirely.
    """

    def __init__(self, items: list[str], decay_factor: float = 0.5):
        self._items = list(items) if items else []
        self._decay_factor = decay_factor
        self._weights = self._uniform(len(self._items))

    @staticmethod
    def _uniform(n: int) -> list[float]:
        return [1.0 / n] * n if n > 0 else []

    @property
    def items(self) -> list[str]:
        return self._items

    @items.setter
    def items(self, value: list[str]) -> None:
        self._items = list(value) if value else []
        self._weights = self._uniform(len(self._items))

    @property
    def weights(self) -> list[float]:
        return list(self._weights)

    def reset(self) -> None:
        self._weights = self._uniform(len(self._items))

    def select(self) -> Optional[str]:
        if not self._items:
            return None
        if len(self._items) == 1:
            return self._items[0]

        idx = random.choices(range(len(self._items)), weights=self._weights, k=1)[0]
        selected = self._items[idx]

        original = self._weights[idx]
        self._weights[idx] = original * self._decay_factor
        lost = original - self._weights[idx]
        share = lost / (len(self._weights) - 1)
        for i in range(len(self._weights)):
            if i != idx:
                self._weights[i] += share
        total = sum(self._weights)
        self._weights = [w / total for w in self._weights]

        return selected
