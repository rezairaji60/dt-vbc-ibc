from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'
DATA = RESULTS / 'data'

for p in (FIGURES, DATA):
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@dataclass
class Grid2D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    n_x: int
    n_y: int

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        xs = np.linspace(self.x_min, self.x_max, self.n_x)
        ys = np.linspace(self.y_min, self.y_max, self.n_y)
        return np.meshgrid(xs, ys)

    def points(self) -> np.ndarray:
        xx, yy = self.mesh()
        return np.column_stack([xx.ravel(), yy.ravel()])
