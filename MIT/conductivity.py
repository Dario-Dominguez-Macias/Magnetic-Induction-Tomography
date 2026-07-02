import numpy as np
from .configs import GridConfig


def make_sigma_map(grid: GridConfig, X, Y, sigma_background: float = 0.55, anomalies: list | None = None,) -> np.ndarray:
    """Create a conductivity map with elliptical anomalies."""
    sigma = np.full((grid.Nx, grid.Ny), sigma_background)

    if anomalies:
        for anom in anomalies:
            mask = (
                (X - anom["x_c"])**2 / anom["a"]**2
                + (Y - anom["y_c"])**2 / anom["b"]**2
            ) <= 1.0
            sigma[mask] = anom["sigma_val"]

    return sigma


### Add more conductivities