import time
import numpy as np
from scipy.signal import fftconvolve

from .configs import SimConfig
from .physics import Aphi_coil, coulomb_kernel


class MITScanner:
    """Forward model: sigma -> flux measurement."""

    def __init__(self, sim: SimConfig):
        self.sim = sim
        self.grid = sim.grid
        self._precompute()

    def _precompute(self):
        g = self.grid
        Nx, Ny = g.Nx, g.Ny
        dx, dy, h = g.dx, g.dy, g.h_thickness

        x_rel = np.arange(-(Nx - 1), Nx) * dx
        y_rel = np.arange(-(Ny - 1), Ny) * dy
        X_rel, Y_rel = np.meshgrid(x_rel, y_rel, indexing="ij")

        R_rel = np.sqrt(X_rel**2 + Y_rel**2)
        Phi_rel = np.arctan2(Y_rel, X_rel)

        self.sin_phi_full = np.sin(Phi_rel)
        self.cos_phi_full = np.cos(Phi_rel)

        self.Aphi_tx_full = Aphi_coil(
            R_rel,
            self.sim.coil_tx.z_offset,
            self.sim.coil_tx
        )

        self.Aphi_rx_full = Aphi_coil(
            R_rel,
            self.sim.coil_rx.z_offset,
            self.sim.coil_rx
        )

        self.kernel = coulomb_kernel(Nx, Ny, dx, dy, h)

    def forward(self, sigma: np.ndarray) -> np.ndarray:
        """
        Compute forward model:
            sigma -> Flux map (complex Nx×Ny)
        """

        g = self.grid
        Nx, Ny = g.Nx, g.Ny
        dx, dy, h = g.dx, g.dy, g.h_thickness
        omega = self.sim.omega

        log_sigma = np.log(sigma)
        dlog_dx = np.gradient(log_sigma, dx, axis=0)
        dlog_dy = np.gradient(log_sigma, dy, axis=1)

        Flux = np.zeros((Nx, Ny), dtype=complex)
        dV = dx * dy * h

        for i in range(Nx):
            i0 = (Nx - 1) - i
            slx = slice(i0, i0 + Nx)

            for j in range(Ny):
                j0 = (Ny - 1) - j
                sly = slice(j0, j0 + Ny)

                # --- coil fields ---
                A_tx = self.Aphi_tx_full[slx, sly]
                A_rx = self.Aphi_rx_full[slx, sly]

                sin_phi = self.sin_phi_full[slx, sly]
                cos_phi = self.cos_phi_full[slx, sly]

                # --- primary electric field ---
                E1x = -1j * omega * (-A_tx * sin_phi)
                E1y = -1j * omega * ( A_tx * cos_phi)

                # --- secondary field (conductivity correction) ---
                rho = -(E1x * dlog_dx + E1y * dlog_dy)

                V2 = (
                    h / (4 * np.pi)
                ) * fftconvolve(rho, self.kernel, mode="same") * dx * dy

                dE2x = -np.gradient(V2, dx, axis=0)
                dE2y = -np.gradient(V2, dy, axis=1)

                # --- current density ---
                Jx = sigma * (E1x + dE2x)
                Jy = sigma * (E1y + dE2y)

                # --- receiver projection ---
                A_rx_x = -A_rx * sin_phi
                A_rx_y =  A_rx * cos_phi

                flux = np.sum(Jx * A_rx_x + Jy * A_rx_y) * dV
                Flux[i, j] = flux

        return Flux

    def compare(self, sigma_a, sigma_h):
        """Compare two conductivity maps (not physics core)."""

        flux_a = self.forward(sigma_a)
        flux_h = self.forward(sigma_h)

        V_diff = -1j * self.sim.omega * (flux_a - flux_h)

        return {
            "flux_anom": flux_a,
            "flux_healthy": flux_h,
            "V_diff": V_diff,
        }