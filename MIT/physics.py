import numpy as np
from scipy.signal import fftconvolve
from scipy.special import ellipk, ellipe

from .constants import MU_0
from .configs import CoilConfig


def Aphi_coil(rho, z, coil: CoilConfig) -> np.ndarray:
    """Vector potential produced by a circular loop."""
    rho = np.asarray(rho, dtype=float)
    A_phi = np.zeros_like(rho)

    mask = rho > 1e-12
    rho_m = rho[mask]
    a, I = coil.radius, coil.current

    denom = a**2 + rho_m**2 + z**2 + 2 * a * rho_m
    k2 = 4 * a * rho_m / denom
    K = ellipk(k2)
    E = ellipe(k2)
    factor = ((2 - k2) * K - 2 * E) / k2

    A_phi[mask] = (MU_0 * I * a / np.pi) * factor / np.sqrt(denom)
    return A_phi


def Aphi_to_cartesian(A_phi: np.ndarray, Phi: np.ndarray, R: np.ndarray):
    """Convert azimuthal A_phi into cartesian components Ax, Ay."""
    Ax = -A_phi * np.sin(Phi)
    Ay = A_phi * np.cos(Phi)
    Ax[R < 1e-15] = 0.0
    Ay[R < 1e-15] = 0.0
    return Ax, Ay


def primary_field(A_phi: np.ndarray, Phi: np.ndarray, R: np.ndarray, omega: float):
    """Primary electric field obtained from A_phi."""
    Ax, Ay = Aphi_to_cartesian(A_phi, Phi, R)
    return -1j * omega * Ax, -1j * omega * Ay


def coulomb_kernel(Nx: int, Ny: int, dx: float, dy: float, h: float) -> np.ndarray:
    """Coulomb kernel for 2D convolution."""
    kx = np.arange(-Nx + 1, Nx) * dx
    ky = np.arange(-Ny + 1, Ny) * dy
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return 1.0 / np.sqrt(KX**2 + KY**2 + (h / 2.0)**2)


def partial_correction(E1x, E1y, sigma, dx, dy, h):
    """First-order correction due to conductivity gradients."""
    log_sigma = np.log(sigma)
    dlogsig_dx = np.gradient(log_sigma, dx, axis=0)
    dlogsig_dy = np.gradient(log_sigma, dy, axis=1)
    rho_eps0 = -(E1x * dlogsig_dx + E1y * dlogsig_dy)

    Nx, Ny = sigma.shape
    kernel = coulomb_kernel(Nx, Ny, dx, dy, h)
    V2 = (h / (4.0 * np.pi)) * fftconvolve(rho_eps0, kernel, mode="same") * dx * dy

    delta_E2x = -np.gradient(V2, dx, axis=0)
    delta_E2y = -np.gradient(V2, dy, axis=1)
    return V2, delta_E2x, delta_E2y, rho_eps0