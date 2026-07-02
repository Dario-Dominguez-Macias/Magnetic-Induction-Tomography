from dataclasses import dataclass, field
import numpy as np


@dataclass
class CoilConfig:
    """Parameters of a Tx or Rx coil."""
    radius: float
    z_offset: float
    current: float = 1.0


@dataclass
class GridConfig:
    """Configuration of the 2D spatial grid."""
    Lx: float = 0.06
    Ly: float = 0.06
    Nx: int = 24
    Ny: int = 24

    @property
    def dx(self) -> float:
        return self.Lx / self.Nx

    @property
    def dy(self) -> float:
        return self.Ly / self.Ny

    @property
    def h_thickness(self) -> float:
        return self.dx

    def make_mesh(self):
        """
        Return x, y, X, Y, R, Phi for voxel centers.
        """
        x = np.linspace(-self.Lx / 2 + self.dx / 2, self.Lx / 2 - self.dx / 2, self.Nx)
        y = np.linspace(-self.Ly / 2 + self.dy / 2, self.Ly / 2 - self.dy / 2, self.Ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(Y, X)
        return x, y, X, Y, R, Phi


@dataclass
class SimConfig:
    """Parameters of the MIT simulation."""
    frequency: float = 1.0e6
    coil_tx: CoilConfig = field(
        default_factory=lambda: CoilConfig(radius=1.5e-2, z_offset=1.0e-2, current=1.0)
    )
    coil_rx: CoilConfig = field(
        default_factory=lambda: CoilConfig(radius=1.0e-2, z_offset=1.0e-2, current=1.0)
    )
    grid: GridConfig = field(default_factory=GridConfig)

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.frequency