import numpy as np
import matplotlib.pyplot as plt


def plot_field_2d(X, Y, data, title="", label="", cmap="viridis", ax=None):
    """Generic 2D field plot."""
    standalone = ax is None

    if standalone:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    im = ax.pcolormesh(X, Y, data, shading="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.tick_params(direction="in")
    fig.colorbar(im, ax=ax, label=label)

    if standalone:
        plt.tight_layout()
        plt.show()

    return im


def plot_correction_summary(X, Y, sigma, E_total, rho_eps0, delta_E):
    """2x2 panel with conductivity, total field, charge density, and correction."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 7))

    plot_field_2d(
        X, Y, sigma,
        title=r"Conductivity $|\sigma|$",
        label="S/m",
        cmap="plasma",
        ax=ax1,
    )
    plot_field_2d(
        X, Y, np.sqrt(np.abs(E_total[0])**2 + np.abs(E_total[1])**2),
        title=r"Total Electric Field $|E|$",
        label="V/m",
        cmap="viridis",
        ax=ax2,
    )
    plot_field_2d(
        X, Y, np.imag(rho_eps0),
        title=r"Charge Density $\rho/\epsilon_0$",
        label=r"V/m$^2$",
        cmap="inferno",
        ax=ax3,
    )
    plot_field_2d(
        X, Y, np.sqrt(np.abs(delta_E[0])**2 + np.abs(delta_E[1])**2),
        title="Partial Correction",
        label="V/m",
        cmap="cividis",
        ax=ax4,
    )

    plt.tight_layout()
    plt.show()


def plot_bz_scan(X, Y, flux_healthy, flux_anom, r_rx: float):
    """1x3 panel: Bz healthy, Bz anomalous, delta Bz."""
    area = np.pi * r_rx**2
    Bz_h = flux_healthy / area
    Bz_a = flux_anom / area
    Bz_d = Bz_a - Bz_h

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, data, title in zip(
        axes,
        [np.abs(Bz_h), np.abs(Bz_a), np.abs(Bz_d)],
        [r"$B_z$ (Healthy) [T]", r"$B_z$ (Anomalous) [T]", r"$\Delta B_z$ [T]"],
    ):
        plot_field_2d(X, Y, data, title=title, cmap="coolwarm", ax=ax)

    plt.tight_layout()
    plt.show()

    return Bz_h, Bz_a, Bz_d