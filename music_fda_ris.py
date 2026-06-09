# -*- coding: utf-8 -*-
"""
music_fda_ris.py

Function-style 2D-MUSIC simulator for LoRA echo-to-channel prediction.

This file is adapted from your original MUSIC script, but rewritten as a module
so that LoRa_local.py can import and call:

    MusicConfig
    generate_time_varying_ris_phases
    simulate_fda_ris_echo
    estimate_willie_position_music

Important:
    Put this file in the same folder as LoRa_local.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np


@dataclass
class MusicConfig:
    # ---------------------------------------------------------------------
    # 1. System parameters
    # ---------------------------------------------------------------------
    c: float = 3e8
    f0: float = 10e9
    df: float = 1e6

    # BS transmit antennas, BS receive antennas, RIS elements
    N_tx: int = 6
    N_rx: int = 6
    L_ris: int = 12

    # RIS time-varying modulation slots and snapshots
    T_mod: int = 15
    L_snap: int = 200

    # SNR in dB
    SNR_dB: float = 10.0

    # Number of targets. Your current LoRA script uses one Willie target.
    K: int = 1

    # ---------------------------------------------------------------------
    # 2. Geometry
    # ---------------------------------------------------------------------
    r_BR: float = 500.0
    theta_BR: float = np.deg2rad(30.0)

    # Search ranges used by MUSIC if not externally provided
    theta_min_deg: float = -40.0
    theta_max_deg: float = 40.0
    theta_grid_num: int = 81

    range_min: float = 10.0
    range_max: float = 50.0
    range_grid_num: int = 81

    # Random default target, only used in demo
    r_RT_true: float = 30.0
    theta_RT_true: float = np.deg2rad(25.0)

    @property
    def lambda_0(self) -> float:
        return self.c / self.f0

    @property
    def d_bs(self) -> float:
        return self.lambda_0 / 2.0

    @property
    def d_ris(self) -> float:
        return self.lambda_0 / 2.0

    @property
    def echo_dim(self) -> int:
        return self.T_mod * self.N_rx * self.N_tx


def generate_time_varying_ris_phases(
    L_ris: int,
    T_mod: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate time-varying RIS phase matrix.

    Returns
    -------
    Phi : complex ndarray, shape (L_ris, T_mod)
        Each column is the RIS phase vector at one modulation slot.
    """
    rng = np.random.default_rng(seed)
    return np.exp(1j * 2.0 * np.pi * rng.random((L_ris, T_mod)))


def _receive_steering_bs_to_ris(cfg: MusicConfig) -> np.ndarray:
    """
    Receive steering vector a_r, shape (N_rx, 1).
    """
    n_idx = np.arange(cfg.N_rx).reshape(-1, 1)
    a_r = np.exp(
        1j
        * 2.0
        * np.pi
        * cfg.d_bs
        / cfg.lambda_0
        * n_idx
        * np.sin(cfg.theta_BR)
    )
    return a_r


def _transmit_steering_fda(cfg: MusicConfig, target_range: float) -> np.ndarray:
    """
    FDA transmit steering vector a_t(r), shape (N_tx, 1).

    This follows your original code:
        exp(-j 4pi df/c * m * (r_BR + r_RT))
        *
        exp( j 2pi d/lambda * m * sin(theta_BR))
    """
    m_idx = np.arange(cfg.N_tx).reshape(-1, 1)
    a_t = np.exp(
        -1j * 4.0 * np.pi * cfg.df / cfg.c * m_idx * (cfg.r_BR + float(target_range))
    ) * np.exp(
        1j
        * 2.0
        * np.pi
        * cfg.d_bs
        / cfg.lambda_0
        * m_idx
        * np.sin(cfg.theta_BR)
    )
    return a_t


def _ris_time_modulation_vector(
    cfg: MusicConfig,
    Phi: np.ndarray,
    target_angle: float,
) -> np.ndarray:
    """
    RIS time-varying modulation vector g(theta), shape (T_mod, 1).

    This preserves your original term:
        g[t] = (sum_l phi_l,t * exp(-j 2pi d_ris/lambda * l *
                 (sin(theta_BR)+sin(theta_RT))))^2
    """
    Phi = np.asarray(Phi, dtype=np.complex128)
    if Phi.shape != (cfg.L_ris, cfg.T_mod):
        raise ValueError(
            f"Phi must have shape ({cfg.L_ris}, {cfg.T_mod}), got {Phi.shape}."
        )

    l_idx = np.arange(cfg.L_ris).reshape(-1, 1)
    g = np.zeros((cfg.T_mod, 1), dtype=np.complex128)

    for t in range(cfg.T_mod):
        phi_vec = Phi[:, t].reshape(-1, 1)
        phase = np.exp(
            -1j
            * 2.0
            * np.pi
            * cfg.d_ris
            / cfg.lambda_0
            * l_idx
            * (np.sin(cfg.theta_BR) + np.sin(float(target_angle)))
        )
        term = np.sum(phi_vec * phase)
        g[t, 0] = term ** 2

    return g


def build_joint_steering_vector(
    cfg: MusicConfig,
    Phi: np.ndarray,
    target_range: float,
    target_angle: float,
) -> np.ndarray:
    """
    Build joint FDA-RIS-MUSIC steering vector.

    Returns
    -------
    v : complex ndarray, shape (T_mod*N_rx*N_tx, 1)
    """
    a_r = _receive_steering_bs_to_ris(cfg)
    a_t = _transmit_steering_fda(cfg, target_range)
    g = _ris_time_modulation_vector(cfg, Phi, target_angle)
    v = np.kron(g, np.kron(a_r, a_t))
    return v.astype(np.complex128)


def simulate_fda_ris_echo(
    cfg: MusicConfig,
    Phi: np.ndarray,
    true_range: float,
    true_angle: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate one-slot FDA + RIS echo matrix.

    Parameters
    ----------
    cfg : MusicConfig
    Phi : complex ndarray, shape (L_ris, T_mod)
    true_range : float
        Willie range relative to RIS.
    true_angle : float
        Willie angle relative to RIS, in radians.
    seed : int or None

    Returns
    -------
    X : complex ndarray, shape (echo_dim, L_snap)
        Echo matrix used by LoRa_local.py.
    """
    rng = np.random.default_rng(seed)

    v_true = build_joint_steering_vector(cfg, Phi, true_range, true_angle)

    # Complex reflection/source signal, shape (K, L_snap).
    # Current model is for one Willie target, so K=1 is used.
    S = (
        rng.standard_normal((cfg.K, cfg.L_snap))
        + 1j * rng.standard_normal((cfg.K, cfg.L_snap))
    ) / np.sqrt(2.0)

    if cfg.K != 1:
        raise NotImplementedError(
            "This simulator currently supports K=1 target for LoRA training."
        )

    X_clean = v_true @ S

    noise = (
        rng.standard_normal((cfg.echo_dim, cfg.L_snap))
        + 1j * rng.standard_normal((cfg.echo_dim, cfg.L_snap))
    ) / np.sqrt(2.0)

    Ps = np.linalg.norm(X_clean, "fro") ** 2 / (cfg.echo_dim * cfg.L_snap)
    Pn = Ps / (10.0 ** (cfg.SNR_dB / 10.0))
    X = X_clean + np.sqrt(Pn) * noise
    return X.astype(np.complex128)


def estimate_willie_position_music(
    X_echo: np.ndarray,
    cfg: MusicConfig,
    Phi: np.ndarray,
    theta_grid: Optional[np.ndarray] = None,
    range_grid: Optional[np.ndarray] = None,
    return_spectrum: bool = False,
) -> Dict[str, Any]:
    """
    Estimate Willie range and angle using 2D-MUSIC.

    Parameters
    ----------
    X_echo : complex ndarray, shape (echo_dim, L_snap)
    cfg : MusicConfig
    Phi : complex ndarray, shape (L_ris, T_mod)
    theta_grid : ndarray or None
        Angle grid in radians.
    range_grid : ndarray or None
        Range grid in meters.
    return_spectrum : bool

    Returns
    -------
    result : dict
        {
            "range": estimated range,
            "angle": estimated angle in radians,
            optionally "spectrum", "theta_grid", "range_grid"
        }
    """
    X_echo = np.asarray(X_echo, dtype=np.complex128)
    if X_echo.shape[0] != cfg.echo_dim:
        raise ValueError(
            f"X_echo first dimension must be echo_dim={cfg.echo_dim}, got {X_echo.shape[0]}."
        )

    if theta_grid is None:
        theta_grid = (
            np.linspace(cfg.theta_min_deg, cfg.theta_max_deg, cfg.theta_grid_num)
            * np.pi
            / 180.0
        )
    if range_grid is None:
        range_grid = np.linspace(cfg.range_min, cfg.range_max, cfg.range_grid_num)

    theta_grid = np.asarray(theta_grid, dtype=float)
    range_grid = np.asarray(range_grid, dtype=float)

    # Sample covariance
    R_hat = (X_echo @ X_echo.conj().T) / X_echo.shape[1]

    # Eigen-decomposition for Hermitian covariance matrix
    eigvals, eigvecs = np.linalg.eigh(R_hat)
    sort_idx = np.argsort(eigvals)[::-1]
    eigvecs_sorted = eigvecs[:, sort_idx]

    # Noise subspace
    En = eigvecs_sorted[:, cfg.K:]
    En_EnH = En @ En.conj().T

    P_music = np.zeros((len(theta_grid), len(range_grid)), dtype=float)
    eps = 1e-18

    for i, th_test in enumerate(theta_grid):
        for k, r_test in enumerate(range_grid):
            v_test = build_joint_steering_vector(cfg, Phi, r_test, th_test)
            proj = np.abs(v_test.conj().T @ En_EnH @ v_test)[0, 0]
            P_music[i, k] = 1.0 / max(float(proj), eps)

    max_idx = np.unravel_index(np.argmax(P_music), P_music.shape)
    est_angle = float(theta_grid[max_idx[0]])
    est_range = float(range_grid[max_idx[1]])

    result: Dict[str, Any] = {
        "range": est_range,
        "angle": est_angle,
    }

    if return_spectrum:
        result.update(
            {
                "spectrum": P_music,
                "theta_grid": theta_grid,
                "range_grid": range_grid,
            }
        )

    return result


def demo_plot_music_spectrum() -> None:
    """
    Optional demo. Run this file directly if you want to reproduce the MUSIC plot.

    This function is not used by LoRa_local.py.
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Songti SC",
        "STHeiti",
        "Microsoft YaHei",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    cfg = MusicConfig()
    Phi = generate_time_varying_ris_phases(cfg.L_ris, cfg.T_mod, seed=42)
    X = simulate_fda_ris_echo(
        cfg,
        Phi,
        true_range=cfg.r_RT_true,
        true_angle=cfg.theta_RT_true,
        seed=42,
    )

    print("开始 2D-MUSIC 谱峰搜索，这可能需要几秒钟，请稍候...")
    est = estimate_willie_position_music(
        X,
        cfg,
        Phi,
        return_spectrum=True,
    )
    print("搜索完成！")
    print(
        f"Estimated range = {est['range']:.2f} m, "
        f"angle = {np.rad2deg(est['angle']):.2f} deg"
    )

    P_music = est["spectrum"]
    theta_search = est["theta_grid"]
    r_search = est["range_grid"]

    P_music_dB = 10.0 * np.log10(P_music / np.max(P_music))
    R_grid, Theta_grid = np.meshgrid(r_search, theta_search * 180.0 / np.pi)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        R_grid,
        Theta_grid,
        P_music_dB,
        cmap="jet",
        edgecolor="none",
        alpha=0.9,
    )

    ax.set_xlabel("距离 Range (m)", fontweight="bold", labelpad=10)
    ax.set_ylabel("角度 Angle (Degree)", fontweight="bold", labelpad=10)
    ax.set_zlabel("归一化空间谱 P(θ, r) [dB]", fontweight="bold", labelpad=10)
    ax.set_title("级联信道 2D-MUSIC 空间谱", fontweight="bold", fontsize=14)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    ax.view_init(elev=35, azim=-45)

    ax.scatter(
        cfg.r_RT_true,
        cfg.theta_RT_true * 180.0 / np.pi,
        0,
        color="red",
        s=100,
        marker="*",
        label="真实目标位置",
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_plot_music_spectrum()
