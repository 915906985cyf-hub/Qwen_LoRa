from __future__ import annotations

"""
lora_Qwen_geo.py

Direct Willie trajectory / geometric-state prediction with Qwen + PEFT-LoRA.
Output per future slot is [log(range), sin(angle), cos(angle)] instead of full CSI.
This makes the reported NMSE comparable to range/angle prediction settings.

Put in the same folder with:
    lora_Qwen_geo.py
    music_fda_ris.py

Main modes:
    train       Train direct geometry predictor.
    infer_demo  Load checkpoint and show one demo prediction.
    eval_saved  Load checkpoint and test unseen trajectories with persistence baseline.
"""

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoConfig, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
except Exception:
    AutoConfig = None
    AutoModelForCausalLM = None
    LoraConfig = None
    get_peft_model = None


# =============================================================================
# Config
# =============================================================================
@dataclass
class EchoGeoConfig:
    input_len: int = 5
    pred_len: int = 5
    echo_dim: int = 4 * 6 * 6
    L_snap: int = 50

    music_T_mod: int = 4
    music_L_snap: int = 50
    music_L_ris: int = 12
    music_SNR_dB: float = 10.0

    hf_model_name: str = "Qwen/Qwen2.5-0.5B"
    hf_dtype: str = "float16"
    hf_target_modules: str = "q_proj,v_proj"
    hf_trust_remote_code: bool = True
    hf_freeze_base: bool = True
    hf_cache_dir: Optional[str] = None

    batch_size: int = 8
    grad_accum_steps: int = 2
    epochs: int = 80
    lr: float = 1e-5
    weight_decay: float = 3e-4
    grad_clip: float = 0.5
    val_ratio: float = 0.2
    seed: int = 42
    norm_eps: float = 1e-8
    input_noise_std: float = 0.003
    early_stop_patience: int = 15
    early_stop_min_delta: float = 1e-5
    split_by_order: bool = True

    dropout: float = 0.10
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05

    loss_log_range_weight: float = 1.0
    loss_angle_weight: float = 2.0
    loss_temporal_weight: float = 0.20
    loss_acceleration_weight: float = 0.05
    loss_unit_weight: float = 0.05
    horizon_weights: str = "0.5,0.8,1.0,1.2,1.5"

    output_dir: str = "runs/qwen_geo_direct_v1"
    val_pred_samples: int = 512
    save_val_predictions: bool = False


# =============================================================================
# Helpers
# =============================================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def complex_to_ri(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.stack([np.real(x), np.imag(x)], axis=-1).astype(np.float32)


def wrap_angle_rad(theta):
    return (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi


def pos_to_geo(pos_seq: np.ndarray) -> np.ndarray:
    pos_seq = np.asarray(pos_seq, dtype=np.float32)
    r = np.maximum(pos_seq[..., 0], 1e-9)
    th = pos_seq[..., 1]
    return np.stack([np.log(r), np.sin(th), np.cos(th)], axis=-1).astype(np.float32)


def geo_to_range_angle(geo_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    geo_seq = np.asarray(geo_seq, dtype=np.float32)
    r = np.exp(geo_seq[..., 0])
    th = np.arctan2(geo_seq[..., 1], geo_seq[..., 2])
    return r.astype(np.float32), th.astype(np.float32)


class StandardScaler:
    def __init__(self, eps: float = 1e-8):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = float(eps)

    def fit(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float32)
        axes = tuple(range(arr.ndim - 1))
        self.mean = arr.mean(axis=axes, keepdims=True).astype(np.float32)
        self.std = (arr.std(axis=axes, keepdims=True) + self.eps).astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (x * self.std + self.mean).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.std = np.asarray(state["std"], dtype=np.float32)
        self.eps = float(state.get("eps", 1e-8))


def torch_load_robust(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_jsonable)


def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def nmse_db(x: float) -> float:
    return float(10.0 * np.log10(max(float(x), 1e-12)))


# =============================================================================
# Data generation
# =============================================================================
def _try_import_music_module():
    try:
        import music_fda_ris as music_mod
        return music_mod
    except Exception as err:
        raise ImportError("Cannot import music_fda_ris.py. Put it in the same folder.") from err


def _apply_music_overrides(music_cfg, cfg: EchoGeoConfig):
    if hasattr(music_cfg, "T_mod"):
        music_cfg.T_mod = int(cfg.music_T_mod)
    if hasattr(music_cfg, "L_snap"):
        music_cfg.L_snap = int(cfg.music_L_snap)
    if hasattr(music_cfg, "L_ris"):
        music_cfg.L_ris = int(cfg.music_L_ris)
    if hasattr(music_cfg, "SNR_dB"):
        music_cfg.SNR_dB = float(cfg.music_SNR_dB)
    return music_cfg


def generate_sin_trajectory(traj_len: int, seed: int = 1) -> np.ndarray:
    """Same smooth Willie trajectory distribution as previous scripts."""
    rng = np.random.default_rng(seed)
    t = np.arange(traj_len, dtype=np.float32)
    r = 30.0 + rng.uniform(3.0, 7.0) * np.sin(
        2 * np.pi * rng.uniform(0.04, 0.12) * t + rng.uniform(0, 2 * np.pi)
    )
    th = np.deg2rad(20.0) + np.deg2rad(rng.uniform(5.0, 12.0)) * np.cos(
        2 * np.pi * rng.uniform(0.04, 0.12) * t + rng.uniform(0, 2 * np.pi)
    )
    return np.stack([r, th], axis=-1).astype(np.float32)


# Keep old name for compatibility with previous test scripts.
generate_true_trajectory = generate_sin_trajectory


def generate_linear_trajectory(traj_len: int, seed: int = 1) -> np.ndarray:
    """Paper-like constant-velocity trajectory, converted to polar coordinates."""
    rng = np.random.default_rng(seed)
    p0 = np.array([rng.uniform(20.0, 35.0), rng.uniform(5.0, 20.0)], dtype=np.float32)
    speed = rng.uniform(0.4, 1.5)  # m/slot
    direction = rng.uniform(-np.pi, np.pi)
    v = speed * np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)
    xy = p0[None, :] + np.arange(traj_len, dtype=np.float32)[:, None] * v[None, :]
    r = np.linalg.norm(xy, axis=1)
    th = np.arctan2(xy[:, 1], xy[:, 0])
    return np.stack([r, th], axis=-1).astype(np.float32)


def generate_trajectory(traj_len: int, seed: int = 1, traj_type: str = "sin") -> np.ndarray:
    traj_type = traj_type.lower()
    if traj_type == "sin":
        return generate_sin_trajectory(traj_len, seed)
    if traj_type == "linear":
        return generate_linear_trajectory(traj_len, seed)
    if traj_type == "mixed":
        rng = np.random.default_rng(seed)
        return generate_sin_trajectory(traj_len, seed) if rng.random() < 0.5 else generate_linear_trajectory(traj_len, seed)
    raise ValueError("traj_type must be sin, linear, or mixed.")


def generate_echo_geo_trajectory_by_music(
    traj_len: int,
    seed: int,
    cfg: EchoGeoConfig,
    label_mode: str = "true",
    music_grid_small: bool = False,
    traj_type: str = "sin",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    music_mod = _try_import_music_module()
    cfg_m = _apply_music_overrides(music_mod.MusicConfig(), cfg)

    if music_grid_small:
        theta_grid = np.linspace(-40.0, 40.0, 21) * np.pi / 180.0
        range_grid = np.linspace(10.0, 50.0, 21)
    else:
        theta_grid = None
        range_grid = None

    Phi = music_mod.generate_time_varying_ris_phases(cfg_m.L_ris, cfg_m.T_mod, seed=seed)
    true_pos = generate_trajectory(traj_len, seed=seed, traj_type=traj_type)

    echo_list, pos_list = [], []
    for n in range(traj_len):
        true_r = float(true_pos[n, 0])
        true_th = float(true_pos[n, 1])
        X_echo = music_mod.simulate_fda_ris_echo(
            cfg_m,
            Phi,
            true_range=true_r,
            true_angle=true_th,
            seed=seed * 10000 + n,
        )
        if label_mode.lower() == "music":
            est = music_mod.estimate_willie_position_music(
                X_echo,
                cfg_m,
                Phi,
                theta_grid=theta_grid,
                range_grid=range_grid,
                return_spectrum=False,
            )
            label_r = float(est["range"])
            label_th = float(est["angle"])
        elif label_mode.lower() == "true":
            label_r, label_th = true_r, true_th
        else:
            raise ValueError("label_mode must be true or music.")
        echo_list.append(X_echo)
        pos_list.append([label_r, label_th])

    echo_seq = np.asarray(echo_list)
    pos_seq = np.asarray(pos_list, dtype=np.float32)
    geo_seq = pos_to_geo(pos_seq)
    return echo_seq, geo_seq, pos_seq


def build_echo_geo_windows(
    echo_seq: np.ndarray,
    geo_seq: np.ndarray,
    pos_seq: np.ndarray,
    input_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = echo_seq.shape[0]
    window = input_len + pred_len
    if T < window:
        raise ValueError(f"T={T} is too short for input_len+pred_len={window}.")
    X, Y, P, starts = [], [], [], []
    for st in range(T - window + 1):
        X.append(complex_to_ri(echo_seq[st:st + input_len]))
        Y.append(geo_seq[st + input_len:st + window])
        P.append(pos_seq[st + input_len:st + window])
        starts.append(st)
    return (
        np.stack(X, axis=0).astype(np.float32),
        np.stack(Y, axis=0).astype(np.float32),
        np.stack(P, axis=0).astype(np.float32),
        np.asarray(starts, dtype=np.int32),
    )


def generate_training_dataset_by_music(
    num_traj: int,
    traj_len: int,
    cfg: EchoGeoConfig,
    label_mode: str = "true",
    music_grid_small: bool = False,
    traj_type: str = "sin",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_all, Y_all, P_all, S_all = [], [], [], []
    for i in range(num_traj):
        echo_seq, geo_seq, pos_seq = generate_echo_geo_trajectory_by_music(
            traj_len=traj_len,
            seed=cfg.seed + i,
            cfg=cfg,
            label_mode=label_mode,
            music_grid_small=music_grid_small,
            traj_type=traj_type,
        )
        Xi, Yi, Pi, Si = build_echo_geo_windows(echo_seq, geo_seq, pos_seq, cfg.input_len, cfg.pred_len)
        X_all.append(Xi)
        Y_all.append(Yi)
        P_all.append(Pi)
        S_all.append(Si)
        if i == 0 or (i + 1) % 10 == 0:
            print(f"Generated trajectory {i+1}/{num_traj}: X={Xi.shape}, Y_geo={Yi.shape}")
    return (
        np.concatenate(X_all, axis=0),
        np.concatenate(Y_all, axis=0),
        np.concatenate(P_all, axis=0),
        np.concatenate(S_all, axis=0),
    )


class EchoGeoDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


# =============================================================================
# Model
# =============================================================================
def _require_hf_packages() -> None:
    if AutoConfig is None or AutoModelForCausalLM is None or LoraConfig is None or get_peft_model is None:
        raise ImportError("Install HF packages: pip install transformers peft accelerate safetensors")


def _torch_dtype_from_name(name: str):
    name = str(name).lower()
    if name in ["fp16", "float16", "half"]:
        return torch.float16
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if name in ["fp32", "float32", "full"]:
        return torch.float32
    raise ValueError("hf_dtype must be float16, bfloat16, or float32.")


class EchoConvEmbed(nn.Module):
    def __init__(self, cfg: EchoGeoConfig, hidden_size: int):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = int(hidden_size)
        self.net = nn.Sequential(
            nn.Conv2d(2, 12, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout2d(min(cfg.dropout, 0.25)),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout2d(min(cfg.dropout, 0.20)),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(24 * 8 * 8, min(256, self.hidden_size)),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(min(256, self.hidden_size), self.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, E, S, RI = x.shape
        if RI != 2:
            raise ValueError("Last dimension must be 2 for [real, imag].")
        z = x.permute(0, 1, 4, 2, 3).contiguous().view(B * T, 2, E, S)
        z = self.net(z)
        return z.view(B, T, self.hidden_size)


class EchoToGeoHFLoRA(nn.Module):
    def __init__(self, cfg: EchoGeoConfig):
        super().__init__()
        _require_hf_packages()
        self.cfg = cfg
        hf_config = AutoConfig.from_pretrained(
            cfg.hf_model_name,
            trust_remote_code=cfg.hf_trust_remote_code,
            cache_dir=cfg.hf_cache_dir,
        )
        hidden_size = int(getattr(hf_config, "hidden_size"))
        dtype = _torch_dtype_from_name(cfg.hf_dtype)

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_model_name,
            torch_dtype=dtype,
            trust_remote_code=cfg.hf_trust_remote_code,
            cache_dir=cfg.hf_cache_dir,
            low_cpu_mem_usage=True,
        )
        base_model.config.output_hidden_states = True
        base_model.config.use_cache = False
        if cfg.hf_freeze_base:
            for p in base_model.parameters():
                p.requires_grad = False

        target_modules = [x.strip() for x in cfg.hf_target_modules.split(",") if x.strip()]
        peft_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(base_model, peft_config)
        self.echo_embed = EchoConvEmbed(cfg, hidden_size=hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.input_len, hidden_size))
        self.drop = nn.Dropout(cfg.dropout)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, min(512, hidden_size)),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(min(512, hidden_size), cfg.pred_len * 3),
        )
        self._init_io_weights()

    def _init_io_weights(self) -> None:
        for module in list(self.echo_embed.modules()) + list(self.regression_head.modules()):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[0], x.shape[1]
        if T != self.cfg.input_len:
            raise ValueError(f"Expected input_len={self.cfg.input_len}, got {T}.")
        h = self.echo_embed(x) + self.pos_embed[:, :T, :]
        h = self.drop(h)
        llm_dtype = next(self.llm.parameters()).dtype
        h = h.to(dtype=llm_dtype)
        attention_mask = torch.ones(B, T, dtype=torch.long, device=x.device)
        out = self.llm(
            inputs_embeds=h,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]
        last = hidden[:, -1, :].to(dtype=self.regression_head[1].weight.dtype)
        pred = self.regression_head(last)
        return pred.view(B, self.cfg.pred_len, 3)


def freeze_backbone_except_lora_and_io(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = (
            "lora_" in name
            or "echo_embed" in name
            or "pos_embed" in name
            or "regression_head" in name
        )


def trainable_state_dict(model: nn.Module) -> dict:
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# Loss / metrics
# =============================================================================
def parse_horizon_weights(cfg: EchoGeoConfig, device: torch.device) -> torch.Tensor:
    if not str(cfg.horizon_weights).strip():
        w = torch.ones(cfg.pred_len, dtype=torch.float32, device=device)
    else:
        vals = [float(x.strip()) for x in str(cfg.horizon_weights).split(",") if x.strip()]
        if len(vals) != cfg.pred_len:
            print(f"Warning: horizon_weights length {len(vals)} != pred_len {cfg.pred_len}; using uniform weights.")
            w = torch.ones(cfg.pred_len, dtype=torch.float32, device=device)
        else:
            w = torch.as_tensor(vals, dtype=torch.float32, device=device)
    return w / w.mean().clamp_min(1e-8)


def inverse_standardize_tensor(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def weighted_mean(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    view_shape = [1, -1] + [1] * (x.ndim - 2)
    return torch.mean(x * weights.view(*view_shape))


def combined_geo_loss(pred_n, target_n, y_mean_t, y_std_t, cfg: EchoGeoConfig, horizon_w) -> torch.Tensor:
    pred = inverse_standardize_tensor(pred_n, y_mean_t, y_std_t)
    target = inverse_standardize_tensor(target_n, y_mean_t, y_std_t)

    l_log_range = weighted_mean((pred[:, :, 0] - target[:, :, 0]) ** 2, horizon_w)
    l_angle = weighted_mean(torch.sum((pred[:, :, 1:3] - target[:, :, 1:3]) ** 2, dim=-1), horizon_w)

    if cfg.pred_len > 1:
        pd = pred[:, 1:, :] - pred[:, :-1, :]
        td = target[:, 1:, :] - target[:, :-1, :]
        l_temporal = weighted_mean(torch.sum((pd - td) ** 2, dim=-1), horizon_w[1:])
    else:
        l_temporal = torch.zeros((), dtype=pred.dtype, device=pred.device)

    if cfg.pred_len > 2:
        pa = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
        ta = target[:, 2:, :] - 2.0 * target[:, 1:-1, :] + target[:, :-2, :]
        l_acc = weighted_mean(torch.sum((pa - ta) ** 2, dim=-1), horizon_w[2:])
    else:
        l_acc = torch.zeros((), dtype=pred.dtype, device=pred.device)

    sc_norm = torch.sum(pred[:, :, 1:3] ** 2, dim=-1)
    l_unit = weighted_mean((sc_norm - 1.0) ** 2, horizon_w)

    return (
        cfg.loss_log_range_weight * l_log_range
        + cfg.loss_angle_weight * l_angle
        + cfg.loss_temporal_weight * l_temporal
        + cfg.loss_acceleration_weight * l_acc
        + cfg.loss_unit_weight * l_unit
    )


def geo_nmse_normalized_np(pred_n: np.ndarray, true_n: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sum((pred_n - true_n) ** 2) / (np.sum(true_n ** 2) + eps))


def range_nmse_np(pred_r: np.ndarray, true_r: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sum((pred_r - true_r) ** 2) / (np.sum(true_r ** 2) + eps))


def angle_sincos_nmse_np(pred_th: np.ndarray, true_th: np.ndarray, eps: float = 1e-12) -> float:
    num = np.sum((np.sin(pred_th) - np.sin(true_th)) ** 2 + (np.cos(pred_th) - np.cos(true_th)) ** 2)
    den = np.sum(np.sin(true_th) ** 2 + np.cos(true_th) ** 2) + eps
    return float(num / den)


def summarize_geometry(pred_geo: np.ndarray, true_geo: np.ndarray, pred_n=None, true_n=None) -> Dict[str, float]:
    pred_r, pred_th = geo_to_range_angle(pred_geo)
    true_r, true_th = geo_to_range_angle(true_geo)
    range_err = np.abs(pred_r - true_r)
    angle_err = np.abs(wrap_angle_rad(pred_th - true_th)) * 180.0 / np.pi
    r_nmse = range_nmse_np(pred_r, true_r)
    a_nmse = angle_sincos_nmse_np(pred_th, true_th)
    out = {
        "range_mae_m": float(np.mean(range_err)),
        "range_median_abs_err_m": float(np.median(range_err)),
        "range_rmse_m": float(np.sqrt(np.mean(range_err ** 2))),
        "range_nmse": r_nmse,
        "range_nmse_db": nmse_db(r_nmse),
        "angle_mae_deg": float(np.mean(angle_err)),
        "angle_median_abs_err_deg": float(np.median(angle_err)),
        "angle_rmse_deg": float(np.sqrt(np.mean(angle_err ** 2))),
        "angle_sincos_nmse": a_nmse,
        "angle_sincos_nmse_db": nmse_db(a_nmse),
    }
    if pred_n is not None and true_n is not None:
        g_nmse = geo_nmse_normalized_np(pred_n, true_n)
        out["geo_nmse_normalized"] = g_nmse
        out["geo_nmse_normalized_db"] = nmse_db(g_nmse)
    return out


@torch.no_grad()
def evaluate_loss(model, loader, device, y_mean_t, y_std_t, cfg, horizon_w) -> float:
    model.eval()
    vals = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        vals.append(float(combined_geo_loss(model(xb), yb, y_mean_t, y_std_t, cfg, horizon_w).item()))
    return float(np.mean(vals)) if vals else float("nan")


def get_environment_info(device: torch.device) -> Dict[str, Any]:
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": os.sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "selected_device": str(device),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception:
        info["transformers_version"] = None
    try:
        import peft
        info["peft_version"] = peft.__version__
    except Exception:
        info["peft_version"] = None
    return info


# =============================================================================
# Training / loading / prediction
# =============================================================================
def train_echo_to_geo_lora(X, Y, cfg: EchoGeoConfig, save_path="best_model_geo.pt", device=None, output_dir=None):
    set_seed(cfg.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cpu" and cfg.hf_dtype != "float32":
        print(f"CPU detected: switching hf_dtype from {cfg.hf_dtype} to float32.")
        cfg.hf_dtype = "float32"

    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path_obj = Path(save_path)
    if not save_path_obj.is_absolute() and str(save_path_obj.parent) == ".":
        save_path_obj = out_dir / save_path_obj.name
    save_path = str(save_path_obj)

    train_log_csv = out_dir / "train_log.csv"
    summary_json = out_dir / "summary.json"
    val_pred_csv = out_dir / "val_predictions.csv"
    val_metrics_csv = out_dir / "val_metrics.csv"

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    cfg.echo_dim = int(X.shape[2])
    cfg.L_snap = int(X.shape[3])
    cfg.output_dir = str(out_dir)

    n = X.shape[0]
    n_val = max(1, int(n * cfg.val_ratio))
    if cfg.split_by_order:
        train_idx = np.arange(0, n - n_val)
        val_idx = np.arange(n - n_val, n)
    else:
        idx = np.arange(n)
        np.random.shuffle(idx)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_scaler = StandardScaler(cfg.norm_eps)
    y_scaler = StandardScaler(cfg.norm_eps)
    x_scaler.fit(X[train_idx])
    y_scaler.fit(Y[train_idx])
    X_train = x_scaler.transform(X[train_idx])
    Y_train = y_scaler.transform(Y[train_idx])
    X_val = x_scaler.transform(X[val_idx])
    Y_val = y_scaler.transform(Y[val_idx])

    y_mean_t = torch.as_tensor(y_scaler.mean, dtype=torch.float32, device=dev)
    y_std_t = torch.as_tensor(y_scaler.std, dtype=torch.float32, device=dev)
    horizon_w = parse_horizon_weights(cfg, dev)

    train_loader = DataLoader(EchoGeoDataset(X_train, Y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(EchoGeoDataset(X_val, Y_val), batch_size=cfg.batch_size, shuffle=False)

    model = EchoToGeoHFLoRA(cfg)
    freeze_backbone_except_lora_and_io(model)
    model.to(dev)
    total, trainable = count_params(model)
    env_info = get_environment_info(dev)

    print(f"Device: {dev}")
    print(f"Output dir: {out_dir}")
    print(f"Checkpoint: {save_path}")
    print(f"Dataset: train={len(train_idx)}, val={len(val_idx)}, X={X.shape}, Y_geo={Y.shape}")
    print(f"Horizon weights: {horizon_w.detach().cpu().numpy().round(3).tolist()}")
    print(f"Parameters: total={total:,}, trainable={trainable:,} ({100*trainable/max(total,1):.3f}%)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(2, cfg.early_stop_patience // 2))
    best_val, best_epoch, bad_epochs = float("inf"), -1, 0
    train_start = time.time()
    if train_log_csv.exists():
        train_log_csv.unlink()

    for ep in range(1, cfg.epochs + 1):
        ep_start = time.time()
        model.train()
        losses = []
        opt.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = xb.to(dev), yb.to(dev)
            xb_model = xb + cfg.input_noise_std * torch.randn_like(xb) if cfg.input_noise_std > 0 else xb
            raw_loss = combined_geo_loss(model(xb_model), yb, y_mean_t, y_std_t, cfg, horizon_w)
            loss = raw_loss / max(1, int(cfg.grad_accum_steps))
            loss.backward()
            if (step % max(1, int(cfg.grad_accum_steps)) == 0) or (step == len(train_loader)):
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(raw_loss.item()))

        val = evaluate_loss(model, val_loader, dev, y_mean_t, y_std_t, cfg, horizon_w)
        sched.step(val)
        tr = float(np.mean(losses)) if losses else float("nan")
        current_lr = float(opt.param_groups[0]["lr"])
        improved = val < best_val - cfg.early_stop_min_delta
        if improved:
            best_val, best_epoch, bad_epochs = val, ep, 0
            torch.save({
                "config": asdict(cfg),
                "model_state_dict": trainable_state_dict(model),
                "x_scaler": x_scaler.state_dict(),
                "y_scaler": y_scaler.state_dict(),
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
            }, save_path)
        else:
            bad_epochs += 1

        epoch_seconds = time.time() - ep_start
        peak_cuda_mem_gb = float(torch.cuda.max_memory_allocated(dev) / (1024 ** 3)) if dev.type == "cuda" else None
        append_csv_row(train_log_csv, {
            "epoch": ep,
            "train_loss": tr,
            "val_loss": val,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "lr": current_lr,
            "bad_epochs": bad_epochs,
            "epoch_seconds": epoch_seconds,
            "peak_cuda_mem_gb": peak_cuda_mem_gb,
            "improved": int(improved),
        })
        if ep == 1 or ep % 5 == 0 or improved or ep == cfg.epochs:
            print(f"Epoch {ep:04d}/{cfg.epochs} | train_loss={tr:.6e} | val_loss={val:.6e} | best={best_val:.6e}@{best_epoch} | lr={current_lr:.2e} | bad={bad_epochs} | sec={epoch_seconds:.1f}")
        if cfg.early_stop_patience > 0 and bad_epochs >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {ep}. Best val_loss={best_val:.6e} at epoch {best_epoch}.")
            break

    val_prediction_metrics = None
    if cfg.save_val_predictions and Path(save_path).exists():
        val_prediction_metrics = export_validation_predictions(
            checkpoint_path=save_path,
            X_val_n=X_val,
            Y_val_n=Y_val,
            y_scaler=y_scaler,
            cfg=cfg,
            device=dev,
            out_csv=val_pred_csv,
            metrics_csv=val_metrics_csv,
            max_samples=cfg.val_pred_samples,
            batch_size=cfg.batch_size,
            val_indices=val_idx,
        )

    summary = {
        "save_path": save_path,
        "output_dir": str(out_dir),
        "config_info": {
            "config": asdict(cfg),
            "dataset": {"X_shape": list(X.shape), "Y_geo_shape": list(Y.shape), "train_size": int(len(train_idx)), "val_size": int(len(val_idx))},
            "model": {"total_params": int(total), "trainable_params": int(trainable), "trainable_ratio_percent": float(100 * trainable / max(total, 1))},
            "environment": env_info,
        },
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "total_seconds": time.time() - train_start,
        "train_log_csv": str(train_log_csv),
        "val_predictions_csv": str(val_pred_csv) if cfg.save_val_predictions else None,
        "val_metrics_csv": str(val_metrics_csv) if cfg.save_val_predictions else None,
        "validation_prediction_metrics": val_prediction_metrics,
    }
    save_json(summary_json, summary)
    return summary


def load_trained_geo_lora(checkpoint_path: str, device: Optional[str] = None):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch_load_robust(checkpoint_path, map_location=torch.device("cpu"))
    cfg = EchoGeoConfig(**ckpt["config"])
    if dev.type == "cpu" and cfg.hf_dtype != "float32":
        print(f"CPU detected: switching hf_dtype from {cfg.hf_dtype} to float32.")
        cfg.hf_dtype = "float32"
    model = EchoToGeoHFLoRA(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        print("Unexpected keys:", unexpected)
    if missing:
        print(f"Missing keys count: {len(missing)}; frozen HF base is loaded from hf_model_name.")
    model.to(dev)
    model.eval()
    x_scaler = StandardScaler(cfg.norm_eps)
    y_scaler = StandardScaler(cfg.norm_eps)
    x_scaler.load_state_dict(ckpt["x_scaler"])
    y_scaler.load_state_dict(ckpt["y_scaler"])
    return model, x_scaler, y_scaler, cfg, dev, ckpt


@torch.no_grad()
def predict_future_geometry_from_echo_history(echo_history: np.ndarray, checkpoint_path="best_model_geo.pt", device=None):
    model, x_scaler, y_scaler, cfg, dev, _ = load_trained_geo_lora(checkpoint_path, device=device)
    echo_history = np.asarray(echo_history)
    if echo_history.shape != (cfg.input_len, cfg.echo_dim, cfg.L_snap):
        raise ValueError(f"echo_history must have shape ({cfg.input_len},{cfg.echo_dim},{cfg.L_snap}), got {echo_history.shape}")
    X = complex_to_ri(echo_history)[None, ...]
    Xn = x_scaler.transform(X)
    xb = torch.as_tensor(Xn, dtype=torch.float32, device=dev)
    pred_n = model(xb).detach().cpu().numpy()[0]
    pred_geo = y_scaler.inverse_transform(pred_n)
    pred_r, pred_th = geo_to_range_angle(pred_geo)
    return pred_r, pred_th, pred_geo


@torch.no_grad()
def export_validation_predictions(checkpoint_path, X_val_n, Y_val_n, y_scaler, cfg, device, out_csv, metrics_csv, max_samples=512, batch_size=8, val_indices=None):
    model, _, _, _, dev, _ = load_trained_geo_lora(checkpoint_path, device=str(device))
    n_val = int(X_val_n.shape[0])
    selected = np.arange(n_val) if max_samples is None or max_samples <= 0 or max_samples >= n_val else np.unique(np.linspace(0, n_val - 1, int(max_samples), dtype=int))
    pred_batches = []
    for start in range(0, len(selected), batch_size):
        ids = selected[start:start + batch_size]
        xb = torch.as_tensor(X_val_n[ids], dtype=torch.float32, device=dev)
        pred_batches.append(model(xb).detach().cpu().numpy())
    pred_n = np.concatenate(pred_batches, axis=0)
    true_n = Y_val_n[selected]
    pred_geo = y_scaler.inverse_transform(pred_n)
    true_geo = y_scaler.inverse_transform(true_n)
    pred_r, pred_th = geo_to_range_angle(pred_geo)
    true_r, true_th = geo_to_range_angle(true_geo)

    rows = []
    for local_i, original_val_pos in enumerate(selected):
        sample_index = int(val_indices[original_val_pos]) if val_indices is not None else int(original_val_pos)
        for h in range(cfg.pred_len):
            rows.append({
                "sample_index": sample_index,
                "val_position": int(original_val_pos),
                "horizon": int(h + 1),
                "true_range_m": float(true_r[local_i, h]),
                "pred_range_m": float(pred_r[local_i, h]),
                "range_abs_err_m": float(abs(pred_r[local_i, h] - true_r[local_i, h])),
                "true_angle_deg": float(true_th[local_i, h] * 180.0 / np.pi),
                "pred_angle_deg": float(pred_th[local_i, h] * 180.0 / np.pi),
                "angle_abs_err_deg": float(abs(wrap_angle_rad(pred_th[local_i, h] - true_th[local_i, h])) * 180.0 / np.pi),
            })
    metrics = summarize_geometry(pred_geo, true_geo, pred_n=pred_n, true_n=true_n)
    metrics["num_validation_samples_total"] = n_val
    metrics["num_exported_samples"] = int(len(selected))
    metrics["num_exported_rows"] = int(len(rows))
    write_csv_rows(out_csv, rows)
    write_csv_rows(metrics_csv, [metrics])
    return metrics


# =============================================================================
# eval_saved
# =============================================================================
def summarize_eval_rows(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    out_rows = []

    def add_summary(name, d):
        pred_geo = d[["pred_log_range", "pred_sin_angle", "pred_cos_angle"]].to_numpy(dtype=np.float32).reshape(-1, 1, 3)
        true_geo = d[["true_log_range", "true_sin_angle", "true_cos_angle"]].to_numpy(dtype=np.float32).reshape(-1, 1, 3)
        base_geo = d[["baseline_log_range", "baseline_sin_angle", "baseline_cos_angle"]].to_numpy(dtype=np.float32).reshape(-1, 1, 3)
        pm = summarize_geometry(pred_geo, true_geo)
        bm = summarize_geometry(base_geo, true_geo)
        row = {"group": name, "num_rows": int(len(d))}
        for k, v in pm.items():
            row[f"model_{k}"] = v
        for k, v in bm.items():
            row[f"baseline_{k}"] = v
        row["range_mae_improvement_vs_baseline_percent"] = 100.0 * (row["baseline_range_mae_m"] - row["model_range_mae_m"]) / max(row["baseline_range_mae_m"], 1e-12)
        row["angle_mae_improvement_vs_baseline_percent"] = 100.0 * (row["baseline_angle_mae_deg"] - row["model_angle_mae_deg"]) / max(row["baseline_angle_mae_deg"], 1e-12)
        out_rows.append(row)

    add_summary("all", df)
    for h, dh in df.groupby("horizon"):
        add_summary(f"horizon_{int(h)}", dh)
    return out_rows


@torch.no_grad()
def evaluate_saved_checkpoint(checkpoint_path, output_dir="eval_geo_saved", num_test_traj=100, traj_len=32, seed_start=10000, label_mode="true", music_grid_small=False, traj_type="sin", device=None, batch_size=16):
    model, x_scaler, y_scaler, cfg, dev, ckpt = load_trained_geo_lora(checkpoint_path, device=device)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for traj_id in range(num_test_traj):
        seed = seed_start + traj_id
        echo_seq, geo_seq, pos_seq = generate_echo_geo_trajectory_by_music(traj_len, seed, cfg, label_mode, music_grid_small, traj_type)
        X, Y_true, P_true, starts = build_echo_geo_windows(echo_seq, geo_seq, pos_seq, cfg.input_len, cfg.pred_len)
        Xn = x_scaler.transform(X)
        pred_batches = []
        for b0 in range(0, Xn.shape[0], batch_size):
            xb = torch.as_tensor(Xn[b0:b0 + batch_size], dtype=torch.float32, device=dev)
            pred_batches.append(model(xb).detach().cpu().numpy())
        pred_n = np.concatenate(pred_batches, axis=0)
        Y_pred = y_scaler.inverse_transform(pred_n)
        pred_r, pred_th = geo_to_range_angle(Y_pred)
        true_r, true_th = geo_to_range_angle(Y_true)

        for wi in range(Y_true.shape[0]):
            base_idx = int(starts[wi]) + int(cfg.input_len) - 1
            baseline_pos = pos_seq[base_idx]
            baseline_geo = pos_to_geo(baseline_pos[None, :])[0]
            for h in range(cfg.pred_len):
                rows.append({
                    "traj_id": traj_id,
                    "seed": seed,
                    "window_start": int(starts[wi]),
                    "horizon": int(h + 1),
                    "true_range_m": float(true_r[wi, h]),
                    "pred_range_m": float(pred_r[wi, h]),
                    "baseline_range_m": float(baseline_pos[0]),
                    "range_abs_err_m": float(abs(pred_r[wi, h] - true_r[wi, h])),
                    "baseline_range_abs_err_m": float(abs(baseline_pos[0] - true_r[wi, h])),
                    "true_angle_deg": float(true_th[wi, h] * 180.0 / np.pi),
                    "pred_angle_deg": float(pred_th[wi, h] * 180.0 / np.pi),
                    "baseline_angle_deg": float(baseline_pos[1] * 180.0 / np.pi),
                    "angle_abs_err_deg": float(abs(wrap_angle_rad(pred_th[wi, h] - true_th[wi, h])) * 180.0 / np.pi),
                    "baseline_angle_abs_err_deg": float(abs(wrap_angle_rad(baseline_pos[1] - true_th[wi, h])) * 180.0 / np.pi),
                    "true_log_range": float(Y_true[wi, h, 0]),
                    "pred_log_range": float(Y_pred[wi, h, 0]),
                    "baseline_log_range": float(baseline_geo[0]),
                    "true_sin_angle": float(Y_true[wi, h, 1]),
                    "pred_sin_angle": float(Y_pred[wi, h, 1]),
                    "baseline_sin_angle": float(baseline_geo[1]),
                    "true_cos_angle": float(Y_true[wi, h, 2]),
                    "pred_cos_angle": float(Y_pred[wi, h, 2]),
                    "baseline_cos_angle": float(baseline_geo[2]),
                })
        if traj_id == 0 or (traj_id + 1) % 10 == 0:
            s = summarize_eval_rows(rows)[0]
            print(f"[{traj_id+1:03d}/{num_test_traj}] range MAE={s['model_range_mae_m']:.3f} m, angle MAE={s['model_angle_mae_deg']:.3f} deg")

    summary_rows = summarize_eval_rows(rows)
    detail_csv = out_dir / "geo_eval_detail.csv"
    summary_csv = out_dir / "geo_eval_summary.csv"
    summary_json = out_dir / "geo_eval_summary.json"
    write_csv_rows(detail_csv, rows)
    write_csv_rows(summary_csv, summary_rows)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(out_dir),
        "num_test_traj": num_test_traj,
        "traj_len": traj_len,
        "traj_type": traj_type,
        "input_len": cfg.input_len,
        "pred_len": cfg.pred_len,
        "checkpoint_best_val_loss": ckpt.get("best_val_loss", None),
        "checkpoint_best_epoch": ckpt.get("best_epoch", None),
        "overall": summary_rows[0],
        "by_horizon": summary_rows[1:],
    }
    save_json(summary_json, summary)
    print("\nEvaluation finished.")
    print("summary:", summary_csv)
    print("detail :", detail_csv)
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    return summary


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Direct Qwen-LoRA Willie geometry predictor.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer_demo", "eval_saved"])
    parser.add_argument("--save_path", type=str, default="best_model_geo.pt")
    parser.add_argument("--input_len", type=int, default=5)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--num_traj", type=int, default=1000)
    parser.add_argument("--traj_len", type=int, default=32)
    parser.add_argument("--label_mode", type=str, default="true", choices=["true", "music"])
    parser.add_argument("--music_grid_small", action="store_true")
    parser.add_argument("--traj_type", type=str, default="sin", choices=["sin", "linear", "mixed"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="runs/qwen_geo_direct_v1")
    parser.add_argument("--val_pred_samples", type=int, default=512)
    parser.add_argument("--save_val_predictions", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--input_noise_std", type=float, default=0.003)
    parser.add_argument("--loss_log_range_weight", type=float, default=1.0)
    parser.add_argument("--loss_angle_weight", type=float, default=2.0)
    parser.add_argument("--loss_temporal_weight", type=float, default=0.20)
    parser.add_argument("--loss_acceleration_weight", type=float, default=0.05)
    parser.add_argument("--loss_unit_weight", type=float, default=0.05)
    parser.add_argument("--horizon_weights", type=str, default="0.5,0.8,1.0,1.2,1.5")
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-5)
    parser.add_argument("--random_sample_split", action="store_true")
    parser.add_argument("--music_T_mod", type=int, default=4)
    parser.add_argument("--music_L_snap", type=int, default=50)
    parser.add_argument("--music_L_ris", type=int, default=12)
    parser.add_argument("--music_SNR_dB", type=float, default=10.0)
    parser.add_argument("--hf_model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--hf_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hf_target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--no_hf_trust_remote_code", action="store_true")
    parser.add_argument("--unfreeze_hf_base", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval_output_dir", type=str, default="eval_geo_saved")
    parser.add_argument("--num_test_traj", type=int, default=100)
    parser.add_argument("--seed_start", type=int, default=10000)
    args = parser.parse_args()

    cfg = EchoGeoConfig(
        input_len=args.input_len,
        pred_len=args.pred_len,
        music_T_mod=args.music_T_mod,
        music_L_snap=args.music_L_snap,
        music_L_ris=args.music_L_ris,
        music_SNR_dB=args.music_SNR_dB,
        hf_model_name=args.hf_model_name,
        hf_cache_dir=args.hf_cache_dir,
        hf_dtype=args.hf_dtype,
        hf_target_modules=args.hf_target_modules,
        hf_trust_remote_code=not args.no_hf_trust_remote_code,
        hf_freeze_base=not args.unfreeze_hf_base,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        input_noise_std=args.input_noise_std,
        dropout=args.dropout,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        loss_log_range_weight=args.loss_log_range_weight,
        loss_angle_weight=args.loss_angle_weight,
        loss_temporal_weight=args.loss_temporal_weight,
        loss_acceleration_weight=args.loss_acceleration_weight,
        loss_unit_weight=args.loss_unit_weight,
        horizon_weights=args.horizon_weights,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        split_by_order=not args.random_sample_split,
        output_dir=args.output_dir,
        val_pred_samples=args.val_pred_samples,
        save_val_predictions=args.save_val_predictions,
    )

    if args.mode == "train":
        X, Y, P, starts = generate_training_dataset_by_music(
            num_traj=args.num_traj,
            traj_len=args.traj_len,
            cfg=cfg,
            label_mode=args.label_mode,
            music_grid_small=args.music_grid_small,
            traj_type=args.traj_type,
        )
        print(f"Dataset ready: X={X.shape}, Y_geo={Y.shape}, P={P.shape}")
        info = train_echo_to_geo_lora(X, Y, cfg, save_path=args.save_path, device=args.device, output_dir=args.output_dir)
        print("\nTraining finished.")
        print(json.dumps(info, indent=2, ensure_ascii=False))

    elif args.mode == "infer_demo":
        echo_seq, geo_seq, pos_seq = generate_echo_geo_trajectory_by_music(
            traj_len=args.input_len + args.pred_len,
            seed=cfg.seed,
            cfg=cfg,
            label_mode=args.label_mode,
            music_grid_small=args.music_grid_small,
            traj_type=args.traj_type,
        )
        pred_r, pred_th, _ = predict_future_geometry_from_echo_history(echo_seq[:args.input_len], checkpoint_path=args.save_path, device=args.device)
        print("Predicted ranges:", pred_r)
        print("Predicted angles deg:", np.rad2deg(pred_th))
        print("Reference positions:")
        print(pos_seq[args.input_len:args.input_len + args.pred_len])

    elif args.mode == "eval_saved":
        evaluate_saved_checkpoint(
            checkpoint_path=args.save_path,
            output_dir=args.eval_output_dir,
            num_test_traj=args.num_test_traj,
            traj_len=args.traj_len,
            seed_start=args.seed_start,
            label_mode=args.label_mode,
            music_grid_small=args.music_grid_small,
            traj_type=args.traj_type,
            device=args.device,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
