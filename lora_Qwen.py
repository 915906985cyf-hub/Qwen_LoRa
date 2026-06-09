from __future__ import annotations

"""
LoRa_hf_qwen_lora.py

Hugging Face pretrained LLM + PEFT-LoRA version for echo-to-channel prediction.

Main idea:
    complex echo history -> CNN echo encoder -> inputs_embeds of a pretrained HF LLM
    -> LoRA-adapted attention layers -> regression head -> future Willie channel.

This is different from the previous local GPT-style model:
    previous: local randomly initialized GPT-style backbone
    this:     pretrained Hugging Face LLM backbone + PEFT LoRA adapters

Recommended first base model:
    Qwen/Qwen2.5-0.5B

Before running:
    pip install transformers peft accelerate safetensors
"""



import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Hugging Face / PEFT imports are checked at runtime, so the file gives a clear
# error message if the packages have not been installed.
try:
    from transformers import AutoConfig, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover
    AutoConfig = None
    AutoModelForCausalLM = None
    LoraConfig = None
    get_peft_model = None

Array = np.ndarray


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class EchoLoRAConfig:
    input_len: int = 5
    pred_len: int = 5
    L_ris: int = 12

    # Echo tensor shape. These values are updated after data generation/loading.
    echo_dim: int = 4 * 6 * 6
    L_snap: int = 50

    # MUSIC simulator overrides. These reduce the raw echo dimension and help
    # prevent the first embedding layer from becoming too large.
    music_T_mod: int = 4
    music_L_snap: int = 50
    music_L_ris: int = 12
    music_SNR_dB: float = 10.0

    # Hugging Face pretrained LLM backbone.
    # Qwen/Qwen2.5-0.5B is a practical first choice. If VRAM is limited,
    # use a smaller model; if VRAM is sufficient, try a 1B-scale model.
    hf_model_name: str = "Qwen/Qwen2.5-0.5B"
    hf_dtype: str = "float16"  # float16, bfloat16, or float32
    hf_target_modules: str = "q_proj,v_proj"
    hf_trust_remote_code: bool = True
    hf_freeze_base: bool = True

    # Channel model used for labels h_w = sqrt(L0 r^-alpha_w) a(theta).
    L0: float = 1e-3
    alpha_w: float = 2.2

    # Moderate GPT-style numerical backbone. This is stronger than the strict anti-overfitting baseline,
    # but still much smaller than the original flatten-Linear model.
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.15

    # LoRA adapters.
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.08

    # Training.
    batch_size: int = 8
    epochs: int = 50
    lr: float = 2e-5
    weight_decay: float = 3e-4
    grad_clip: float = 0.5
    val_ratio: float = 0.2
    seed: int = 42
    norm_eps: float = 1e-8

    # Regularization and safer validation.
    input_noise_std: float = 0.003
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-5
    split_by_order: bool = True  # True: last 20% windows/trajectories as validation.


# =============================================================================
# Basic helpers
# =============================================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def complex_to_ri(x: np.ndarray) -> np.ndarray:
    """Complex array -> real-imag last dimension."""
    x = np.asarray(x)
    return np.stack([np.real(x), np.imag(x)], axis=-1).astype(np.float32)


def ri_to_complex(x: np.ndarray) -> np.ndarray:
    """Real-imag last dimension -> complex array."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != 2:
        raise ValueError("Last dimension must be 2 for [real, imag].")
    return x[..., 0] + 1j * x[..., 1]


def wrap_angle_rad(theta: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi


def ula_steering(L: int, angle_rad: float, d_over_lambda: float = 0.5) -> np.ndarray:
    idx = np.arange(L, dtype=float)
    phase = 2.0 * np.pi * d_over_lambda * idx * np.sin(angle_rad)
    return np.exp(1j * phase)


def ris_willie_channel_from_range_angle(
    L_ris: int,
    target_range: float,
    target_angle: float,
    L0: float = 1e-3,
    alpha_w: float = 2.2,
) -> np.ndarray:
    """Construct STAR-RIS-to-Willie channel h_w from range and angle."""
    target_range = max(float(target_range), 1e-9)
    return np.sqrt(L0 * target_range ** (-alpha_w)) * ula_steering(L_ris, float(target_angle))


def channel_to_range_angle(
    h_w: np.ndarray,
    L0: float = 1e-3,
    alpha_w: float = 2.2,
    d_over_lambda: float = 0.5,
) -> Tuple[float, float]:
    """
    Recover an approximate range/angle from a LoS channel vector.

    This is mainly for compatibility with existing AO solvers that still expect
    (range, angle). The channel itself is the predicted quantity.
    """
    h = np.asarray(h_w, dtype=np.complex128).reshape(-1)
    L = h.size
    norm2 = max(float(np.vdot(h, h).real), 1e-30)
    target_range = (L * L0 / norm2) ** (1.0 / alpha_w)

    if L >= 2:
        # h[l+1] conj(h[l]) ~= exp(j * 2pi d/lambda sin(theta)).
        phase_step = np.angle(np.mean(h[1:] * np.conj(h[:-1])))
        sin_theta = phase_step / (2.0 * np.pi * d_over_lambda)
        sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
        theta = float(np.arcsin(sin_theta))
    else:
        theta = 0.0

    return float(target_range), float(wrap_angle_rad(theta))


def channel_sequence_to_range_angle(
    h_seq: np.ndarray,
    L0: float = 1e-3,
    alpha_w: float = 2.2,
) -> Tuple[np.ndarray, np.ndarray]:
    h_seq = np.asarray(h_seq)
    ranges, angles = [], []
    for h in h_seq:
        r, th = channel_to_range_angle(h, L0=L0, alpha_w=alpha_w)
        ranges.append(r)
        angles.append(th)
    return np.asarray(ranges, dtype=float), np.asarray(angles, dtype=float)


class StandardScaler:
    """Feature-wise standardization for real-valued tensors."""

    def __init__(self, eps: float = 1e-8):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = float(eps)

    def fit(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float32)
        self.mean = arr.mean(axis=tuple(range(arr.ndim - 1)), keepdims=True).astype(np.float32)
        self.std = (arr.std(axis=tuple(range(arr.ndim - 1)), keepdims=True) + self.eps).astype(np.float32)

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


# =============================================================================
# Dataset construction
# =============================================================================
def build_echo_channel_windows(
    echo_seq: np.ndarray,
    h_seq: np.ndarray,
    input_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build supervised windows.

    echo_seq: complex, shape (T, echo_dim, L_snap)
    h_seq:    complex, shape (T, L_ris)
    returns:
        X: real, shape (num_samples,input_len,echo_dim,L_snap,2)
        Y: real, shape (num_samples,pred_len,L_ris,2)
    """
    echo_seq = np.asarray(echo_seq)
    h_seq = np.asarray(h_seq)
    if echo_seq.ndim != 3:
        raise ValueError("echo_seq must have shape (T, echo_dim, L_snap).")
    if h_seq.ndim != 2:
        raise ValueError("h_seq must have shape (T, L_ris).")
    if echo_seq.shape[0] != h_seq.shape[0]:
        raise ValueError("echo_seq and h_seq must have the same time length.")

    T = echo_seq.shape[0]
    window = input_len + pred_len
    if T < window:
        raise ValueError(f"T={T} is too short for input_len+pred_len={window}.")

    X, Y = [], []
    for start in range(T - window + 1):
        X.append(complex_to_ri(echo_seq[start:start + input_len]))
        Y.append(complex_to_ri(h_seq[start + input_len:start + window]))
    return np.stack(X, axis=0), np.stack(Y, axis=0)


def _try_import_music_module():
    try:
        import music_fda_ris as music_mod
        return music_mod
    except Exception as err:
        raise ImportError(
            "Cannot import music_fda_ris.py. Save your MUSIC code as music_fda_ris.py "
            "in the same folder, or train from a saved echo/channel .npz file."
        ) from err


def _apply_music_overrides(music_cfg, cfg: EchoLoRAConfig):
    """Make the MUSIC simulator consistent with the anti-overfitting setup."""
    if hasattr(music_cfg, "T_mod"):
        music_cfg.T_mod = int(cfg.music_T_mod)
    if hasattr(music_cfg, "L_snap"):
        music_cfg.L_snap = int(cfg.music_L_snap)
    if hasattr(music_cfg, "L_ris"):
        music_cfg.L_ris = int(cfg.music_L_ris)
    if hasattr(music_cfg, "SNR_dB"):
        music_cfg.SNR_dB = float(cfg.music_SNR_dB)
    return music_cfg


def generate_true_trajectory(traj_len: int, seed: int = 1) -> np.ndarray:
    """Smooth synthetic Willie range-angle trajectory."""
    rng = np.random.default_rng(seed)
    t = np.arange(traj_len, dtype=np.float32)
    r = 30.0 + rng.uniform(3.0, 7.0) * np.sin(2 * np.pi * rng.uniform(0.04, 0.12) * t + rng.uniform(0, 2*np.pi))
    th = np.deg2rad(20.0) + np.deg2rad(rng.uniform(5.0, 12.0)) * np.cos(2 * np.pi * rng.uniform(0.04, 0.12) * t + rng.uniform(0, 2*np.pi))
    return np.stack([r, th], axis=-1).astype(np.float32)


def generate_echo_channel_trajectory_by_music(
    traj_len: int,
    seed: int,
    L0: float = 1e-3,
    alpha_w: float = 2.2,
    label_mode: str = "true",
    theta_grid: Optional[np.ndarray] = None,
    range_grid: Optional[np.ndarray] = None,
    cfg_override: Optional[EchoLoRAConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one trajectory using your MUSIC simulator.

    label_mode:
        "true"  - label h_w is constructed from true range/angle.
        "music" - label h_w is constructed from MUSIC-estimated range/angle.

    returns:
        echo_seq: complex, shape (T, echo_dim, L_snap)
        h_seq:    complex, shape (T, L_ris)
        pos_seq:  real,    shape (T, 2), actual label positions used for h_seq
    """
    music_mod = _try_import_music_module()
    cfg_m = music_mod.MusicConfig()
    if cfg_override is not None:
        cfg_m = _apply_music_overrides(cfg_m, cfg_override)

    Phi = music_mod.generate_time_varying_ris_phases(cfg_m.L_ris, cfg_m.T_mod, seed=seed)
    true_pos = generate_true_trajectory(traj_len, seed=seed)

    echo_list, h_list, pos_list = [], [], []
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
            raise ValueError("label_mode must be 'true' or 'music'.")

        h_w = ris_willie_channel_from_range_angle(
            cfg_m.L_ris,
            label_r,
            label_th,
            L0=L0,
            alpha_w=alpha_w,
        )
        echo_list.append(X_echo)
        h_list.append(h_w)
        pos_list.append([label_r, label_th])

    return np.asarray(echo_list), np.asarray(h_list), np.asarray(pos_list, dtype=np.float32)


def generate_training_dataset_by_music(
    num_traj: int,
    traj_len: int,
    cfg: EchoLoRAConfig,
    label_mode: str = "true",
    music_grid_small: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if music_grid_small:
        theta_grid = np.linspace(-40.0, 40.0, 21) * np.pi / 180.0
        range_grid = np.linspace(10.0, 50.0, 21)
    else:
        theta_grid = None
        range_grid = None

    X_all, Y_all = [], []
    for i in range(num_traj):
        echo_seq, h_seq, _ = generate_echo_channel_trajectory_by_music(
            traj_len=traj_len,
            seed=cfg.seed + i,
            L0=cfg.L0,
            alpha_w=cfg.alpha_w,
            label_mode=label_mode,
            theta_grid=theta_grid,
            range_grid=range_grid,
            cfg_override=cfg,
        )
        Xi, Yi = build_echo_channel_windows(echo_seq, h_seq, cfg.input_len, cfg.pred_len)
        X_all.append(Xi)
        Y_all.append(Yi)
        if i == 0 or (i + 1) % 5 == 0:
            print(f"Generated trajectory {i+1}/{num_traj}: X={Xi.shape}, Y={Yi.shape}")
    return np.concatenate(X_all, axis=0), np.concatenate(Y_all, axis=0)


def load_echo_channel_npz(path: str, cfg: EchoLoRAConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-saved data.

    Supported .npz fields:
        echo_seq: complex, shape (T,echo_dim,L_snap) or (num_traj,T,echo_dim,L_snap)
        h_seq:    complex, shape (T,L_ris) or (num_traj,T,L_ris)
    OR already-windowed:
        X: real, shape (num_samples,input_len,echo_dim,L_snap,2)
        Y: real, shape (num_samples,pred_len,L_ris,2)
    """
    data = np.load(path, allow_pickle=True)
    if "X" in data and "Y" in data:
        return data["X"].astype(np.float32), data["Y"].astype(np.float32)

    if "echo_seq" not in data or "h_seq" not in data:
        raise ValueError("npz must contain either X/Y or echo_seq/h_seq.")

    echo_seq = data["echo_seq"]
    h_seq = data["h_seq"]
    if echo_seq.ndim == 3:
        return build_echo_channel_windows(echo_seq, h_seq, cfg.input_len, cfg.pred_len)

    if echo_seq.ndim == 4:
        X_all, Y_all = [], []
        for i in range(echo_seq.shape[0]):
            Xi, Yi = build_echo_channel_windows(echo_seq[i], h_seq[i], cfg.input_len, cfg.pred_len)
            X_all.append(Xi)
            Y_all.append(Yi)
        return np.concatenate(X_all, axis=0), np.concatenate(Y_all, axis=0)

    raise ValueError("Unsupported echo_seq shape in npz.")


class EchoChannelDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


# =============================================================================
# Hugging Face pretrained LLM + PEFT-LoRA model
# =============================================================================
def _require_hf_packages() -> None:
    if AutoConfig is None or AutoModelForCausalLM is None or LoraConfig is None or get_peft_model is None:
        raise ImportError(
            "Missing Hugging Face packages. Please install them first:\n"
            "    pip install transformers peft accelerate safetensors\n"
        )


def _torch_dtype_from_name(name: str):
    name = str(name).lower()
    if name in ["fp16", "float16", "half"]:
        return torch.float16
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if name in ["fp32", "float32", "full"]:
        return torch.float32
    raise ValueError("hf_dtype must be one of: float16, bfloat16, float32.")


class EchoConvEmbed(nn.Module):
    """
    CNN echo encoder.

    It compresses each complex echo matrix:
        (echo_dim, L_snap, 2)
    into one vector compatible with the hidden size of the pretrained LLM.

    The final output is:
        (B, input_len, hidden_size)
    which will be passed to the HF model as inputs_embeds.
    """

    def __init__(self, cfg: EchoLoRAConfig, hidden_size: int):
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
        # x: (B, input_len, echo_dim, L_snap, 2)
        B, T, E, S, RI = x.shape
        if RI != 2:
            raise ValueError("The last dimension of x must be 2 for [real, imag].")
        z = x.permute(0, 1, 4, 2, 3).contiguous().view(B * T, 2, E, S)
        z = self.net(z)
        return z.view(B, T, self.hidden_size)


class EchoToChannelHFLoRA(nn.Module):
    """
    Historical complex echoes -> future complex Willie channel vectors
    using a Hugging Face pretrained causal language model as the temporal backbone.

    Important:
        We do NOT tokenize the echo matrix as text.
        Instead, the numerical echo is projected into the LLM embedding space and
        passed through the model via inputs_embeds.
    """

    def __init__(self, cfg: EchoLoRAConfig):
        super().__init__()
        _require_hf_packages()
        self.cfg = cfg

        hf_config = AutoConfig.from_pretrained(
            cfg.hf_model_name,
            trust_remote_code=cfg.hf_trust_remote_code,
        )
        hidden_size = int(getattr(hf_config, "hidden_size"))
        self.hidden_size = hidden_size

        dtype = _torch_dtype_from_name(cfg.hf_dtype)

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_model_name,
            torch_dtype=dtype,
            trust_remote_code=cfg.hf_trust_remote_code,
            low_cpu_mem_usage=True,
        )

        # Make sure the model returns hidden states for regression.
        base_model.config.output_hidden_states = True
        base_model.config.use_cache = False

        # Optional: freeze the pretrained base. PEFT will make LoRA parameters trainable.
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

        # Numerical front-end and regression output head remain trainable.
        self.echo_embed = EchoConvEmbed(cfg, hidden_size=hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.input_len, hidden_size))
        self.drop = nn.Dropout(cfg.dropout)

        out_dim = cfg.pred_len * cfg.L_ris * 2
        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, min(512, hidden_size)),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(min(512, hidden_size), out_dim),
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
        # x: (B, input_len, echo_dim, L_snap, 2)
        B, T = x.shape[0], x.shape[1]
        if T != self.cfg.input_len:
            raise ValueError(f"Expected input_len={self.cfg.input_len}, got {T}.")

        h = self.echo_embed(x) + self.pos_embed[:, :T, :]
        h = self.drop(h)

        # HF model weights may be fp16/bf16. inputs_embeds must match that dtype.
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
        return pred.view(B, self.cfg.pred_len, self.cfg.L_ris, 2)


# Keep this name so the rest of the training/inference code can stay unchanged.
EchoToChannelLoRAGPT = EchoToChannelHFLoRA


def freeze_backbone_except_lora_and_io(model: nn.Module) -> None:
    """
    For the HF version:
        - pretrained base weights remain frozen
        - PEFT LoRA weights are trainable
        - echo encoder / positional embeddings / regression head are trainable
    """
    for name, p in model.named_parameters():
        trainable = (
            "lora_" in name
            or "echo_embed" in name
            or "pos_embed" in name
            or "regression_head" in name
        )
        p.requires_grad = trainable


def trainable_state_dict(model: nn.Module) -> dict:
    """
    Save only trainable parameters instead of saving the whole pretrained LLM.
    This keeps checkpoints small. Loading will reconstruct the base HF model
    from cfg.hf_model_name, then load these trainable weights.
    """
    return {
        name: p.detach().cpu()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# Training / inference
# =============================================================================
def channel_nmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # pred/target: (B,pred_len,L_ris,2)
    num = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    den = torch.sum(target ** 2, dim=(1, 2, 3)).clamp_min(eps)
    return torch.mean(num / den)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    vals = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        vals.append(float(channel_nmse_loss(model(xb), yb).item()))
    return float(np.mean(vals)) if vals else float("nan")


def train_echo_to_channel_lora(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: EchoLoRAConfig,
    save_path: str = "lora_echo_to_channel.pt",
    device: Optional[str] = None,
) -> dict:
    set_seed(cfg.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    cfg.echo_dim = int(X.shape[2])
    cfg.L_snap = int(X.shape[3])
    cfg.L_ris = int(Y.shape[2])

    n = X.shape[0]
    n_val = max(1, int(n * cfg.val_ratio))
    if cfg.split_by_order:
        # Because generate_training_dataset_by_music concatenates trajectories in
        # order, this usually means the last group of trajectories is reserved for
        # validation. This is safer than random adjacent-window leakage.
        train_idx = np.arange(0, n - n_val)
        val_idx = np.arange(n - n_val, n)
    else:
        idx = np.arange(n)
        np.random.shuffle(idx)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

    # Fit scalers only on training data to avoid validation leakage.
    x_scaler = StandardScaler(eps=cfg.norm_eps)
    y_scaler = StandardScaler(eps=cfg.norm_eps)
    x_scaler.fit(X[train_idx])
    y_scaler.fit(Y[train_idx])
    X_train = x_scaler.transform(X[train_idx])
    Y_train = y_scaler.transform(Y[train_idx])
    X_val = x_scaler.transform(X[val_idx])
    Y_val = y_scaler.transform(Y[val_idx])

    train_loader = DataLoader(
        EchoChannelDataset(X_train, Y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        EchoChannelDataset(X_val, Y_val),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = EchoToChannelLoRAGPT(cfg)
    freeze_backbone_except_lora_and_io(model)
    model.to(dev)
    total, trainable = count_params(model)
    print(f"Device: {dev}")
    print(f"Dataset: train={len(train_idx)}, val={len(val_idx)}, X={X.shape}, Y={Y.shape}")
    print(f"Split mode: {'order/trajectory-like' if cfg.split_by_order else 'random sample'}")
    print(f"Parameters: total={total:,}, trainable={trainable:,} ({100*trainable/total:.2f}%)")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=max(2, cfg.early_stop_patience // 2),
    )
    best_val, best_epoch = float("inf"), -1
    bad_epochs = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            if cfg.input_noise_std > 0:
                xb_model = xb + cfg.input_noise_std * torch.randn_like(xb)
            else:
                xb_model = xb
            pred = model(xb_model)
            loss = channel_nmse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
            opt.step()
            losses.append(float(loss.item()))

        val = evaluate(model, val_loader, dev)
        sched.step(val)
        tr = float(np.mean(losses)) if losses else float("nan")
        current_lr = float(opt.param_groups[0]["lr"])

        improved = val < best_val - cfg.early_stop_min_delta
        if improved:
            best_val, best_epoch = val, ep
            bad_epochs = 0
            torch.save({
                "config": asdict(cfg),
                "model_state_dict": trainable_state_dict(model),
                "x_scaler": x_scaler.state_dict(),
                "y_scaler": y_scaler.state_dict(),
                "best_val_nmse": best_val,
                "best_epoch": best_epoch,
            }, save_path)
        else:
            bad_epochs += 1

        if ep == 1 or ep % 5 == 0 or ep == cfg.epochs or improved:
            print(
                f"Epoch {ep:04d}/{cfg.epochs} | "
                f"train_NMSE={tr:.6e} | val_NMSE={val:.6e} | "
                f"best={best_val:.6e}@{best_epoch} | lr={current_lr:.2e} | bad={bad_epochs}"
            )

        if cfg.early_stop_patience > 0 and bad_epochs >= cfg.early_stop_patience:
            print(
                f"Early stopping at epoch {ep}. "
                f"Best val_NMSE={best_val:.6e} at epoch {best_epoch}."
            )
            break

    return {
        "save_path": save_path,
        "best_val_nmse": best_val,
        "best_epoch": best_epoch,
        "total_params": total,
        "trainable_params": trainable,
    }


def load_trained_echo_lora(checkpoint_path: str, device: Optional[str] = None):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint_path, map_location=dev)
    cfg = EchoLoRAConfig(**ckpt["config"])
    model = EchoToChannelLoRAGPT(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        print("Unexpected keys when loading checkpoint:", unexpected)
    # Missing keys are expected because the frozen HF base model is loaded from hf_model_name.
    model.to(dev)
    model.eval()
    x_scaler = StandardScaler(eps=cfg.norm_eps)
    y_scaler = StandardScaler(eps=cfg.norm_eps)
    x_scaler.load_state_dict(ckpt["x_scaler"])
    y_scaler.load_state_dict(ckpt["y_scaler"])
    return model, x_scaler, y_scaler, cfg, dev


@torch.no_grad()
def predict_future_channels_from_echo_history(
    echo_history: np.ndarray,
    checkpoint_path: str = "lora_echo_to_channel.pt",
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Predict future Willie channels from historical echoes.

    echo_history:
        complex array with shape (input_len, echo_dim, L_snap)
    returns:
        complex array with shape (pred_len, L_ris)
    """
    model, x_scaler, y_scaler, cfg, dev = load_trained_echo_lora(checkpoint_path, device=device)
    echo_history = np.asarray(echo_history)
    if echo_history.shape != (cfg.input_len, cfg.echo_dim, cfg.L_snap):
        raise ValueError(f"echo_history must have shape ({cfg.input_len},{cfg.echo_dim},{cfg.L_snap}), got {echo_history.shape}.")
    X = complex_to_ri(echo_history)[None, ...]
    Xn = x_scaler.transform(X)
    xb = torch.as_tensor(Xn, dtype=torch.float32, device=dev)
    pred_n = model(xb).cpu().numpy()[0]
    pred_ri = y_scaler.inverse_transform(pred_n)
    return ri_to_complex(pred_ri).astype(np.complex128)


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="HF pretrained LLM + PEFT-LoRA echo-to-channel predictor for Willie CSI.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer_demo"])
    parser.add_argument("--data_npz", type=str, default=None, help="Optional .npz with echo_seq/h_seq or X/Y.")
    parser.add_argument("--save_path", type=str, default="lora_echo_to_channel_hf_qwen.pt")
    parser.add_argument("--input_len", type=int, default=5)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--num_traj", type=int, default=80)
    parser.add_argument("--traj_len", type=int, default=32)
    parser.add_argument("--label_mode", type=str, default="true", choices=["true", "music"])
    parser.add_argument("--music_grid_small", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--lora_dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--input_noise_std", type=float, default=0.003)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-5)
    parser.add_argument("--random_sample_split", action="store_true", help="Use random sample split instead of order/trajectory-like split.")
    parser.add_argument("--music_T_mod", type=int, default=4)
    parser.add_argument("--music_L_snap", type=int, default=50)
    parser.add_argument("--music_L_ris", type=int, default=12)
    parser.add_argument("--music_SNR_dB", type=float, default=10.0)
    parser.add_argument("--hf_model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--hf_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hf_target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--no_hf_trust_remote_code", action="store_true")
    parser.add_argument("--unfreeze_hf_base", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = EchoLoRAConfig(
        input_len=args.input_len,
        pred_len=args.pred_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        input_noise_std=args.input_noise_std,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        split_by_order=not args.random_sample_split,
        music_T_mod=args.music_T_mod,
        music_L_snap=args.music_L_snap,
        music_L_ris=args.music_L_ris,
        music_SNR_dB=args.music_SNR_dB,
        hf_model_name=args.hf_model_name,
        hf_dtype=args.hf_dtype,
        hf_target_modules=args.hf_target_modules,
        hf_trust_remote_code=not args.no_hf_trust_remote_code,
        hf_freeze_base=not args.unfreeze_hf_base,
    )

    if args.mode == "train":
        if args.data_npz is not None:
            X, Y = load_echo_channel_npz(args.data_npz, cfg)
        else:
            X, Y = generate_training_dataset_by_music(
                num_traj=args.num_traj,
                traj_len=args.traj_len,
                cfg=cfg,
                label_mode=args.label_mode,
                music_grid_small=args.music_grid_small,
            )
        print(f"Dataset ready: X={X.shape}, Y={Y.shape}")
        info = train_echo_to_channel_lora(X, Y, cfg, save_path=args.save_path, device=args.device)
        print("\nTraining finished.")
        print(json.dumps(info, indent=2))

    elif args.mode == "infer_demo":
        echo_seq, h_seq, pos = generate_echo_channel_trajectory_by_music(
            traj_len=args.input_len + args.pred_len,
            seed=cfg.seed,
            L0=cfg.L0,
            alpha_w=cfg.alpha_w,
            label_mode=args.label_mode,
            cfg_override=cfg,
        )
        pred = predict_future_channels_from_echo_history(
            echo_seq[:args.input_len],
            checkpoint_path=args.save_path,
            device=args.device,
        )
        print("Predicted h_w shape:", pred.shape)
        print("Label h_w shape:", h_seq[args.input_len:args.input_len + args.pred_len].shape)
        pred_r, pred_th = channel_sequence_to_range_angle(pred, L0=cfg.L0, alpha_w=cfg.alpha_w)
        print("Predicted equivalent ranges:", pred_r)
        print("Predicted equivalent angles deg:", np.rad2deg(pred_th))
        print("Reference label positions:")
        print(pos[args.input_len:args.input_len + args.pred_len])


if __name__ == "__main__":
    main()
