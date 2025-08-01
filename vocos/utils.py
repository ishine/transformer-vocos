import math
import os
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
import torchaudio
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_schedule_with_warmup_lr_lambda(current_step: int, *,
                                               num_warmup_steps: int,
                                               num_training_steps: int,
                                               num_cycles: float):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps))
    return max(
        0.0,
        0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float = 0.5,
                                    last_epoch: int = -1):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def frame_paddings(paddings: torch.Tensor, *, frame_size: int,
                   hop_size: int) -> torch.Tensor:
    """Frames paddings.

    Args:
        paddings: A Tensor of shape `[..., seq_len]`.
        frame_size: Size of frames.
        hop_size: Hop size of frames.

    Returns:
        A Tensor of shape `[..., num_windows]`, where each value represents
        the maximum padding value in the corresponding frame.

    Raises:
        ValueError: If the input is invalid.
    """
    if hop_size > frame_size:
        raise ValueError(
            f"hop_size {hop_size} must be smaller than frame_size {frame_size}."
        )

    # Unfold to create overlapping frames
    paddings_frame = paddings.unfold(-1, frame_size, hop_size)

    # Compute max padding per frame
    out_paddings = paddings_frame.max(dim=-1).values
    return out_paddings


class STFT(torch.nn.Module):

    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        padding="center",
        window_fn=torch.hann_window,
    ) -> None:
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.paddding = padding
        self.win = window_fn(self.win_length)

    def forward(self, audio, paddings):
        """
        Args:
            audio: (B, C, T) or (B, T) or (T,)
            paddings: (B, T) or (T,) optional, corresponding to the padding mask in the audio data
                    (1 indicates padding, 0 indicates valid data).
            (mel_features, out_paddings)

        Returns:
            mel_features: (B, n_mels, T')
            out_paddings: (B, T') propagated padding information.
        """
        # Manual padding is needed when `padding="same"`
        pad = (self.win_length or self.n_fft) - self.hop_length
        if self.padding == "center":
            pad_left, pad_right = pad // 2, pad - pad // 2
            audio = torch.nn.functional.pad(audio, (pad_left, pad_right),
                                            mode="reflect")
            # Padding should also be adjusted
            if paddings is not None:
                paddings = torch.nn.functional.pad(paddings,
                                                   (pad_left, pad_right),
                                                   value=1)
        elif self.padding == "same":
            audio = torch.nn.functional.pad(audio, (0, pad), mode="reflect")
            # Padding should also be adjusted
            if paddings is not None:
                paddings = torch.nn.functional.pad(paddings, (0, pad), value=1)

        spec_f = torch.stft(
            audio,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.win,
            return_complex=True,
            center=False,
        )

        # Compute padding propagation
        out_paddings = frame_paddings(paddings,
                                      frame_size=self.n_fft,
                                      hop_size=self.hop_length)
        return spec_f, out_paddings


class Spectrogram(STFT):

    def __init__(self,
                 n_fft=1024,
                 hop_length=256,
                 padding="center",
                 window_fn=torch.hann_window,
                 window_norm: bool = True,
                 power: Optional[int] = None) -> None:
        super().__init__(n_fft, hop_length, padding, window_fn)
        self.power = power
        self.window_norm = window_norm

    def forward(self, audio, paddings):
        shape = audio.size()
        audio = audio.reshape(-1, shape[-1])
        spec_f, out_paddings = super().forward(audio, paddings)
        spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])

        if self.window_norm:
            spec_f /= self.win.pow(2.0).sum().sqrt()
        if self.power:
            if self.power == 1.0:
                return spec_f.abs(), out_paddings
            return spec_f.abs().pow(self.power)
        return spec_f, out_paddings


class MelSpectrogram(torch.nn.Module):

    def __init__(
        self,
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding="center",
        power=1,
        fmin=0,
        fmax=8000,
        norm="slaney",
        mel_scale="slaney",
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False,
            power=power,
            f_min=fmin,
            f_max=fmax,
            norm=norm,
            mel_scale=mel_scale,
        )

    def forward(self, audio, paddings=None):
        """
        Args:
            audio: (B, C, T) or (B, T) or (T,)
            paddings: (B, T) or (T,) optional, corresponding to the padding mask in the audio data
                    (1 indicates padding, 0 indicates valid data).
            (mel_features, out_paddings)

        Returns:
            mel_features: (B, n_mels, T')
            out_paddings: (B, T') propagated padding information.
        """
        # Manual padding is needed when `padding="same"`
        pad = (self.mel_spec.win_length
               or self.mel_spec.n_fft) - self.mel_spec.hop_length
        if self.padding == "center":
            pad_left, pad_right = pad // 2, pad - pad // 2
            audio = torch.nn.functional.pad(audio, (pad_left, pad_right),
                                            mode="reflect")
            # Padding should also be adjusted
            if paddings is not None:
                paddings = torch.nn.functional.pad(paddings,
                                                   (pad_left, pad_right),
                                                   value=1)
        elif self.padding == "same":
            audio = torch.nn.functional.pad(audio, (0, pad), mode="reflect")
            # Padding should also be adjusted
            if paddings is not None:
                paddings = torch.nn.functional.pad(paddings, (0, pad), value=1)

        # Compute Mel spectrogram
        mel = self.mel_spec(audio)  # (B, n_mels, T')

        # Compute padding propagation
        out_paddings = frame_paddings(paddings,
                                      frame_size=self.mel_spec.n_fft,
                                      hop_size=self.mel_spec.hop_length)

        # Avoid log(0) errors
        features = torch.log(torch.clip(mel, min=1e-6))

        return features, out_paddings


def init_distributed(configs):

    local_rank = os.environ.get('LOCAL_RANK', 0)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    print('training on multiple gpus, this gpu {}'.format(local_rank) +
          ', rank {}, world_size {}'.format(rank, world_size))

    return world_size, local_rank, rank


def get_model(config):
    if hasattr(config, 'model_type'):
        if config.model_type == 'conformer':
            from efficient_conformer.model import Conformer
            return Conformer(config)
    from vocos.model import Transformer
    return Transformer(config)
