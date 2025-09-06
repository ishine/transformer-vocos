from typing import Optional, Tuple

import torch
from wenet.transformer.encoder import TransformerEncoder
from wenet.utils.common import mask_to_bias
from wenet.utils.mask import causal_or_lookahead_mask


class Transformer(TransformerEncoder):

    def __init__(self, config):
        super().__init__(
            input_size=config.n_mels,
            n_expert=config.n_expert,
            n_expert_activated=config.n_expert_activated,
            attention_heads=config.attention_heads,
            linear_units=config.linear_units,
            num_blocks=config.num_blocks,
            dropout_rate=config.dropout_rate,
            positional_dropout_rate=config.positional_dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            input_layer=config.input_layer,
            pos_enc_layer_type=config.pos_enc_layer_type,
            normalize_before=config.normalize_before,
            static_chunk_size=config.static_chunk_size,
            use_dynamic_chunk=config.use_dynamic_chunk,
            global_cmvn=config.global_cmvn,
            use_dynamic_left_chunk=config.use_dynamic_left_chunk,
            query_bias=config.query_bias,
            key_bias=config.key_bias,
            value_bias=config.value_bias,
            activation_type=config.activation_type,
            gradient_checkpointing=config.gradient_checkpointing,
            use_sdpa=config.use_sdpa,
            layer_norm_type=config.layer_norm_type,
            norm_eps=config.norm_eps,
            n_kv_head=config.n_kv_head,
            head_dim=config.head_dim,
            selfattention_layer_type=config.selfattention_layer_type,
            mlp_type=config.mlp_type,
            mlp_bias=config.mlp_bias,
        )
        self.config = config

    def forward(self, mels, masks):

        masks = masks.unsqueeze(1)
        xs, pos_emb, masks = self.embed(mels, masks)
        mask_pad = masks
        att_mask = causal_or_lookahead_mask(masks, self.config.right_context,
                                            self.config.left_context)
        if self.use_sdpa:
            att_mask = mask_to_bias(att_mask, xs.dtype)
        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, att_mask, pos_emb,
                                                  mask_pad)
        else:
            xs = self.forward_layers(xs, att_mask, pos_emb, mask_pad)
        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_chunk_by_chunk(self,
                               xs,
                               decoding_chunk_size,
                               num_decoding_left_chunks=-1):
        pass

    def forward_chunk(self, xs, offset, required_cache_size, att_cache,
                      cnn_cache, att_mask):
        pass


class ISTFT(torch.nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor, mask: Optional[torch.Tensor] = None):

        pad = (self.win_length - self.hop_length) // 2
        assert spec.dim() == 3
        B, N, T = spec.shape
        if mask is not None:
            assert mask.shape[0] == B and mask.shape[1] == T

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]
        if mask is not None:
            mask_exp = mask.unsqueeze(1)  # (B, 1, T)
            ifft = ifft * mask_exp

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]
        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        # mask: (B, T) -> (B, 1, T)
        mask_exp = mask.unsqueeze(1).to(dtype=y.dtype)
        ones_patch = torch.ones(
            B, self.win_length, T, device=spec.device,
            dtype=y.dtype) * mask_exp
        folded_mask = torch.nn.functional.fold(
            ones_patch,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length))[:, 0, 0, pad:-pad]
        effective_mask = (folded_mask > 0).to(dtype=torch.float32)
        return y, effective_mask


class ISTFTHead(torch.nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        out_dim = config.n_fft + 2
        self.out = torch.nn.Linear(config.output_size, out_dim)
        self.istft = ISTFT(
            n_fft=config.n_fft,
            hop_length=config.hop_size,
            win_length=config.n_fft,
        )

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.
            mask (Tensor): (B,L)

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)
        audio, mask = self.istft(S, mask)
        return audio, mask
