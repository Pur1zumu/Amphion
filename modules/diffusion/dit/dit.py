import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @property
    def sin_cached(self):
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.40. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead."
        )
        return self._sin_cached

    @property
    def cos_cached(self):
        logger.warning_once(
            "The cos_cached attribute will be removed in 4.40. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead."
        )
        return self._cos_cached

    def forward(self, x, position_ids, seq_len=None):
        if seq_len is not None:
            logger.warning_once("The `seq_len` argument is deprecated and unused. It will be removed in v4.40.")

        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        # backwards compatibility
        self._cos_cached = cos
        self._sin_cached = sin
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MLP(nn.Module):
    r"""Multi-layer perceptron.

    Args:
        hidden_size: The number of hidden units.
        intermediate_size: The number of intermediate units.
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU("tanh")

    def forward(self, x):
        return self.down_proj(self.activation(self.up_proj(x)))


class Attention(nn.Module):
    r"""Multi-head self-attention.

    Args:
        hidden_size: The number of hidden units.
        num_attention_heads: The number of attention heads.
        attention_dropout: The dropout rate for attention weights.
        max_position_embeddings: The maximum position embeddings.
        rope_theta: The base for the rotary position embeddings.
    """

    def __init__(self, hidden_size, num_attention_heads, attention_dropout=0.0,
                    max_position_embeddings=2048, rope_theta=10000):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.head_dim = hidden_size // num_attention_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
            )


    def forward(self, hidden_states, attn_mask=None, encoder_hidden_states=None):
        r"""Applies multi-head self-attention.

        Args:
            hidden_states: The input tensor.
            attn_mask: The attention mask.
            encoder_hidden_states: The encoder hidden states. 
                                   The length of the encoder hidden states is ensured to be the same as the hidden states.
        
        Returns:
            The output tensor.
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        if encoder_hidden_states is not None:
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # convert attn_mask from [bsz, q_len] to [bsz, q_len, q_len]
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(1).expand(-1, q_len, -1)

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attn_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=8/3, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(hidden_size, num_heads=num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(hidden_size, num_heads=num_heads, **block_kwargs)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = MLP(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c, seq_c, mask=None):
        shift_msa, scale_msa, gate_msa, \
            shift_mca, scale_mca, gate_mca, \
                shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(modulate(self.norm2(x), shift_mca, scale_mca), mask, seq_c)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=72,
        hidden_size=256,
        depth=6,
        num_heads=4,
        mlp_ratio=8/3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.input_layer = nn.Linear(in_channels, hidden_size)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, c, seq_c, mask=None):
        """
        Forward pass of DiT.
        x: (N, T, D) tensor of input data
        t: (N,) tensor of diffusion timesteps
        c: (N, D) tensor of conditioning variables
        seq_c: (N, T, D) tensor of conditioning variables for each timestep
        """

        c = t + c
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x, c, seq_c, mask)
        x = self.final_layer(x, c)
        return x
