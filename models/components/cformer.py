import torch
from torch import nn
from torch.nn import functional as F
from models.components.attention import MultiHeadSelfAttention, FlashMultiHeadSelfAttention


from models.components.rope import RotaryPositionEmbedding

import torch.utils.checkpoint as checkpoint

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    In the cited paper beta is set to 1 and is not learnable;
    but by the Swish definition it is learnable parameter otherwise
    it is SiLU activation function (https://paperswithcode.com/method/swish)
    """
    def __init__(self, size_in, size_out, beta_is_learnable=True, bias=True):
        """
        Args:
            size_in: input embedding dimension
            size_out: output embedding dimension
            beta_is_learnable: whether beta is learnable or set to 1, learnable by default
            bias: whether use bias term, enabled by default
        """
        super().__init__()
        self.linear = nn.Linear(size_in, size_out, bias=bias)
        self.linear_gate = nn.Linear(size_in, size_out, bias=bias)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=beta_is_learnable)  

    def forward(self, x):
        linear_out = self.linear(x)
        swish_out = linear_out * torch.sigmoid(self.beta * linear_out)
        return swish_out * self.linear_gate(x)


class CFormer(nn.Module):
    def __init__(self, embed_dim, pair_dim, num_blocks, num_heads, use_rot_emb=True, attn_qkv_bias=False, transition_dropout=0.0, attention_dropout=0.0, residual_dropout=0.0, transition_factor=4, use_flash_attn=False):
        super().__init__()

        self.use_flash_attn = use_flash_attn

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, pair_dim, num_heads, use_rot_emb, attn_qkv_bias, transition_dropout, attention_dropout, residual_dropout, transition_factor, use_flash_attn) for _ in range(num_blocks)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.pair_final_layer_norm = nn.LayerNorm(pair_dim)

    def forward(self, x, struct_embed, key_padding_mask=None, need_attn_weights=False, attn_mask=None):
        attn_weights = None
        if need_attn_weights:
            attn_weights = []

        for block in self.blocks:
        #     x, struct_embed, attn = checkpoint.checkpoint(
        #         block, 
        #         x,
        #         struct_embed,
        #         key_padding_mask,
        #         need_attn_weights,
        #         use_reentrant=False
        #         )
            x, struct_embed, attn = block(x, struct_embed, key_padding_mask, attn_mask)
            if need_attn_weights:
                attn_weights.append(attn)

        x = self.final_layer_norm(x)
        struct_embed = self.pair_final_layer_norm(struct_embed)
        return x, struct_embed, attn_weights

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, pair_dim, num_heads, use_rot_emb=True, attn_qkv_bias=False, transition_dropout=0.0, attention_dropout=0.0, residual_dropout=0.0, transition_factor=4, use_flash_attn=False):
        super().__init__()
        
        self.use_flash_attn = use_flash_attn

        if use_flash_attn:
            self.mh_attn = FlashMultiHeadSelfAttention(embed_dim, num_heads, attention_dropout, causal=False, use_rot_emb=use_rot_emb, bias=attn_qkv_bias)
        else:
            self.mh_attn = MultiHeadSelfAttention(embed_dim, pair_dim, num_heads, attention_dropout, use_rot_emb, attn_qkv_bias)
        
        self.attn_layer_norm = nn.LayerNorm(embed_dim)

        self.transition = nn.Sequential(
                SwiGLU(embed_dim, int(2 / 3 * transition_factor * embed_dim), beta_is_learnable=True, bias=True),
                nn.Dropout(p=transition_dropout),
                nn.Linear(int(2 / 3 * transition_factor * embed_dim), embed_dim, bias=True),
        )
        
        
        # self.transition_struct = nn.Sequential(
        #         SwiGLU(embed_dim, int(2 / 3 * transition_factor * embed_dim), beta_is_learnable=True, bias=True),
        #         nn.Dropout(p=transition_dropout),
        #         nn.Linear(int(2 / 3 * transition_factor * embed_dim), embed_dim, bias=True),
        # )
        
        self.out_layer_norm = nn.LayerNorm(embed_dim)
        self.pair_layer_norm = nn.LayerNorm(pair_dim)
        
        self.residual_dropout_1 = nn.Dropout(p=residual_dropout)
        self.residual_dropout_2 = nn.Dropout(p=residual_dropout)

    def forward(self, x, struct_embed, key_padding_mask=None, attn_mask=None):
        x = self.attn_layer_norm(x)
        # if self.use_flash_attn:
        #     mh_out, attn = self.mh_attn(x, key_padding_mask=key_padding_mask, return_attn_probs=need_attn_weights)
        # else:
        # Temporarily unable flash_attn
        mh_out, struct_out, attn = self.mh_attn(x, struct_embed, attn_mask, key_pad_mask=key_padding_mask)
        x = x + self.residual_dropout_1(mh_out)
        struct_embed = struct_embed + self.residual_dropout_1(struct_out)
        residual = x
        struct_residual = struct_embed
        x = self.out_layer_norm(x)
        struct_embed = self.pair_layer_norm(struct_embed)
        x = residual + self.residual_dropout_2(self.transition(x))
        struct_embed = struct_residual + self.residual_dropout_2(struct_embed)
        return x, struct_embed, attn