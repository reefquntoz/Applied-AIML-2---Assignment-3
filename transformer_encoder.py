"""Shared Transformer encoder architecture for the transformer assignment.

Provides scaled_dot_product_attention() and the MultiHeadAttention, PositionalEncoding,
TransformerEncoderBlock, and TransformerEncoder modules used across all notebook versions.
"""

import math

import torch
import torch.nn.functional as F


# make the scaled dot product attention
def scaled_dot_product_attention(query, key, value, mask=None):
    """Compute scaled dot-product attention.

    Args:
        query (torch.Tensor): Query tensor of shape (..., seq_len_q, d_k).
        key (torch.Tensor): Key tensor of shape (..., seq_len_k, d_k).
        value (torch.Tensor): Value tensor of shape (..., seq_len_k, d_v).
        mask (torch.Tensor, optional): Mask broadcastable to the attention score shape; positions where mask == 0 are set to -1e9 before the softmax. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The attention output of shape
            (..., seq_len_q, d_v) and the attention weights of shape
            (..., seq_len_q, seq_len_k).
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # attention formula. matrix multiplication (dot product)
    if mask is not None: # for decoder. leave it for further project
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1) # converts logits into probability distribution
    output = torch.matmul(attn_weights, value) # matrix multiplication
    return output, attn_weights


class MultiHeadAttention(torch.nn.Module):
    """Multi-head self-attention module.

    Acts as self-attention throughout this notebook. Query, key, and value all come from the same input x.
    Each token's representation is updated by attending to every other token in the sequence, including itself.

    Attributes:
        num_heads (int): Number of attention heads.
        d_model (int): Dimensionality of the input/output representations.
        depth (int): Dimensionality of each head (d_model / num_heads).
        wq (torch.nn.Linear): Linear projection for queries.
        wk (torch.nn.Linear): Linear projection for keys.
        wv (torch.nn.Linear): Linear projection for values.
        dense (torch.nn.Linear): Output projection after concatenating heads.
    """
    def __init__(self, num_heads, d_model):
        """Initialise the multi-head attention layer.

        Args:
            num_heads (int): Number of attention heads.
            d_model (int): Dimensionality of the input/output representations.
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads # floor division, round down to the nearest integer.

        self.wq = torch.nn.Linear(d_model, d_model, )
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)
        self.dense = torch.nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """Reshape the last dimension into (num_heads, depth) and transpose.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, d_model).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_heads, seq_len, depth).
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth) # reshape PyTorch tensor x to separate feature dimension into individual attention heads
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        """Apply multi-head attention to the query, key, and value tensors.

        Args:
            query (torch.Tensor): Shape (batch_size, seq_len, d_model).
            key (torch.Tensor): Shape (batch_size, seq_len, d_model).
            value (torch.Tensor): Shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Attention output of shape
                (batch_size, seq_len, d_model) and the per-head attention
                weights.
        """
        batch_size = query.size(0)
        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        scaled_attention, attn_weights = scaled_dot_product_attention(query, key, value, mask) # execute the matmul to get the attention weight
        scaled_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # Transposes back to (batch, seq_len, num_heads, depth)
        # Contiguous: to follow the exact order as the new shape implies
        # Reshapes (.view) to merge heads back into (batch, seq_len, d_model)
        # — this is the "concatenate all heads" step from the original Transformer paper.

        output = self.dense(scaled_attention)
        return output, attn_weights


# build the positional encoding
class PositionalEncoding(torch.nn.Module):
    """Fixed sinusoidal positional encoding.

    Builds a fixed (non-learned) matrix pe of shape (max_len, d_model) using the sinusoidal formula from "Attention Is All You Need".
    Each position gets a unique pattern of sine/cosine values across the d_model dimensions, letting the model infer relative and absolute token positions.

    Attributes:
        pe (torch.Tensor): Registered buffer of shape (1, max_len, d_model) holding the precomputed positional encodings.
    """
    def __init__(self, d_model, max_len=5000):
        """Precompute the sinusoidal positional encoding table.

        Args:
            d_model (int): Dimensionality of the token embeddings.
            max_len (int, optional): Maximum sequence length supported.
                Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model) # tensor with scalar 0 values
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # creates 2D column vector. Unsqueeze adds new dimension of size 1 at index 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # uses log to stabilise to compute the power
        # the purpose of div_term is to give each dimension pair a different sinusoidal frequency.
        # stacking many different frequencies across the embedding dimensions helps the model represent absolute position and relative offset

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Each position gets a unique pattern of sine/cosine values across the d_model dimensions.
        # This lets the model infer relative and absolute positions, and generalizes reasonably to sequence lengths not seen during training (unlike a learned positional embedding table).

        pe = pe.unsqueeze(0) # shape (1, max_len, d_model) — batch-first
        self.register_buffer('pe', pe) # stores pe as part of the module's state (so it moves with .to(device) etc.) but explicitly not as a trainable parameter (no gradients).
        # register a non-trainable tensor.

    def forward(self, x):
        """Add positional encodings to the input embeddings.

        Args:
            x (torch.Tensor): Input embeddings of shape
                (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Input embeddings with positional encodings added,
                of the same shape as x.
        """
        x = x + self.pe[:, :x.size(1), :] # slice along seq_len (dim 1), broadcast over batch (dim 0)
        return x


# make the encoder block
class TransformerEncoderBlock(torch.nn.Module):
    """Single Transformer encoder block.

    Applies multi-head self-attention followed by a residual connection and layer normalisation,
    then a position-wise feed-forward network followed by another residual connection and layer normalisation.
    Residual connections let gradients flow directly through the network, makes many layers trainable.

    Attributes:
        mha (MultiHeadAttention): Multi-head self-attention sublayer.
        ffn (torch.nn.Sequential): Position-wise feed-forward network.
        layernorm1 (torch.nn.LayerNorm): Layer norm after attention sublayer.
        layernorm2 (torch.nn.LayerNorm): Layer norm after feed-forward sublayer.
        dropout (torch.nn.Dropout): Dropout applied to each sublayer's output.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """Initialise the encoder block's sublayers.

        Args:
            d_model (int): Dimensionality of the token representations.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden dimensionality of the feed-forward network.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(num_heads, d_model)
        self.ffn = torch.nn.Sequential( # feed forward network
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(), # ReLU activation in between
            torch.nn.Linear(d_ff, d_model)
        )
        # The FFN applies independently to each position/token. This is where most of the "reasoning capacity" per layer lives —
        # On the other hand, attention mixes information between tokens, the FFN transforms information within each token.

        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """Apply self-attention and the feed-forward network to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        attn_output, _ = self.mha(x, x, x, mask) # execute the multi-head attention
        out1 = self.layernorm1(x + self.dropout(attn_output)) # add residual connection & norm 1
        ffn_output = self.ffn(out1) # execute the feedforward network
        out2 = self.layernorm2(out1 + self.dropout(ffn_output)) # add residual connection & norm 2
        return out2


class TransformerEncoder(torch.nn.Module):
    """Full Transformer encoder.

    Embeds input token ids, adds positional encodings, passes them through a stack of TransformerEncoderBlock layers, mean-pools the resulting
    token representations (ignoring padding), and projects the pooled representation to two class probabilities via a classification head.

    Attributes:
        embedding (torch.nn.Embedding): Token embedding lookup table.
        pos_encoding (PositionalEncoding): Sinusoidal positional encoding module.
        enc_layers (torch.nn.ModuleList): Stack of TransformerEncoderBlock layers.
        classification (torch.nn.Linear): Output layer mapping pooled representations to 2 class logits.
        dropout (torch.nn.Dropout): Dropout applied after positional encoding.
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_length, pad_token_id=0, dropout=0.1):
        """Initialise the embedding, positional encoding, encoder stack, and classification head.

        Args:
            num_layers (int): Number of stacked encoder blocks.
            d_model (int): Dimensionality of the token representations.
            num_heads (int): Number of attention heads per encoder block.
            d_ff (int): Hidden dimensionality of each block's feed-forward network.
            input_vocab_size (int): Size of the input token vocabulary.
            max_seq_length (int): Maximum input sequence length.
            pad_token_id (int, optional): Token id used for padding, whose embedding row is fixed at zero. Defaults to 0.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(TransformerEncoder, self).__init__() # initialise the TransformerEncoder
        self.embedding = torch.nn.Embedding(input_vocab_size, d_model, padding_idx=pad_token_id) # a learned lookup table converting integer token IDs into d_model-dimensional dense vectors. padding_idx keeps the [PAD] row fixed at zero and excluded from gradient updates.
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.enc_layers = torch.nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.classification = torch.nn.Linear(in_features=d_model, out_features=2)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """Compute class probabilities for a batch of tokenised datapoints.

        Embeds the input token ids, scales by sqrt(d_model) to rebalance the relative magnitude against the positional encoding,
        adds positional encoding, applies dropout, then passes the result sequentially through every encoder block.
        The resulting token representations are mean-pooled (ignoring padding positions when a mask is given) and projected to class probabilities via the classification head.

        Args:
            x (torch.Tensor): Input token ids of shape (batch_size, seq_len).
            mask (torch.Tensor, optional): Mask of shape (batch_size, seq_len) where 1 marks a real token and 0 marks padding. Defaults to None.

        Returns:
            torch.Tensor: Class probabilities of shape (batch_size, 2).
        """

        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim) # use sqrt to rebalance the magnitude against the pe
        x = self.pos_encoding(x) # add positional encoding
        x = self.dropout(x)

        encoder_mask = None
        if mask is not None: # mask is always provided via train_utils, since inputs are padded
            encoder_mask = mask.unsqueeze(1).unsqueeze(2) # (batch, 1, 1, seq_len) for attention broadcasting


        for layer in self.enc_layers:
            x = layer(x, encoder_mask)

        if mask is not None: # mask is always provided via train_utils, since inputs are padded
            mask_expanded = mask.unsqueeze(-1).float() # (batch, seq_len, 1)
            summed = (x * mask_expanded).sum(dim=1) # sum only real (non-padding) tokens
            counted = mask_expanded.sum(dim=1).clamp(min=1e-9) # avoid divide-by-zero for safety
            # using clamp to restric all elements in a tensor to a specified minimum and maximum range

            pooled = summed / counted # mean-pool over real tokens only
        else:
            pooled = x.mean(dim=1) # fallback: mean over all tokens if no mask given

        logits = self.classification(pooled) # (batch, 2), one prediction per review
        probabilities = F.softmax(logits, dim=-1)# implement softmax activation function to produce probabilities

        return probabilities
