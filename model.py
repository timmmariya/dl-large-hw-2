import torch
import torch.nn as nn
import torch.nn as nn
from torch.nn import Transformer
from torch import Tensor
import math


# Туториал: https://pytorch.org/tutorials/beginner/translation_transformer.html

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # here should be a tensor of size (1, max_len, embed_dim), dummy dimension is needed for proper addition
        pe = torch.zeros(1, max_len, embed_dim)

        # чтобы 10 тысяч в степень не возводить, сначала прологарифмируем, потом возьмем экспоненту от результата
        dividing_part = torch.exp((-1) * (math.log(10000.0) / embed_dim) * torch.arange(0, embed_dim, 2))
        pe[:, :, 0::2] = torch.sin(dividing_part * torch.arange(max_len).unsqueeze(1))
        pe[:, :, 1::2] = torch.cos(dividing_part * torch.arange(max_len).unsqueeze(1))

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):

        needed_len = x.shape[1]
        x = x + self.pe[:, :needed_len]

        return x


class TranslationModel(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        """
        Creates a standard Transformer encoder-decoder model.
        :param num_encoder_layers: Number of encoder layers
        :param num_decoder_layers: Number of decoder layers
        :param emb_size: Size of intermediate vector representations for each token
        :param dim_feedforward: Size of intermediate representations in FFN layers
        :param n_head: Number of attention heads for each layer
        :param src_vocab_size: Number of tokens in the source language vocabulary
        :param tgt_vocab_size: Number of tokens in the target language vocabulary
        :param dropout_prob: Dropout probability throughout the model
        """

        super(TranslationModel, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.fc = nn.Linear(emb_size, tgt_vocab_size)

        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size)

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor):
        """
        Given tokens from a batch of source and target sentences, predict
        logits for next tokens in target sentences.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        answer = self.transformer(src_emb, tgt_emb,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask)
        return self.fc(answer)
