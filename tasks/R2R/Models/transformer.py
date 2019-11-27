
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.R2R.utils import sinusoid_encoding_table


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed forward layer
    """

    def __init__(self, d_model=512, d_ff=2048, dropout=.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        pwff = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
        pwff = self.dropout(pwff)
        out = self.layer_norm(input + pwff)
        return out


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights(gain=1.0)

    def init_weights(self, gain=1.0):
        nn.init.xavier_normal_(self.fc_q.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_k.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_v.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_o.weight, gain=gain)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            # TODO a bit different from Herdade et al. 2019
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, 0)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer with Dropout and Layer Normalization.
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(MultiHeadAttention, self).__init__()

        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.attention(queries, keys, values, attention_mask, attention_weights)
        att = self.dropout(att)
        return self.layer_norm(queries + att)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(EncoderLayer, self).__init__()
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, mask_self_att)
        enc_att = self.enc_att(self_att, enc_output, enc_output, mask_enc_att)
        ff = self.pwff(enc_att)
        return ff


class BaseEncoder(nn.Module):
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(BaseEncoder, self).__init__()

        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N)])
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input, attention_mask, attention_weights=None):
        # input (b_s, seq_len, d_in)
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)

        return out, attention_mask


class InstructionEncoderOld(BaseEncoder):
    def __init__(self, N, d_in=300, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(InstructionEncoderOld, self).__init__(N, d_model, d_k, d_v, h, d_ff, dropout)
        self.fc = nn.Linear(d_in, self.d_model, bias=True)

    def forward(self, input, attention_mask, attention_weights=None):
        # input (b_s, seq_len, d_in)
        data, mask = input

        out = F.relu(self.fc(data))
        out = self.dropout(out)
        out = self.layer_norm(out)

        pe = sinusoid_encoding_table(out.shape[1], out.shape[2])
        pe = pe.expand(out.shape[0], pe.shape[0], pe.shape[1]).cuda()
        out = out + pe.masked_fill(mask, 0)

        out, _ = super(InstructionEncoderOld, self).forward(out, attention_mask, attention_weights=attention_weights)
        return out


class ImagePlainEncoder(BaseEncoder):
    def __init__(self, N, d_in=300, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(ImagePlainEncoder, self).__init__(N, d_model, d_k, d_v, h, d_ff, dropout)
        self.fc = nn.Linear(d_in, self.d_model)

    def forward(self, input, attention_mask, attention_weights=None):
        # input (b_s, seq_len, d_in)
        data = input

        out = F.relu(self.fc(data))
        out = self.dropout(out)
        out = self.layer_norm(out)

        out, _ = super(ImagePlainEncoder, self).forward(out, attention_mask, attention_weights)
        return out


class Decoder(nn.Module):
    def __init__(self, N, in_features=2051, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N)])
        self.fc = nn.Linear(in_features, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input, enc_output, self_att_mask, enc_att_mask):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)

        for l in self.layers:
            out = l(out, enc_output, self_att_mask, enc_att_mask)

        return out


class ActionDecoder(nn.Module):
    def __init__(self, N, in_features=7, n_output=6, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(ActionDecoder, self).__init__()
        self.d_model = d_model

        self.layers_w = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N)])
        self.layers_i = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N)])

        self.fc_in = nn.Linear(in_features, d_model, bias=False)
        self.fc_out = nn.Linear(d_model*2, n_output)

    def forward(self, input, w_t, i_t, enc_att_mask_w, enc_att_mask_i):
        seq_len = input.shape[1]

        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        input = self.fc_in(input)

        pe = sinusoid_encoding_table(input.shape[1], input.shape[2])
        pe = pe.expand(input.shape[0], pe.shape[0], pe.shape[1]).cuda()

        out_i = input + pe
        out_w = input + pe

        for i, l in enumerate(self.layers_w):
            out_w = l(out_w, w_t, mask_self_attention, enc_att_mask_w)

        for i, l in enumerate(self.layers_i):
            out_i = l(out_i, i_t, mask_self_attention, enc_att_mask_i)

        out = torch.cat((out_w, out_i), -1)

        last_out = out[:, -1, :]
        preds = self.fc_out(last_out)

        probs = F.softmax(preds, dim=-1)
        return preds, probs


class ActionDecoderHL(nn.Module):
    def __init__(self, N, in_features=2048, d_embedding=512, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, action_dropout=0.1):
        super(ActionDecoderHL, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.layers_w = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N)])
        self.layers_i = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N)])

        self.fc_in = nn.Linear(in_features, d_model, bias=False)

        self.fc_ctx = nn.Linear(d_model * 2, d_embedding, bias=False)
        self.fc_v = nn.Linear(2051, d_embedding, bias=False)

        self.drop_ctx = nn.Dropout(p=action_dropout)
        self.drop_vtx = nn.Dropout(p=action_dropout)
        self.fc_i = nn.Linear(d_model, d_embedding)

    def forward(self, input, w_t, i_t, v_t, enc_att_mask_w, enc_att_mask_i):
        seq_len = input.shape[1]

        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        input = F.relu(self.fc_in(input))
        input = self.dropout(input)
        input = self.layer_norm(input)

        pe = sinusoid_encoding_table(input.shape[1], input.shape[2])
        pe = pe.expand(input.shape[0], pe.shape[0], pe.shape[1]).cuda()

        out_i = input + pe
        out_w = input + pe

        for i, l in enumerate(self.layers_w):
            out_w = l(out_w, w_t, mask_self_attention, enc_att_mask_w)

        for i, l in enumerate(self.layers_i):
            out_i = l(out_i, i_t, mask_self_attention, enc_att_mask_i)

        out = torch.cat((out_w, out_i), -1)
        last_out = out[:, -1, :]

        b, _, d = v_t.shape

        v_t = torch.cat((v_t, torch.zeros(b, 1, d).cuda()), 1)  # includes stop action
        vtx = self.drop_vtx(F.relu(self.fc_v(v_t)))  # shape (B, 36+1, d_embedding)
        ctx = self.drop_ctx(self.fc_ctx(last_out))

        predictions = torch.bmm(vtx, ctx.unsqueeze(dim=-1))
        return predictions.squeeze(dim=-1), ctx
