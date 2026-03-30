import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RNN Cell (from scratch) ───────────────────────────────────────────────────
class RNNCell(nn.Module):
    """
    Vanilla RNN cell.
    h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.W_ih = nn.Linear(input_dim, hidden_dim)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.W_ih(x) + self.W_hh(h))


# ── LSTM Cell (from scratch) ──────────────────────────────────────────────────
class LSTMCell(nn.Module):
    """
    LSTM cell with all 4 gates computed in one fused linear.
    i = sigmoid(W_ii*x + W_hi*h + b_i)   input gate
    f = sigmoid(W_if*x + W_hf*h + b_f)   forget gate
    g = tanh   (W_ig*x + W_hg*h + b_g)   cell gate
    o = sigmoid(W_io*x + W_ho*h + b_o)   output gate
    c_t = f * c_{t-1} + i * g
    h_t = o * tanh(c_t)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.W_ih = nn.Linear(input_dim, 4 * hidden_dim)
        self.W_hh = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.W_ih(x) + self.W_hh(h)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


# ── Multi-layer RNN ───────────────────────────────────────────────────────────
class RNNLayer(nn.Module):
    """Stack of RNN cells with dropout between layers."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [RNNCell(input_dim if i == 0 else hidden_dim, hidden_dim)
             for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        if h is None:
            h = torch.zeros(self.num_layers, batch, self.hidden_dim, device=x.device)

        hs = list(h.unbind(0))
        outputs = []

        for t in range(seq_len):
            inp = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                hs[layer_idx] = cell(inp, hs[layer_idx])
                inp = self.dropout(hs[layer_idx]) if layer_idx < self.num_layers - 1 else hs[layer_idx]
            outputs.append(inp)

        all_outputs = torch.stack(outputs, dim=1)
        final_h = torch.stack(hs, dim=0)
        return all_outputs, final_h


# ── Multi-layer LSTM ──────────────────────────────────────────────────────────
class LSTMLayer(nn.Module):
    """Stack of LSTM cells with dropout between layers."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
             for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = x.shape
        if state is None:
            h = torch.zeros(self.num_layers, batch, self.hidden_dim, device=x.device)
            c = torch.zeros(self.num_layers, batch, self.hidden_dim, device=x.device)
        else:
            h, c = state

        hs = list(h.unbind(0))
        cs = list(c.unbind(0))
        outputs = []

        for t in range(seq_len):
            inp = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                hs[layer_idx], cs[layer_idx] = cell(inp, hs[layer_idx], cs[layer_idx])
                inp = self.dropout(hs[layer_idx]) if layer_idx < self.num_layers - 1 else hs[layer_idx]
            outputs.append(inp)

        all_outputs = torch.stack(outputs, dim=1)
        final_h = torch.stack(hs, dim=0)
        final_c = torch.stack(cs, dim=0)
        return all_outputs, (final_h, final_c)


# ── Attention ─────────────────────────────────────────────────────────────────
class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention.
    score(h_t, h_s) = v^T * tanh(W_q * h_t + W_k * h_s)
    context = sum(softmax(scores) * encoder_outputs)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query  = self.W_q(decoder_hidden).unsqueeze(1)
        keys   = self.W_k(encoder_outputs)
        scores = self.v(torch.tanh(query + keys)).squeeze(-1)

        if src_mask is not None:
            scores = scores.masked_fill(src_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


# ── Encoder ───────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """Encodes cipher digit sequence. Returns ALL hidden states for attention."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        cell_type: str,
    ):
        super().__init__()
        self.cell_type = cell_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        if cell_type == "rnn":
            self.rnn = RNNLayer(embedding_dim, hidden_dim, num_layers, dropout)
        else:
            self.rnn = LSTMLayer(embedding_dim, hidden_dim, num_layers, dropout)

    def forward(self, src: torch.Tensor):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


# ── Decoder with Attention ────────────────────────────────────────────────────
class AttentionDecoder(nn.Module):
    """Decoder with Bahdanau attention over encoder outputs."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        cell_type: str,
    ):
        super().__init__()
        self.cell_type = cell_type
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_dim)

        rnn_input_dim = embedding_dim + hidden_dim

        if cell_type == "rnn":
            self.rnn = RNNLayer(rnn_input_dim, hidden_dim, num_layers, dropout)
        else:
            self.rnn = LSTMLayer(rnn_input_dim, hidden_dim, num_layers, dropout)

        self.fc_out = nn.Linear(hidden_dim + hidden_dim, vocab_size)

    def forward(
        self,
        tgt: torch.Tensor,
        hidden,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ):
        tgt = tgt.unsqueeze(1)
        embedded = self.dropout(self.embedding(tgt))

        if self.cell_type == "rnn":
            top_hidden = hidden[-1]
        else:
            top_hidden = hidden[0][-1]

        context, attn_weights = self.attention(top_hidden, encoder_outputs, src_mask)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        prediction = self.fc_out(torch.cat([output, context], dim=-1))
        return prediction, hidden, attn_weights


# ── Seq2Seq ───────────────────────────────────────────────────────────────────
class Seq2Seq(nn.Module):
    """Encoder-decoder seq2seq with Bahdanau attention."""

    def __init__(self, encoder: Encoder, decoder: AttentionDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return src == 0

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=src.device)
        src_mask = self._make_src_mask(src)

        encoder_outputs, hidden = self.encoder(src)
        dec_input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden, _ = self.decoder(dec_input, hidden, encoder_outputs, src_mask)
            outputs[:, t, :] = output

            if torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = tgt[:, t]
            else:
                dec_input = output.argmax(dim=-1)

        return outputs

    def decode_greedy(
        self, src: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy decoding for inference.
        Returns:
            predictions:  (batch, decoded_len) token ids
            confidences:  (batch, decoded_len) softmax probability of chosen token
        """
        self.eval()
        with torch.no_grad():
            src_mask = self._make_src_mask(src)
            encoder_outputs, hidden = self.encoder(src)
            dec_input = torch.full(
                (src.shape[0],), sos_idx, dtype=torch.long, device=src.device
            )
            predictions = []
            confidences = []

            for _ in range(max_len):
                output, hidden, _ = self.decoder(
                    dec_input, hidden, encoder_outputs, src_mask
                )
                probs = torch.softmax(output, dim=-1)       # (batch, vocab_size)
                top1  = probs.argmax(dim=-1)                # (batch,)
                conf  = probs.max(dim=-1).values            # (batch,) confidence score

                predictions.append(top1)
                confidences.append(conf)
                dec_input = top1

                if (top1 == eos_idx).all():
                    break

        # (batch, decoded_len)
        return torch.stack(predictions, dim=1), torch.stack(confidences, dim=1)


# ── Model factory ─────────────────────────────────────────────────────────────
def build_seq2seq(cfg: dict, cipher_vocab_size: int, plain_vocab_size: int) -> Seq2Seq:
    cell_type = cfg["model"]["type"]
    emb_dim   = cfg["model"]["embedding_dim"]
    hid_dim   = cfg["model"]["hidden_dim"]
    n_layers  = cfg["model"]["num_layers"]
    dropout   = cfg["model"]["dropout"]

    encoder = Encoder(cipher_vocab_size, emb_dim, hid_dim, n_layers, dropout, cell_type)
    decoder = AttentionDecoder(plain_vocab_size, emb_dim, hid_dim, n_layers, dropout, cell_type)
    return Seq2Seq(encoder, decoder)
