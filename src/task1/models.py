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
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)  # query (decoder)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)  # key   (encoder)
        self.v   = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,   # (batch, hidden_dim) — top layer hidden
        encoder_outputs: torch.Tensor,  # (batch, src_len, hidden_dim)
        src_mask: torch.Tensor | None = None,  # (batch, src_len) — True where PAD
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # decoder_hidden: (batch, hidden) → (batch, 1, hidden)
        query = self.W_q(decoder_hidden).unsqueeze(1)
        # encoder_outputs: (batch, src_len, hidden)
        keys  = self.W_k(encoder_outputs)

        # scores: (batch, src_len, 1) → (batch, src_len)
        scores = self.v(torch.tanh(query + keys)).squeeze(-1)

        # Mask padding positions with -inf before softmax
        if src_mask is not None:
            scores = scores.masked_fill(src_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)              # (batch, src_len)

        # context: (batch, 1, src_len) x (batch, src_len, hidden) → (batch, hidden)
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
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # outputs: (batch, src_len, hidden_dim) ← used by attention
        # hidden: final hidden state(s)
        return outputs, hidden


# ── Decoder with Attention ────────────────────────────────────────────────────
class AttentionDecoder(nn.Module):
    """
    Decoder that uses Bahdanau attention over encoder outputs.
    At each step:
      1. Compute attention context from encoder outputs
      2. Concatenate [embedding, context] as RNN/LSTM input
      3. Project RNN output to vocab
    """

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

        # Input to RNN = embedding + context vector
        rnn_input_dim = embedding_dim + hidden_dim

        if cell_type == "rnn":
            self.rnn = RNNLayer(rnn_input_dim, hidden_dim, num_layers, dropout)
        else:
            self.rnn = LSTMLayer(rnn_input_dim, hidden_dim, num_layers, dropout)

        # Project [rnn_output, context] → vocab
        self.fc_out = nn.Linear(hidden_dim + hidden_dim, vocab_size)

    def forward(
        self,
        tgt: torch.Tensor,           # (batch,) single token
        hidden,                       # decoder hidden state
        encoder_outputs: torch.Tensor,  # (batch, src_len, hidden)
        src_mask: torch.Tensor | None = None,
    ):
        tgt = tgt.unsqueeze(1)                              # (batch, 1)
        embedded = self.dropout(self.embedding(tgt))        # (batch, 1, emb_dim)

        # Get top-layer hidden for attention query
        if self.cell_type == "rnn":
            top_hidden = hidden[-1]                         # (batch, hidden)
        else:
            top_hidden = hidden[0][-1]                      # (batch, hidden)

        context, attn_weights = self.attention(
            top_hidden, encoder_outputs, src_mask
        )                                                   # (batch, hidden)

        # Concatenate embedding with context
        rnn_input = torch.cat(
            [embedded, context.unsqueeze(1)], dim=-1
        )                                                   # (batch, 1, emb+hidden)

        output, hidden = self.rnn(rnn_input, hidden)        # (batch, 1, hidden)
        output = output.squeeze(1)                          # (batch, hidden)

        # Predict from [rnn_output, context]
        prediction = self.fc_out(
            torch.cat([output, context], dim=-1)
        )                                                   # (batch, vocab_size)

        return prediction, hidden, attn_weights


# ── Seq2Seq ───────────────────────────────────────────────────────────────────
class Seq2Seq(nn.Module):
    """Encoder-decoder seq2seq with Bahdanau attention."""

    def __init__(self, encoder: Encoder, decoder: AttentionDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """True where src == PAD — these positions are masked in attention."""
        return src == 0  # (batch, src_len)

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
        dec_input = tgt[:, 0]  # <SOS>

        for t in range(1, tgt_len):
            output, hidden, _ = self.decoder(
                dec_input, hidden, encoder_outputs, src_mask
            )
            outputs[:, t, :] = output

            if torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = tgt[:, t]
            else:
                dec_input = output.argmax(dim=-1)

        return outputs

    def decode_greedy(
        self, src: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int
    ) -> torch.Tensor:
        """Greedy decoding for inference."""
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
                top1 = output.argmax(dim=-1)
                predictions.append(top1)
                confidences.append(torch.softmax(output, dim=-1).max(dim=-1).values)
                dec_input = top1

                if (top1 == eos_idx).all():
                    break

        return torch.stack(predictions, dim=1), torch.stack(confidences, dim=1)  # (batch, decoded_len), (batch, decoded_len)


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
