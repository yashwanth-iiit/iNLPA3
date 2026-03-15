import torch
import torch.nn as nn


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
        # x: (batch, input_dim)
        # h: (batch, hidden_dim)
        return torch.tanh(self.W_ih(x) + self.W_hh(h))


# ── LSTM Cell (from scratch) ──────────────────────────────────────────────────
class LSTMCell(nn.Module):
    """
    LSTM cell.
    i = sigmoid(W_ii*x + W_hi*h + b_i)   input gate
    f = sigmoid(W_if*x + W_hf*h + b_f)   forget gate
    g = tanh   (W_ig*x + W_hg*h + b_g)   cell gate
    o = sigmoid(W_io*x + W_ho*h + b_o)   output gate
    c_t = f * c_{t-1} + i * g
    h_t = o * tanh(c_t)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # All 4 gates in one linear for efficiency
        self.W_ih = nn.Linear(input_dim, 4 * hidden_dim)
        self.W_hh = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, input_dim)
        # h: (batch, hidden_dim)
        # c: (batch, hidden_dim)
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
        # x: (batch, seq_len, input_dim)
        # h: (num_layers, batch, hidden_dim) or None
        batch, seq_len, _ = x.shape
        if h is None:
            h = torch.zeros(self.num_layers, batch, self.hidden_dim, device=x.device)

        hs = list(h.unbind(0))  # one per layer
        outputs = []

        for t in range(seq_len):
            inp = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                hs[layer_idx] = cell(inp, hs[layer_idx])
                inp = self.dropout(hs[layer_idx]) if layer_idx < self.num_layers - 1 else hs[layer_idx]
            outputs.append(inp)

        all_outputs = torch.stack(outputs, dim=1)          # (batch, seq_len, hidden)
        final_h = torch.stack(hs, dim=0)                   # (num_layers, batch, hidden)
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
        # x: (batch, seq_len, input_dim)
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


# ── Encoder ───────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Encodes cipher digit sequence into context vector.
    Supports both RNN and LSTM.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        cell_type: str,  # "rnn" or "lstm"
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
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, emb_dim)
        outputs, hidden = self.rnn(embedded)
        # outputs: (batch, src_len, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim) for RNN
        #         ((num_layers, batch, hidden_dim), (num_layers, batch, hidden_dim)) for LSTM
        return outputs, hidden


# ── Decoder ───────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    """
    Decodes hidden state into plain character sequence.
    Supports both RNN and LSTM.
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
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        if cell_type == "rnn":
            self.rnn = RNNLayer(embedding_dim, hidden_dim, num_layers, dropout)
        else:
            self.rnn = LSTMLayer(embedding_dim, hidden_dim, num_layers, dropout)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt: torch.Tensor, hidden):
        # tgt: (batch,) — single token at each step
        # hidden: encoder hidden state
        tgt = tgt.unsqueeze(1)                             # (batch, 1)
        embedded = self.dropout(self.embedding(tgt))       # (batch, 1, emb_dim)
        output, hidden = self.rnn(embedded, hidden)        # (batch, 1, hidden_dim)
        prediction = self.fc_out(output.squeeze(1))        # (batch, vocab_size)
        return prediction, hidden


# ── Seq2Seq ───────────────────────────────────────────────────────────────────
class Seq2Seq(nn.Module):
    """
    Full encoder-decoder seq2seq model for cipher decryption.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        # src: (batch, src_len)
        # tgt: (batch, tgt_len)
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=src.device)

        _, hidden = self.encoder(src)

        # First decoder input is <SOS>
        dec_input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(dec_input, hidden)
            outputs[:, t, :] = output
            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = tgt[:, t]
            else:
                dec_input = output.argmax(dim=-1)

        return outputs

    def decode_greedy(
        self, src: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int
    ) -> list:
        """Greedy decoding for inference — no teacher forcing."""
        self.eval()
        with torch.no_grad():
            _, hidden = self.encoder(src)
            dec_input = torch.full((src.shape[0],), sos_idx, dtype=torch.long, device=src.device)
            predictions = []

            for _ in range(max_len):
                output, hidden = self.decoder(dec_input, hidden)
                top1 = output.argmax(dim=-1)
                predictions.append(top1)
                dec_input = top1

                # Stop if all sequences in batch have produced EOS
                if (top1 == eos_idx).all():
                    break

        return torch.stack(predictions, dim=1)  # (batch, decoded_len)


# ── Model factory ─────────────────────────────────────────────────────────────
def build_seq2seq(cfg: dict, cipher_vocab_size: int, plain_vocab_size: int) -> Seq2Seq:
    cell_type = cfg["model"]["type"]  # "rnn" or "lstm"
    emb_dim = cfg["model"]["embedding_dim"]
    hid_dim = cfg["model"]["hidden_dim"]
    n_layers = cfg["model"]["num_layers"]
    dropout = cfg["model"]["dropout"]

    encoder = Encoder(cipher_vocab_size, emb_dim, hid_dim, n_layers, dropout, cell_type)
    decoder = Decoder(plain_vocab_size, emb_dim, hid_dim, n_layers, dropout, cell_type)
    return Seq2Seq(encoder, decoder)
