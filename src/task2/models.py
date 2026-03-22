import torch
import torch.nn as nn
import torch.nn.functional as F


# ── LSTM Cell (from scratch) ──────────────────────────────────────────────────
class LSTMCell(nn.Module):
    """Single LSTM cell with all 4 gates."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.W_ih = nn.Linear(input_dim, 4 * hidden_dim)
        self.W_hh = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)

    def forward(self, x, h, c):
        gates = self.W_ih(x) + self.W_hh(h)
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_t = f * c + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


# ── Bidirectional LSTM ────────────────────────────────────────────────────────
class BiLSTMLayer(nn.Module):
    """
    Bidirectional LSTM from scratch.
    Runs a forward LSTM and a backward LSTM independently,
    then concatenates their outputs: output_dim = 2 * hidden_dim.

    For multi-layer BiLSTM, each layer takes the concatenated
    output of the previous layer as input.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.fwd_cells = nn.ModuleList()
        self.bwd_cells = nn.ModuleList()

        for i in range(num_layers):
            # First layer takes input_dim, subsequent layers take 2*hidden_dim
            in_dim = input_dim if i == 0 else 2 * hidden_dim
            self.fwd_cells.append(LSTMCell(in_dim, hidden_dim))
            self.bwd_cells.append(LSTMCell(in_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        batch, seq_len, _ = x.shape
        current_input = x

        for layer_idx in range(self.num_layers):
            # ── Forward pass ──────────────────────────────────────────────────
            h_f = torch.zeros(batch, self.hidden_dim, device=x.device)
            c_f = torch.zeros(batch, self.hidden_dim, device=x.device)
            fwd_out = []
            for t in range(seq_len):
                h_f, c_f = self.fwd_cells[layer_idx](current_input[:, t, :], h_f, c_f)
                fwd_out.append(h_f)
            fwd_out = torch.stack(fwd_out, dim=1)       # (batch, seq_len, hidden)

            # ── Backward pass ─────────────────────────────────────────────────
            h_b = torch.zeros(batch, self.hidden_dim, device=x.device)
            c_b = torch.zeros(batch, self.hidden_dim, device=x.device)
            bwd_out = [None] * seq_len
            for t in reversed(range(seq_len)):
                h_b, c_b = self.bwd_cells[layer_idx](current_input[:, t, :], h_b, c_b)
                bwd_out[t] = h_b
            bwd_out = torch.stack(bwd_out, dim=1)       # (batch, seq_len, hidden)

            # Concatenate and apply dropout between layers
            current_input = torch.cat([fwd_out, bwd_out], dim=-1)  # (batch, seq_len, 2*hidden)
            if layer_idx < self.num_layers - 1:
                current_input = self.dropout(current_input)

        return current_input  # (batch, seq_len, 2*hidden_dim)


class BiLSTMForMLM(nn.Module):
    """
    Bidirectional LSTM for Masked Language Modeling (MLM).
    Input : sequence with some tokens replaced by <MASK>
    Output: logits over vocab at every position
    Loss is only computed at masked positions (label == -100 elsewhere)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = BiLSTMLayer(embedding_dim, hidden_dim, num_layers, dropout)
        self.fc_out = nn.Linear(2 * hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))      # (batch, seq_len, emb_dim)
        output = self.bilstm(embedded)                  # (batch, seq_len, 2*hidden)
        logits = self.fc_out(output)                    # (batch, seq_len, vocab_size)
        return logits


# ── SSM Layer ─────────────────────────────────────────────────────────────────
class SSMLayer(nn.Module):
    """
    Simplified diagonal SSM layer inspired by S4/Mamba.

    Discretized state space model:
        x_t = A_bar * x_{t-1} + B_bar * u_t
        y_t = C * x_t + D * u_t

    A is parameterized as diagonal for efficiency.
    Delta (step size) is learned per input dimension.
    """

    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.state_dim = state_dim

        # Diagonal SSM parameters
        self.log_A = nn.Parameter(torch.zeros(state_dim))          # log(-A) for stability
        self.B     = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)
        self.C     = nn.Parameter(torch.randn(input_dim, state_dim) * 0.01)
        self.D     = nn.Parameter(torch.ones(input_dim))
        self.log_delta = nn.Parameter(torch.zeros(input_dim))      # learnable step size

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (batch, seq_len, input_dim)
        batch, seq_len, input_dim = u.shape

        # Discretize: ZOH method
        delta = F.softplus(self.log_delta)              # (input_dim,) positive
        A = -torch.exp(self.log_A)                      # (state_dim,) negative diagonal

        # A_bar = exp(delta * A): (input_dim, state_dim)
        A_bar = torch.exp(torch.outer(delta, A))        # (input_dim, state_dim)
        # Use mean across input_dim for state update (diagonal approximation)
        A_bar_state = A_bar.mean(0)                     # (state_dim,)

        # B_bar = delta * B (simplified ZOH): (state_dim, input_dim)
        B_bar = self.B * delta.unsqueeze(0)             # (state_dim, input_dim)

        # Sequential scan
        x = torch.zeros(batch, self.state_dim, device=u.device)
        outputs = []

        for t in range(seq_len):
            u_t = u[:, t, :]                                        # (batch, input_dim)
            Bu  = torch.einsum('ni,bi->bn', B_bar, u_t)             # (batch, state_dim)
            x   = x * A_bar_state.unsqueeze(0) + Bu                 # (batch, state_dim)
            y   = torch.einsum('in,bn->bi', self.C, x) + self.D * u_t  # (batch, input_dim)
            outputs.append(y)

        return torch.stack(outputs, dim=1)              # (batch, seq_len, input_dim)


class SSMBlock(nn.Module):
    """SSM with pre-norm residual connection, GELU activation and dropout."""

    def __init__(self, hidden_dim: int, state_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ssm  = SSMLayer(hidden_dim, state_dim)
        self.act  = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class SSMForNWP(nn.Module):
    """
    Stacked SSM blocks for Next Character Prediction (NWP).
    Causal: given characters 0..t, predict character at t+1.
    Input : tokens[0..n-1]
    Target: tokens[1..n]
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        state_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.blocks     = nn.ModuleList([
            SSMBlock(hidden_dim, state_dim, dropout) for _ in range(num_layers)
        ])
        self.norm    = nn.LayerNorm(hidden_dim)
        self.fc_out  = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        h = self.dropout(self.embedding(x))             # (batch, seq_len, emb_dim)
        h = self.input_proj(h)                          # (batch, seq_len, hidden_dim)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.fc_out(h)                           # (batch, seq_len, vocab_size)


# ── Model factories ───────────────────────────────────────────────────────────
def build_bilstm(cfg: dict, vocab_size: int) -> BiLSTMForMLM:
    return BiLSTMForMLM(
        vocab_size    = vocab_size,
        embedding_dim = cfg["model"]["embedding_dim"],
        hidden_dim    = cfg["model"]["hidden_dim"],
        num_layers    = cfg["model"]["num_layers"],
        dropout       = cfg["model"]["dropout"],
    )


def build_ssm(cfg: dict, vocab_size: int) -> SSMForNWP:
    return SSMForNWP(
        vocab_size    = vocab_size,
        embedding_dim = cfg["model"]["embedding_dim"],
        hidden_dim    = cfg["model"]["hidden_dim"],
        state_dim     = cfg["model"]["state_dim"],
        num_layers    = cfg["model"]["num_layers"],
        dropout       = cfg["model"]["dropout"],
    )
