import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── LSTM Cell (reused from task1) ─────────────────────────────────────────────
class LSTMCell(nn.Module):
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


# ── Bidirectional LSTM for MLM ────────────────────────────────────────────────
class BiLSTMLayer(nn.Module):
    """
    Bidirectional LSTM — runs one forward LSTM and one backward LSTM,
    concatenates their outputs at each position.
    Output dim = 2 * hidden_dim
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Forward LSTM layers
        self.fwd_cells = nn.ModuleList([
            LSTMCell(input_dim if i == 0 else 2 * hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        # Backward LSTM layers
        self.bwd_cells = nn.ModuleList([
            LSTMCell(input_dim if i == 0 else 2 * hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        batch, seq_len, _ = x.shape

        # Initialize hidden states
        h_fwd = [torch.zeros(batch, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c_fwd = [torch.zeros(batch, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        h_bwd = [torch.zeros(batch, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c_bwd = [torch.zeros(batch, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]

        # Forward pass
        fwd_outputs = []
        inp = x
        for layer_idx in range(self.num_layers):
            layer_outputs = []
            for t in range(seq_len):
                h_fwd[layer_idx], c_fwd[layer_idx] = self.fwd_cells[layer_idx](
                    inp[:, t, :], h_fwd[layer_idx], c_fwd[layer_idx]
                )
                layer_outputs.append(h_fwd[layer_idx])
            fwd_out = torch.stack(layer_outputs, dim=1)  # (batch, seq_len, hidden)
            if layer_idx < self.num_layers - 1:
                inp = fwd_out

        # Backward pass
        inp = x
        for layer_idx in range(self.num_layers):
            layer_outputs = []
            for t in reversed(range(seq_len)):
                h_bwd[layer_idx], c_bwd[layer_idx] = self.bwd_cells[layer_idx](
                    inp[:, t, :], h_bwd[layer_idx], c_bwd[layer_idx]
                )
                layer_outputs.insert(0, h_bwd[layer_idx])
            bwd_out = torch.stack(layer_outputs, dim=1)  # (batch, seq_len, hidden)
            if layer_idx < self.num_layers - 1:
                inp = bwd_out

        # Concatenate forward and backward outputs
        output = torch.cat([fwd_out, bwd_out], dim=-1)  # (batch, seq_len, 2*hidden)
        return output


class BiLSTMForMLM(nn.Module):
    """
    Bidirectional LSTM for Masked Language Modeling.
    Reads full sequence (with masks), predicts original token at each masked position.
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


# ── SSM (State Space Model) for NWP ──────────────────────────────────────────
class SSMLayer(nn.Module):
    """
    Simplified S4-inspired State Space Model layer.

    Continuous-time SSM:
        x'(t) = A * x(t) + B * u(t)
        y(t)  = C * x(t) + D * u(t)

    Discretized with step size delta:
        x_t = A_bar * x_{t-1} + B_bar * u_t
        y_t = C * x_t + D * u_t

    Where:
        A_bar = exp(delta * A)
        B_bar = (A_bar - I) * A^{-1} * B

    We parameterize A as a diagonal matrix for efficiency.
    """

    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        # SSM parameters
        # A: diagonal — initialized with HiPPO-like values
        self.log_A_real = nn.Parameter(torch.zeros(state_dim))
        self.A_imag = nn.Parameter(torch.randn(state_dim) * 0.01)

        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(input_dim, state_dim) * 0.01)
        self.D = nn.Parameter(torch.ones(input_dim))

        # Log step size (learnable)
        self.log_delta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (batch, seq_len, input_dim)
        batch, seq_len, _ = u.shape

        # Compute discretization
        delta = F.softplus(self.log_delta)              # (input_dim,)

        # A is diagonal complex: A = -exp(log_A_real) + i * A_imag
        A_real = -torch.exp(self.log_A_real)            # (state_dim,)

        # ZOH discretization for diagonal A:
        # A_bar = exp(delta * A)
        # Using real part only for stability
        delta_A = torch.einsum('d,n->dn', delta, A_real)  # (input_dim, state_dim)
        A_bar = torch.exp(delta_A)                         # (input_dim, state_dim)

        # B_bar = delta * B (simplified)
        delta_B = torch.einsum('d,nd->nd', delta, self.B)  # (state_dim, input_dim) → scaled

        # Scan over sequence
        x = torch.zeros(batch, self.state_dim, device=u.device)
        outputs = []

        for t in range(seq_len):
            u_t = u[:, t, :]                            # (batch, input_dim)
            # x = A_bar * x + B_bar * u_t
            # Using mean of A_bar over input dims for state update
            A_bar_mean = A_bar.mean(0)                  # (state_dim,)
            B_u = torch.einsum('nd,bd->bn', delta_B, u_t)  # (batch, state_dim)
            x = x * A_bar_mean.unsqueeze(0) + B_u
            # y = C * x + D * u_t
            y = torch.einsum('dn,bn->bd', self.C, x) + self.D * u_t
            outputs.append(y)

        return torch.stack(outputs, dim=1)              # (batch, seq_len, input_dim)


class SSMBlock(nn.Module):
    """SSM layer with residual connection, layer norm and dropout."""

    def __init__(self, hidden_dim: int, state_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ssm = SSMLayer(hidden_dim, state_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class SSMForNWP(nn.Module):
    """
    Stacked SSM blocks for Next Word/Character Prediction.
    Causal model — predicts token at position t+1 given 0..t.
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
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SSMBlock(hidden_dim, state_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))      # (batch, seq_len, emb_dim)
        hidden = self.input_proj(embedded)              # (batch, seq_len, hidden_dim)

        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.norm(hidden)
        logits = self.fc_out(hidden)                    # (batch, seq_len, vocab_size)
        return logits


# ── Model factories ───────────────────────────────────────────────────────────
def build_bilstm(cfg: dict, vocab_size: int) -> BiLSTMForMLM:
    return BiLSTMForMLM(
        vocab_size=vocab_size,
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
    )


def build_ssm(cfg: dict, vocab_size: int) -> SSMForNWP:
    return SSMForNWP(
        vocab_size=vocab_size,
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        state_dim=cfg["model"]["state_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
    )
