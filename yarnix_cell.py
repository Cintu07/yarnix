"""
YarnixCell v4 — Multi-Clock Phase (Cheat Code 4)

Each neuron gets a different "clock speed" via its persistence value.

Biological inspiration: The brain uses different oscillation frequencies
for different types of processing:
  - Gamma waves (40Hz)  → immediate word-by-word tracking
  - Beta waves  (20Hz)  → sentence-level structure
  - Alpha waves (10Hz)  → paragraph/topic tracking
  - Theta waves (4Hz)   → chapter-level memory / narrative arcs

Implementation: Split the hidden_size into 4 bands, each with a different
persistence value. Fast clocks decay quickly (respond to recent input),
slow clocks decay slowly (hold onto long-range context).

Band layout for hidden_size=256:
  [0:64]   → Ultra-fast  (persistence=0.50)  → character/word patterns
  [64:128] → Fast        (persistence=0.80)  → phrase/sentence structure
  [128:192]→ Slow        (persistence=0.95)  → paragraph/topic tracking
  [192:256]→ Ultra-slow  (persistence=0.999) → chapter-level memory

The readout MLP learns to weight each timescale appropriately.
"""

import torch
import torch.nn as nn
import numpy as np


class YarnixCellV4(nn.Module):
    """
    Multi-clock recurrent cell with 4 timescale bands.
    Each band has its own persistence value for the phase accumulator.
    """
    def __init__(self, input_size, hidden_size, harmonics=[1, 2, 4, 8],
                 quantization_strength=0.05,
                 clock_speeds=(0.50, 0.80, 0.95, 0.999)):
        super(YarnixCellV4, self).__init__()
        self.hidden_size = hidden_size
        self.harmonics = harmonics
        self.quantization_strength = quantization_strength
        self.two_pi = 2.0 * np.pi
        
        # Split hidden into equal bands for each clock speed
        self.n_bands = len(clock_speeds)
        self.band_size = hidden_size // self.n_bands
        assert hidden_size % self.n_bands == 0, \
            f"hidden_size ({hidden_size}) must be divisible by n_bands ({self.n_bands})"
        
        # Build persistence vector: each neuron gets its band's clock speed
        persistence_vec = []
        for speed in clock_speeds:
            persistence_vec.extend([speed] * self.band_size)
        self.register_buffer('persistence',
                             torch.tensor(persistence_vec, dtype=torch.float32))
        
        # Gate weights: input + rich hidden → 3 gates (reset, update, candidate)
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size * 3))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 3))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 3))
        
        # Phase reader: harmonics + winding → phase signal
        n_harm = len(harmonics) * hidden_size * 2
        self.phase_reader = nn.Sequential(
            nn.Linear(n_harm + hidden_size, hidden_size),
            nn.Tanh(),
        )
        
        # Winding counter projection
        self.winding_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Cross-band mixer: lets different timescales communicate
        # This is key — without it the bands would be isolated
        self.band_mixer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias)
        
        for m in self.phase_reader.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.winding_proj.weight)
        
        for m in self.band_mixer.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, h_state, local_angle, winding_count):
        """
        x:             (batch, input_size)
        h_state:       (batch, hidden_size) — rich features
        local_angle:   (batch, hidden_size) — bounded [0, 2π]
        winding_count: (batch, hidden_size) — integer rotation count (no grad)
        
        Returns: (output, new_h, new_angle, new_winding)
        """
        # === GRU-STYLE GATES ===
        gates = x @ self.weight_ih + h_state @ self.weight_hh + self.bias
        r_gate, z_gate, n_gate = gates.chunk(3, dim=-1)
        r = torch.sigmoid(r_gate)
        z = torch.sigmoid(z_gate)
        n = torch.tanh(r * n_gate)
        
        # === MULTI-CLOCK PHASE ACCUMULATION ===
        # Each neuron's persistence is different based on its band
        # Fast neurons (0.5) forget quickly → track recent chars
        # Slow neurons (0.999) remember forever → track chapter themes
        phi_shift = torch.sigmoid(z_gate) * self.two_pi
        raw_angle = local_angle * self.persistence + phi_shift
        
        # Quantization sieve
        q_grid = np.pi / 4.0
        q_snap = torch.round(raw_angle / q_grid) * q_grid
        raw_angle = raw_angle + self.quantization_strength * (q_snap - raw_angle)
        
        # === BOUNDED WRAP + WINDING COUNT ===
        new_rotations = torch.floor(raw_angle / self.two_pi)
        new_winding = winding_count + new_rotations.detach()
        new_angle = raw_angle - new_rotations * self.two_pi
        
        # === READ PHASE (all bands, all harmonics) ===
        harm_feats = []
        for h in self.harmonics:
            harm_feats.append(torch.cos(h * new_angle))
            harm_feats.append(torch.sin(h * new_angle))
        harm_vec = torch.cat(harm_feats, dim=-1)
        
        winding_feat = self.winding_proj(torch.tanh(new_winding.detach() * 0.01))
        phase_input = torch.cat([harm_vec, winding_feat], dim=-1)
        phase_signal = self.phase_reader(phase_input)
        
        # === CROSS-BAND MIXING ===
        # Let the fast clocks inform the slow clocks and vice versa
        mixed = self.band_mixer(phase_signal)
        
        # === GRU UPDATE + MIXED PHASE INJECTION ===
        h_new = (1 - z) * h_state + z * n + 0.1 * mixed
        
        return h_new, h_new, new_angle, new_winding


class YarnixModelV4(nn.Module):
    """Stacked YarnixCellV4 with multi-clock phase tracking."""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2,
                 harmonics=[1, 2, 4, 8], quantization_strength=0.05,
                 clock_speeds=(0.50, 0.80, 0.95, 0.999)):
        super(YarnixModelV4, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            inp = input_size if i == 0 else hidden_size
            self.layers.append(YarnixCellV4(
                inp, hidden_size, harmonics,
                quantization_strength, clock_speeds
            ))
        
        # Readout with band-aware structure:
        # Two-layer MLP so it can learn to weight timescales
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )
        for m in self.readout.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, input_seq, return_sequence=False):
        batch_size, seq_len, _ = input_seq.size()
        device = input_seq.device
        
        h_states = [torch.zeros(batch_size, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]
        angles   = [torch.zeros(batch_size, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]
        windings = [torch.zeros(batch_size, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]
        
        outputs = []
        for t in range(seq_len):
            current = input_seq[:, t, :]
            for i, layer in enumerate(self.layers):
                current, h_states[i], angles[i], windings[i] = layer(
                    current, h_states[i], angles[i], windings[i])
            if return_sequence:
                outputs.append(self.readout(current))
        
        if return_sequence:
            return torch.stack(outputs, dim=1)
        return self.readout(current)
