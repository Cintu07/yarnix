# Yarnix Configuration
import math

def get_lock_strength(epoch, total_epochs, peak_strength=0.125, floor_strength=0.03125):
    """Gaussian Annealing for quantization strength scheduling."""
    mu = total_epochs / 2.0
    sigma = total_epochs / 6.0
    factor = math.exp(-0.5 * ((epoch - mu) / sigma) ** 2)
    floor = floor_strength if epoch > (total_epochs * 0.5) else 0.0
    return max(peak_strength * factor, floor)

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
HIDDEN_SIZE = 256
HARMONICS = [1, 2, 4, 8]
LR = 0.001
FULL_STATE = True  # Enable infinite context via unwrapped phase
