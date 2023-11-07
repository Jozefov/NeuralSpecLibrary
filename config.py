import torch

# Device setting
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # How much can predicted spectrum weight more than actual value
# MASS_SHIFT = 5
#
# # Scaling factor for spectrum validation and computation
# INTENSITY_POWER = 0.5
#
# # Scaling factor for spectrum validation and computation
# MASS_POWER = 1.0
