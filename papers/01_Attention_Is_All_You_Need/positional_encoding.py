"""
Sinusoidal Positional Encoding (from 'Attention Is All You Need').

This file implements fixed positional encodings using sine/cosine waves,
so Transformers can capture token order without recurrence.
"""

import numpy as np
