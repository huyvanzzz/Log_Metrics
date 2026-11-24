"""Shared atomic predictor/RNG serialization; method is enforced by strict signatures."""
from smolvlm_map_predictor.checkpoint import (save, read, rank_rng, rng_state, restore_rng)
