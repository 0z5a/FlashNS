"""Sample partitions and global normalization for manual parameter gradients."""

import math
from itertools import pairwise


def shard_bounds(global_count, world_size, rank):
    if global_count < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid global count, world size or rank")
    return global_count * rank // world_size, global_count * (rank + 1) // world_size


def global_normalization(weights):
    values = [float(w) for w in weights]
    if any(not math.isfinite(w) or w < 0 for w in values):
        raise ValueError("fixed quadrature weights must be finite and nonnegative")
    denominator = math.fsum(values)
    if denominator <= 0:
        raise ValueError("global quadrature mass must be positive")
    return denominator


def parameter_count(widths):
    if len(widths) < 2 or any(w <= 0 for w in widths):
        raise ValueError("positive input, hidden and output widths required")
    return sum(i * o + o for i, o in pairwise(widths))
