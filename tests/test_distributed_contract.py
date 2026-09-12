import math

import pytest

from flashns.distributed_contract import (
    global_normalization,
    parameter_count,
    shard_bounds,
)


@pytest.mark.parametrize("points,world", [(17, 1), (17, 2), (17, 4), (3, 4), (0, 4)])
def test_shards_cover_global_ids_once_including_empty_ranks(points, world):
    shards = [list(range(*shard_bounds(points, world, rank))) for rank in range(world)]
    assert [i for shard in shards for i in shard] == list(range(points))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


@pytest.mark.parametrize("points,world", [(17, 4), (3, 4)])
def test_weighted_local_sums_have_one_global_denominator(points, world):
    weights = [0.5 + (i % 7) / 8 for i in range(points)]
    derivative = [math.cos(i / 3) for i in range(points)]
    z = global_normalization(weights)
    expected = math.fsum(w * d for w, d in zip(weights, derivative)) / z
    partials = []
    for rank in range(world):
        start, stop = shard_bounds(points, world, rank)
        partials.append(
            math.fsum(weights[i] * derivative[i] for i in range(start, stop)) / z
        )
    assert math.fsum(partials) == pytest.approx(expected, abs=1e-15)


def test_model_gradient_payload_comes_from_parameters():
    assert parameter_count([2, 64, 64, 3]) == 4547
    assert parameter_count([2, 64, 64, 3]) * 8 == 36376


@pytest.mark.parametrize("weights", [[0, 0], [1, -1], [float("nan")], [float("inf")]])
def test_invalid_global_weights_rejected(weights):
    with pytest.raises(ValueError):
        global_normalization(weights)
