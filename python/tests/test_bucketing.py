"""GPU-free tests for the gradient bucketing logic used by the overlapping DDP.

These exercise `plan_buckets` only, so they run anywhere the wheel is installed
(no CUDA/NCCL needed) and guard the invariant the overlap engine relies on:
every parameter lands in exactly one bucket and no bucket exceeds the cap unless
a single parameter is itself larger than the cap.
"""

from gradsync import plan_buckets


def _flatten(buckets):
    return [i for b in buckets for i in b]


def test_every_param_assigned_once_and_in_order():
    sizes = [10, 20, 5, 40, 15]
    buckets = plan_buckets(sizes, cap_elems=30)
    assert _flatten(buckets) == list(range(len(sizes)))  # each index once, in order


def test_respects_cap_when_possible():
    sizes = [10, 10, 10, 10]
    buckets = plan_buckets(sizes, cap_elems=25)
    # Greedy fill: [0,1] = 20 (adding 2 -> 30 > 25 so break), [2,3] = 20.
    assert buckets == [[0, 1], [2, 3]]
    for b in buckets:
        assert sum(sizes[i] for i in b) <= 25


def test_oversized_param_gets_its_own_bucket():
    sizes = [5, 100, 5]
    buckets = plan_buckets(sizes, cap_elems=30)
    # 100 exceeds the cap: it flushes [0], sits alone as [1], then [2] follows.
    assert buckets == [[0], [1], [2]]


def test_single_bucket_when_everything_fits():
    sizes = [1, 2, 3]
    assert plan_buckets(sizes, cap_elems=1000) == [[0, 1, 2]]


def test_empty():
    assert plan_buckets([], cap_elems=10) == []
