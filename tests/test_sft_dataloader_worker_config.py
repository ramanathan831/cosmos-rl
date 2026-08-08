import pickle

from cosmos_rl.policy.worker.sft_worker import _dataloader_worker_kwargs
from cosmos_rl.utils.cache import DiskCache


def test_nonzero_workers_use_spawn_and_prefetch():
    assert _dataloader_worker_kwargs(1, 1) == {
        "num_workers": 1,
        "multiprocessing_context": "spawn",
        "prefetch_factor": 1,
    }


def test_zero_workers_do_not_apply_prefetch_or_context():
    assert _dataloader_worker_kwargs(0, 1) == {"num_workers": 0}


def test_disk_cache_is_spawn_picklable(tmp_path):
    cache = DiskCache(str(tmp_path / "cache"))
    restored = pickle.loads(pickle.dumps(cache))

    assert restored.cache_path == cache.cache_path
    assert restored.executor is not cache.executor

    cache.executor.shutdown(wait=True)
    restored.executor.shutdown(wait=True)
