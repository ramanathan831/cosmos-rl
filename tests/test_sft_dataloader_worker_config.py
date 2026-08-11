import pickle

from cosmos_rl.policy.config import SFTDataConfig
from cosmos_rl.policy.worker.sft_worker import SFTDataset
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


def test_disabled_dataset_cache_processes_samples_directly(monkeypatch):
    def fail_if_cache_is_created(*_args, **_kwargs):
        raise AssertionError("DiskCache must not be created in direct mode")

    monkeypatch.setenv("COSMOS_CACHE", "/must-not-be-used")
    monkeypatch.setattr(
        "cosmos_rl.policy.worker.sft_worker.cache.DiskCache",
        fail_if_cache_is_created,
    )
    config = SFTDataConfig(type="sft", enable_dataset_cache=False)

    dataset = SFTDataset(
        config,
        dataset=[{"conversations": []}],
        data_packer=object(),
        is_user_dataset=True,
        enable_cache=False,
    )

    assert dataset.cache is None
