import pickle

import torch

from cosmos_rl.policy.config import SFTDataConfig
from cosmos_rl.policy.worker.sft_worker import SFTDataset
from cosmos_rl.policy.worker.sft_worker import _dataloader_worker_kwargs
from cosmos_rl.policy.worker.sft_worker import _initialize_dataloader_worker
from cosmos_rl.utils.cache import DiskCache


class _VideoReaderRegistrationDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, _index):
        import qwen_vl_utils.vision_process as vision_process

        return (
            vision_process.VIDEO_READER_BACKENDS["torchvision"].__module__,
            __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        )


def test_nonzero_workers_use_spawn_and_prefetch():
    assert _dataloader_worker_kwargs(1, 1) == {
        "num_workers": 1,
        "multiprocessing_context": "spawn",
        "prefetch_factor": 1,
    }


def test_zero_workers_do_not_apply_prefetch_or_context():
    assert _dataloader_worker_kwargs(0, 1) == {"num_workers": 0}


def test_system_pyav_worker_initializer_is_spawned(monkeypatch):
    monkeypatch.setenv("COSMOS_DATALOADER_VIDEO_DECODER", "system_pyav")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-A,GPU-B,GPU-C,GPU-D")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    kwargs = _dataloader_worker_kwargs(1, 1)

    assert kwargs["worker_init_fn"].func is _initialize_dataloader_worker
    assert kwargs["worker_init_fn"].keywords == {"decoder_device_index": 3}
    loader = torch.utils.data.DataLoader(
        _VideoReaderRegistrationDataset(), batch_size=1, **kwargs
    )
    module, visible_device = next(iter(loader))
    assert module == ["cosmos_rl.utils.system_pyav_video_reader"]
    assert visible_device == ["GPU-D"]


def test_system_pyav_worker_initializer_accepts_single_visible_device(monkeypatch):
    monkeypatch.setenv("COSMOS_DATALOADER_VIDEO_DECODER", "system_pyav")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-ONLY")

    _initialize_dataloader_worker(0, decoder_device_index=7)

    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "GPU-ONLY"


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
