import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
# from hydra.utils import instantiate
from torch.utils.data import DataLoader, ConcatDataset, Subset
from omegaconf import ListConfig, DictConfig
from hmr4d.utils.pylogger import Log
from numpy.random import choice
from torch.utils.data import default_collate
from hmr4d.dataset.pure_motion.amass import AmassDataset
from hmr4d.dataset.bedlam.bedlam import BedlamDatasetV2
from hmr4d.dataset.bedlam.bedlam2 import Bedlam2Dataset
from hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
from hmr4d.dataset.rich.rich_motion_test import RichSmplFullSeqDataset
from hmr4d.dataset.threedpw.threedpw_motion_test import ThreedpwSmplFullSeqDataset
from hmr4d.dataset.bedlam.bedlam2_test import Bedlam2TestDataset

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def collate_fn(batch):
    """Handle meta and Add batch size to the return dict
    Args:
        batch: list of dict, each dict is a data point
    """
    # Assume all keys in the batch are the same
    return_dict = {}
    for k in batch[0].keys():
        if k.startswith("meta"):  # data information, do not batch
            return_dict[k] = [d[k] for d in batch]
        else:
            return_dict[k] = default_collate([d[k] for d in batch])
    return_dict["B"] = len(batch)
    return return_dict


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_opts: DictConfig, loader_opts: DictConfig, limit_each_trainset=None, task="fit"):
        """This is a general datamodule that can be used for any dataset.
        Train uses ConcatDataset
        Val and Test use CombinedLoader, sequential, completely consumes ecah iterable sequentially, and returns a triplet (data, idx, iterable_idx)

        Args:
            dataset_opts: the target of the dataset. e.g. dataset_opts.train = {_target_: ..., limit_size: None}
            loader_opts: the options for the dataset
            limit_each_trainset: limit the size of each dataset, None means no limit, useful for debugging
        """
        super().__init__()
        self.loader_opts = loader_opts
        self.limit_each_trainset = limit_each_trainset

        # Train uses concat dataset
        if "train" in dataset_opts and task == "fit":
            assert "train" in self.loader_opts, "train not in loader_opts"
            split_opts = dataset_opts.get("train")
            assert isinstance(split_opts, DictConfig), "split_opts should be a dict for each dataset"
            dataset = []
            dataset_num = len(split_opts)
            for idx, (k, v) in enumerate(split_opts.items()):
                if k == "pure_motion_amass":
                    dataset_i = AmassDataset()
                elif k == "imgfeat_bedlam":
                    dataset_i = BedlamDatasetV2()
                elif k == "imgfeat_bedlam2":
                    dataset_i = Bedlam2Dataset()
                else:
                    raise ValueError(f"Unknown dataset: {k}")
                if self.limit_each_trainset:
                    dataset_i = Subset(dataset_i, choice(len(dataset_i), self.limit_each_trainset))
                    
                dataset.append(dataset_i)
                Log.info(f"[Train Dataset][{idx+1}/{dataset_num}]: name={k}, size={len(dataset[-1])}, {k}")
            dataset = ConcatDataset(dataset)
            self.trainset = dataset
            Log.info(f"[Train Dataset][All]: ConcatDataset size={len(dataset)}")
            Log.info(f"")

        # Val and Test use sequential dataset
        for split in ("val", "test"):
            if split not in dataset_opts:
                continue
            assert split in self.loader_opts, f"split={split} not in loader_opts"
            split_opts = dataset_opts.get(split)
            assert isinstance(split_opts, DictConfig), "split_opts should be a dict for each dataset"
            dataset = []
            dataset_num = len(split_opts)
            for idx, (k, v) in enumerate(split_opts.items()):
                if k == "emdb1":
                    dataset_i = EmdbSmplFullSeqDataset(split=v.split, flip_test=v.flip_test)
                elif k == "emdb2":
                    dataset_i = EmdbSmplFullSeqDataset(split=v.split, flip_test=v.flip_test)
                elif k == "rich":
                    dataset_i = RichSmplFullSeqDataset(vid_presets=v.vid_presets)
                elif k == "3dpw":
                    dataset_i = ThreedpwSmplFullSeqDataset(flip_test=v.flip_test)
                elif k == "bedlam2":
                    dataset_i = Bedlam2TestDataset(version=v.version)
                else:
                    raise ValueError(f"Unknown dataset: {k}")
                dataset.append(dataset_i)
                dataset_type = "Val Dataset" if split == "val" else "Test Dataset"
                Log.info(f"[{dataset_type}][{idx+1}/{dataset_num}]: name={k}, size={len(dataset[-1])}, {k}")
            setattr(self, f"{split}sets", dataset)
            Log.info(f"")

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            return DataLoader(
                self.trainset,
                shuffle=True,
                num_workers=self.loader_opts.train.num_workers,
                persistent_workers=True and self.loader_opts.train.num_workers > 0,
                batch_size=self.loader_opts.train.batch_size,
                drop_last=True,
                collate_fn=collate_fn,
            )
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valsets"):
            loaders = []
            for valset in self.valsets:
                loaders.append(
                    DataLoader(
                        valset,
                        shuffle=False,
                        num_workers=self.loader_opts.val.num_workers,
                        persistent_workers=True and self.loader_opts.val.num_workers > 0,
                        batch_size=self.loader_opts.val.batch_size,
                        collate_fn=collate_fn,
                        # pin_memory=True,
                        # prefetch_factor=4,
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return None

    def test_dataloader(self):
        if hasattr(self, "testsets"):
            loaders = []
            for testset in self.testsets:
                loaders.append(
                    DataLoader(
                        testset,
                        shuffle=False,
                        num_workers=self.loader_opts.test.num_workers,
                        persistent_workers=False,
                        batch_size=self.loader_opts.test.batch_size,
                        collate_fn=collate_fn,
                        # pin_memory=True,
                        # prefetch_factor=4,
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return super().test_dataloader()
