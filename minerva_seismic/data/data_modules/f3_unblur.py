import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from minerva.data.readers import MultiReader, PatchedArrayReader
from minerva.data.datasets import SupervisedReconstructionDataset
from minerva.transforms import _Transform, Identity
from typing import Sequence, Union, Optional
from pathlib import Path
import os


class F3UnblurDataModule(L.LightningDataModule):

    def __init__(
        self,
        data_root: Union[str, Path],
        data_layers: Sequence[str],
        transform: _Transform = None,
        batch_size: int = 1,
        val_percent: float = 0.2,
        num_workers: Optional[int] = None
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.data_layers = data_layers
        self.transform = transform
        self.batch_size = batch_size
        self.val_percent = val_percent
        self.num_workers = num_workers or os.cpu_count()

        self.datasets = {}
    
    def setup(self, stage=None):
        if stage == "fit":
            train_files = [self.data_root / f"F3_train_{layer}.npy" for layer in self.data_layers]
            train_data = [np.load(file) for file in train_files]
            train_reader = MultiReader([
                PatchedArrayReader(d, (1, d.shape[1], d.shape[2]))
                for d in train_data
            ])

            train_dataset = SupervisedReconstructionDataset(
                [train_reader, train_reader],
                [self.transform, Identity()]
            )

            train_range = range(int((1 - self.val_percent) * len(train_dataset)))
            val_range = range(int((1 - self.val_percent) * len(train_dataset)), len(train_dataset))

            self.datasets['val'] = Subset(train_dataset, val_range)
            self.datasets['train'] = Subset(train_dataset, train_range)
        
        elif stage == "test" or stage == "predict":
            test_files = [self.data_root / f"F3_test_{layer}" for layer in self.data_layers]
            test_data = [np.load(file) for file in test_files]
            test_reader = MultiReader([
                PatchedArrayReader(d, d.shape[-2:])
                for d in test_data
            ])

            test_dataset = SupervisedReconstructionDataset(
                [test_reader, test_reader],
                [self.transform, Identity()]
            )

            self.datasets['test'] = test_dataset
            self.datasets['predict'] = test_dataset
        
        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

