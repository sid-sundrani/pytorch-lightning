# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from distutils.version import LooseVersion
from unittest import mock

import pytest
import torch
import torch.distributed as torch_distrib
from torch import nn

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.native_amp import NativeAMPPlugin
from pytorch_lightning.plugins.pipe_plugin import HAS_FAIRSCALE, PipePlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.backends.launcher import DDPLauncher
from tests.base.boring_model import BoringModel, RandomDataset


class SequentialModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(torch.nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.layer(x)

    def loss(self, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        output = self.layers(batch)

        loss = torch.zeros(1).to(self.device)

        if self.final_stage:
            loss = self.loss(output)
            self.manual_backward(loss, opt)
            assert torch.stack([torch.abs(p.grad).sum() for p in self.parameters()]).sum() > 0
            self.manual_optimizer_step(opt)
            assert torch.stack([torch.abs(p.grad).sum() for p in self.parameters()]).sum() == 0
        else:
            self.back_helper(output)

        self.log("train_loss", loss, sync_dist=True, on_step=True,
                 on_epoch=True, reduce_fx=torch.sum, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        output = self.layers(batch)
        if self.final_stage:
            loss = self.loss(output)

    def test_step(self, batch, batch_idx):
        output = self.layer(batch)
        if self.final_stage:
            loss = self.loss(batch, output)

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layers.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))


@pytest.mark.skipif(not HAS_FAIRSCALE, reason="test requires fairscale to be installed")
@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1', reason="test should be run outside of pytest")
@DDPLauncher.run("--distributed_backend ddp --gpus 2")
def test_pipe_plugin_ddp(tmpdir, args=None):

    model = SequentialModel()
    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        gpus=args.gpus,
        distributed_backend=args.distributed_backend,
        plugins=[PipePlugin(balance=[2, 1], version=1)],
        automatic_optimization=False,
    )
    trainer.fit(model)

    assert len(trainer.dev_debugger.pbar_added_metrics) > 0


def run_optimizer(ctx, model):
    for opt in model.trainer.optimizers:
        opt.step()


class SequentialModelRPC(LightningModule):

    _count = 0
    _called = 0

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(torch.nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.layer(x)

    def loss(self, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        if batch_idx % 2 == 0:
            self._count += 1
            opt = self.optimizers()
            output = self.layers(batch)
            loss = self.loss(output)
            self.log("train_loss", loss, on_epoch=True, prog_bar=True)
            self.manual_backward(loss, opt)
            assert torch.stack([torch.abs(p.grad).sum() for p in self.parameters()]).sum() > 0
            self.manual_optimizer_step(opt)
        else:
            opt = self.optimizers()

            def optimizer_closure():
                self._count += 1
                output = self.layers(batch)
                loss = self.loss(output)
                self.log("train_loss", loss, on_epoch=True, prog_bar=True)
                self.manual_backward(loss, opt)
                assert torch.stack([torch.abs(p.grad).sum() for p in self.parameters()]).sum() > 0
            self.manual_optimizer_step(opt, optimizer_closure=optimizer_closure)
        self._called += 1
        assert self._called == self._count
        assert torch.stack([torch.abs(p.grad).sum() for p in self.parameters()]).sum() == 0

    def validation_step(self, batch, batch_idx):
        print(torch_distrib.get_rank(), batch, self.layers)
        with torch.cuda.amp.autocast():
            output = self.layers(batch)
            loss = self.loss(output)
            return loss

    def test_step(self, batch, batch_idx):
        with torch.cuda.amp.autocast():
            output = self.layers(batch)
            return self.loss(batch, output)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layers.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))


def cleanup(ctx, model):
    del model


@pytest.mark.skipif(not HAS_FAIRSCALE, reason="test requires fairscale to be installed")
@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1', reason="test should be run outside of pytest")
def test_pipe_plugin_ddp_rpc_manual(tmpdir, args=None):
    model = SequentialModelRPC()
    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        gpus=2,
        distributed_backend="ddp",
        plugins=[PipePlugin(balance=[2, 1], version=2)],
        automatic_optimization=False,
    )
    trainer.fit(model)

    assert len(trainer.dev_debugger.pbar_added_metrics) > 0

    model.foreach_worker(cleanup, include_self=True)

    del model


@pytest.mark.skipif(not HAS_FAIRSCALE, reason="test requires fairscale to be installed")
@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1', reason="test should be run outside of pytest")
def test_pipe_plugin_ddp_rpc_manual_amp(tmpdir, args=None):
    model = SequentialModelRPC()
    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        gpus=2,
        precision=16,
        amp_backend="native",
        distributed_backend="ddp",
        plugins=[PipePlugin(balance=[2, 1], version=2)],
        automatic_optimization=False,
    )
    with pytest.raises(MisconfigurationException, match='not supported in Automatic Mixed Precision'):
        trainer.fit(model)

        assert len(trainer.dev_debugger.pbar_added_metrics) > 0

    del model


class SequentialModelRPCAutomatic(LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(torch.nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.layer(x)

    def loss(self, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        output = self.layers(batch)
        loss = self.loss(output)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.layers(batch)
        loss = self.loss(output)
        return loss

    def test_step(self, batch, batch_idx):
        output = self.layer(batch)
        return self.loss(batch, output)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layers.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))


@pytest.mark.skipif(not HAS_FAIRSCALE, reason="test requires fairscale to be installed")
@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1', reason="test should be run outside of pytest")
def test_pipe_plugin_ddp_rpc_automatic(tmpdir, args=None):
    model = SequentialModelRPCAutomatic()
    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        gpus=2,
        distributed_backend="ddp",
        plugins=[PipePlugin(balance=[2, 1], version=2)],
        automatic_optimization=True,
    )

    with pytest.raises(MisconfigurationException, match='PipePlugin is currently not'):
        trainer.fit(model)

        assert len(trainer.dev_debugger.pbar_added_metrics) > 0

        model.foreach_worker(cleanup, include_self=True)

    del model