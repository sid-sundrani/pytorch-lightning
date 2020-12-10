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
"""Test deprecated functionality which will be removed in vX.Y.Z"""
import sys
from argparse import ArgumentParser
from unittest import mock

import pytest
import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.metrics.functional.classification import auc
from pytorch_lightning.profiler.profilers import PassThroughProfiler, SimpleProfiler
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate


def test_tbd_remove_in_v1_3_0(tmpdir):
    with pytest.deprecated_call(match='will no longer be supported in v1.3'):
        callback = ModelCheckpoint()
        Trainer(checkpoint_callback=callback, callbacks=[], default_root_dir=tmpdir)

    # Deprecate prefix
    with pytest.deprecated_call(match='will be removed in v1.3'):
        ModelCheckpoint(prefix='temp')

    # Deprecate auto mode
    with pytest.deprecated_call(match='will be removed in v1.3'):
        ModelCheckpoint(mode='auto')

    with pytest.deprecated_call(match='will be removed in v1.3'):
        EarlyStopping(mode='auto')

    with pytest.deprecated_call(match="The setter for self.hparams in LightningModule is deprecated"):
        class DeprecatedHparamsModel(LightningModule):
            def __init__(self, hparams):
                super().__init__()
                self.hparams = hparams
        DeprecatedHparamsModel({})


def test_tbd_remove_in_v1_2_0():
    with pytest.deprecated_call(match='will be removed in v1.2'):
        checkpoint_cb = ModelCheckpoint(filepath='.')

    with pytest.deprecated_call(match='will be removed in v1.2'):
        checkpoint_cb = ModelCheckpoint('.')

    with pytest.raises(MisconfigurationException, match='inputs which are not feasible'):
        checkpoint_cb = ModelCheckpoint(filepath='.', dirpath='.')


# TODO: remove bool from Trainer.profiler param in v1.3.0, update profiler_connector.py
@pytest.mark.parametrize(['profiler', 'expected'], [
    (True, SimpleProfiler),
    (False, PassThroughProfiler),
])
def test_trainer_profiler_remove_in_v1_3_0(profiler, expected):
    with pytest.deprecated_call(match='will be removed in v1.3'):
        trainer = Trainer(profiler=profiler)
        assert isinstance(trainer.profiler, expected)


@pytest.mark.parametrize(
    ['cli_args', 'expected_parsed_arg', 'expected_profiler'],
    [
        ('--profiler', True, SimpleProfiler),
        ('--profiler True', True, SimpleProfiler),
        ('--profiler False', False, PassThroughProfiler),
    ],
)
def test_trainer_cli_profiler_remove_in_v1_3_0(cli_args, expected_parsed_arg, expected_profiler):
    cli_args = cli_args.split(' ')
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        parser = ArgumentParser(add_help=False)
        parser = Trainer.add_argparse_args(parent_parser=parser)
        args = Trainer.parse_argparser(parser)

    assert getattr(args, "profiler") == expected_parsed_arg
    trainer = Trainer.from_argparse_args(args)
    assert isinstance(trainer.profiler, expected_profiler)


def _soft_unimport_module(str_module):
    # once the module is imported  e.g with parsing with pytest it lives in memory
    if str_module in sys.modules:
        del sys.modules[str_module]


class ModelVer0_6(EvalModelTemplate):

    # todo: this shall not be needed while evaluate asks for dataloader explicitly
    def val_dataloader(self):
        return self.dataloader(train=False)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        return {'val_loss': torch.tensor(0.6)}

    def validation_end(self, outputs):
        return {'val_loss': torch.tensor(0.6)}

    def test_dataloader(self):
        return self.dataloader(train=False)

    def test_end(self, outputs):
        return {'test_loss': torch.tensor(0.6)}


class ModelVer0_7(EvalModelTemplate):

    # todo: this shall not be needed while evaluate asks for dataloader explicitly
    def val_dataloader(self):
        return self.dataloader(train=False)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        return {'val_loss': torch.tensor(0.7)}

    def validation_end(self, outputs):
        return {'val_loss': torch.tensor(0.7)}

    def test_dataloader(self):
        return self.dataloader(train=False)

    def test_end(self, outputs):
        return {'test_loss': torch.tensor(0.7)}
