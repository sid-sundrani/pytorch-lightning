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
import os
from unittest import mock
from unittest.mock import patch

import pytest

from pytorch_lightning import Trainer
from tests.base.boring_model import BoringModel


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_reload_dataloaders_every_epoch_remove_in_v1_4_0(tmpdir):

    with pytest.deprecated_call(match='will be removed in v1.4'):
        model = BoringModel()

        trainer = Trainer(
            default_root_dir=tmpdir,
            limit_train_batches=0.3,
            limit_val_batches=0.3,
            reload_dataloaders_every_epoch=True,
            max_epochs=3,
        )
        trainer.fit(model)
        trainer.test()

        # verify the sequence
        calls = trainer.dev_debugger.dataloader_sequence_calls
        expected_sequence = [
            'val_dataloader',
            'train_dataloader',
            'val_dataloader',
            'train_dataloader',
            'val_dataloader',
            'train_dataloader',
            'val_dataloader',
            'test_dataloader'
        ]
        for call, expected in zip(calls, expected_sequence):
            assert call['name'] == expected
