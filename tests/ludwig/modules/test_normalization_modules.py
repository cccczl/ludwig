from typing import Optional

import pytest
import torch

from ludwig.modules.normalization_modules import GhostBatchNormalization

BATCH_SIZE = 16
FC_SIZE = 8


@pytest.mark.parametrize("virtual_batch_size", [None, BATCH_SIZE // 2, BATCH_SIZE - 14, BATCH_SIZE - 10])
@pytest.mark.parametrize("mode", [True, False])  # training (True) or eval(False)
def test_ghostbatchnormalization(mode: bool, virtual_batch_size: Optional[int]) -> None:
    # setup up GhostBatchNormalization layer
    ghost_batch_norm = GhostBatchNormalization(FC_SIZE, virtual_batch_size=virtual_batch_size)

    # set training or eval mode
    ghost_batch_norm.train(mode=mode)

    # setup inputs to test
    inputs = torch.randn([BATCH_SIZE, FC_SIZE], dtype=torch.float32)

    # run tensor through
    norm_tensor = ghost_batch_norm(inputs)

    # check for correctness of output
    assert isinstance(norm_tensor, torch.Tensor)
    assert norm_tensor.shape == (BATCH_SIZE, FC_SIZE)

    # check for required properties
    assert ghost_batch_norm.input_shape == inputs.shape[1:]
    assert ghost_batch_norm.output_shape == inputs.shape[1:]
    assert ghost_batch_norm.input_dtype == torch.float32

    assert isinstance(ghost_batch_norm.moving_mean, torch.Tensor)
    assert ghost_batch_norm.moving_mean.shape == (FC_SIZE,)

    assert isinstance(ghost_batch_norm.moving_variance, torch.Tensor)
    assert ghost_batch_norm.moving_variance.shape == (FC_SIZE,)
