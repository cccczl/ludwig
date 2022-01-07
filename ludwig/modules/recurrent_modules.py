# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
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
# ==============================================================================
import logging
from typing import Optional

import torch
from torch.nn import GRU, LSTM, RNN

from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.torch_utils import LudwigModule

logger = logging.getLogger(__name__)

rnn_layers_registry = {
    "rnn": RNN,
    "gru": GRU,
    "lstm": LSTM,
}


class RecurrentStack(LudwigModule):
    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = 256,
        cell_type: str = "rnn",
        sequence_size: Optional[int] = None,
        num_layers: int = 1,
        bidirectional: bool = False,
        use_bias: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.supports_masking = True
        self.input_size = input_size  # api doc: H_in
        self.hidden_size = hidden_size  # api doc: H_out
        self.sequence_size = sequence_size  # api doc: L (sequence length)

        rnn_layer_class = get_from_registry(cell_type, rnn_layers_registry)

        rnn_params = {"num_layers": num_layers, "bias": use_bias, "dropout": dropout, "bidirectional": bidirectional}

        # Delegate recurrent params to PyTorch's RNN/GRU/LSTM implementations.
        self.layers = rnn_layer_class(input_size, hidden_size, batch_first=True, **rnn_params)

    def input_shape(self) -> torch.Size:
        if self.sequence_size:
            return torch.Size([self.sequence_size, self.input_size])
        return torch.Size([self.input_size])

    def output_shape(self) -> torch.Size:
        if self.sequence_size:
            return torch.Size([self.sequence_size, self.hidden_size])
        return torch.Size([self.hidden_size])

    def forward(self, inputs: torch.Tensor, mask=None):
        hidden, final_state = self.layers(inputs)

        if isinstance(final_state, tuple):
            # lstm cell type
            final_state = final_state[0][-1], final_state[1][-1]
        else:
            # rnn or gru cell type
            final_state = final_state[-1]

        return hidden, final_state
