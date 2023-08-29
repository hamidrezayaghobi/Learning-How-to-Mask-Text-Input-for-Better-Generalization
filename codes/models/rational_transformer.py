import torch
import torch.nn as nn

class TransformerRationalePredictor(nn.Module):
  def __init__(self, num_layers, d_model, num_heads,
               dim_feedforward, dropout_rate=0.1):
    super().__init__()

    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads
    self.dim_feedforward = dim_feedforward

    self.linear = nn.Linear(self.d_model, self.d_model* self.num_heads, dtype=torch.float64)


    #TODO: ADD NUM_HEADS
    self.enc_layers = nn.Sequential(
        *([nn.TransformerEncoderLayer(d_model=self.d_model * self.num_heads,
                                      dtype=torch.float64,
                                      nhead=self.num_heads,
                                      dim_feedforward=self.dim_feedforward,
                                      dropout=dropout_rate,
                                      batch_first=True)] * num_layers)
        )
    self.norm = nn.BatchNorm1d(self.d_model* self.num_heads, dtype=torch.float64)
    self.linear2 = nn.Linear(self.d_model * self.num_heads, self.d_model, dtype=torch.float64)

  def forward(self, x):
    '''
    inputs:
            x : [batch_size, num_tokens]
    '''
    # x = self.linear(x) it must not use because it will disorder the text and in encoder we have positional encoder. even in the picutre
    x = self.enc_layers(x)
    # x = self.norm(x)
    # x = self.linear2(x)
    return x  # Shape `(batch_size, seq_len)
