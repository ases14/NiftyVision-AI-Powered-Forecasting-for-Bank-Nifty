import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(model_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        proj = self.input_linear(x)      # (batch, seq_len, model_dim)
        residual = proj[:, -1, :]        # Save last time step for residual connection
        x_trans = proj.transpose(0, 1)     # (seq_len, batch, model_dim)
        encoded = self.transformer_encoder(x_trans)  # (seq_len, batch, model_dim)
        x_last = encoded[-1, :, :]         # (batch, model_dim)
        x_last = self.dropout(x_last)
        x_final = x_last + residual        # Residual connection
        output = self.fc(x_final)          # (batch, output_dim)
        return output
