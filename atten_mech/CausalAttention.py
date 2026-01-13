import torch
import torch.nn as nn

# Hiding future words with causal masking
# Causal attention ensures that the prediction for position i can depend only on the known outputs at positions less than i.

# Causal mask is an upper triangular matrix with -inf values above the main diagonal and 0s elsewhere.

class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout_rate, qkv_bias=False):
        super().__init__()
        self.w_query = nn.Linear(dim_in, dim_out, bias=False)
        self.w_key = nn.Linear(dim_in, dim_out, bias=False)
        self.w_value = nn.Linear(dim_in, dim_out, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        # Registering buffer to hold the causal mask
        self.register_buffer("mask", torch.triu(torch.ones((context_length, context_length)), diagonal=1))
    
    def forward(self, x):
        batch_size, num_tokens, dim_in = x.shape
        queries = self.w_query(x)  # (N,dim_in) x (dim_in,dim_out) = (N,dim_out)
        keys = self.w_key(x)        # (N,dim_in) x (dim_in,dim_out) = (N,dim_out)
        values = self.w_value(x)    # (N,dim_in) x (dim_in,dim_out) = (N,dim_out)

        attention_scores = queries @ keys.transpose(1, 2) # Changed transpose for batch processing
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attention_weights = torch.softmax(attention_scores / (keys.shape[-1] ** 0.5), dim=-1)

        attention_weights = self.dropout(attention_weights)

        context_vectors = attention_weights @ values

        return context_vectors