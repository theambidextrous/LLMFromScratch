import torch
import torch.nn as nn

# Implementation with torch linear layers
class AttentionMechanism_V2(nn.Module):
    def __init__(self, dim_in, dim_out, qkv_bias=False):
        super().__init__()
        self.w_query = nn.Linear(dim_in, dim_out, bias=False)
        self.w_key = nn.Linear(dim_in, dim_out, bias=False)
        self.w_value = nn.Linear(dim_in, dim_out, bias=False)
    
    def forward(self, x):
        queries = self.w_query(x)  # (N,dim_in) x (dim_in,dim_out) = (N,dim_out)
        keys = self.w_key(x)        # (N,dim_in) x (dim_in,dim_out) = (N,dim_out)
        values = self.w_value(x)    # (N,dim_in) x (dim_in,dim_out) = (N,dim_out)
        
        d_k = keys.shape[1]

        attention_scores = queries @ keys.T  # (N,dim_out) x (dim_out,N) = (N,N)
        attention_weights = torch.softmax(attention_scores / (d_k ** 0.5), dim=-1)  # (N,N)
        context_vectors = attention_weights @ values  # (N,N) x (N,dim_out) = (N,dim_out)
        return context_vectors