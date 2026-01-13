import torch
import torch.nn as nn

# Computing for all input vectors compact implementation

class SelfAttentionMechanism_V1(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.w_query = torch.nn.Parameter(torch.rand(dim_in, dim_out))
        self.w_key = torch.nn.Parameter(torch.rand(dim_in, dim_out))
        self.w_value = torch.nn.Parameter(torch.rand(dim_in, dim_out))
    
    def forward(self, y):
        queries = y @ self.w_query  # (N,dim_in) x (dim_in,dim_out) = (N,dim_out)
        keys = y @ self.w_key        # (N,dim_in) x (dim_in,dim_out) = (N,dim_out)
        values = y @ self.w_value    # (N,dim_in) x (dim_in,dim_out) = (N,dim_out)
        
        d_k = keys.shape[1]

        attention_scores = queries @ keys.T  # (N,dim_out) x (dim_out,N) = (N,N)
        attention_weights = torch.softmax(attention_scores / (d_k ** 0.5), dim=-1)  # (N,N)
        context_vectors = attention_weights @ values  # (N,N) x (N,dim_out) = (N,dim_out)
        return context_vectors