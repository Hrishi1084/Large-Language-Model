#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn


# In[3]:


inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)


# In[4]:


input_query = inputs[1]
input_query


# In[5]:


input_1 = inputs[0]
input_1


# In[6]:


torch.dot(input_query, input_1)


# In[7]:


attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, input_query)
attn_scores_2


# In[8]:


attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
attn_weights_2_tmp


# In[9]:


attn_weights_2_tmp.sum()


# In[10]:


def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim = 0)
softmax_naive(attn_scores_2)


# In[11]:


attn_weights_2 = torch.softmax(attn_scores_2, dim = 0)


# In[12]:


query = inputs[1]

context_vec_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    # print(f"{attn_weights_2[i]} --------> {inputs[i]}")
    context_vec_2 += attn_weights_2[i] * inputs[i]
context_vec_2


# In[13]:


attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

attn_scores


# In[14]:


attn_scores = inputs @ inputs.T
attn_scores


# In[15]:


attn_weights = torch.softmax(attn_scores, dim = 1)
attn_weights


# In[16]:


all_context_vecs = attn_weights @ inputs
all_context_vecs


# In[17]:


x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2


# In[18]:


torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
W_value = torch.nn.Parameter(torch.rand(d_in, d_out))


# In[19]:


query_2 = x_2 @ W_query
query_2


# In[20]:


keys = inputs @ W_key
values = inputs @ W_value


# In[21]:


key_2 = keys[1]
attn_score_22 = torch.dot(key_2, query_2)
attn_score_22


# In[22]:


attn_scores_2 = query_2 @ keys.T
attn_scores_2


# In[23]:


d_k = keys.shape[1]

attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim = -1)
attn_weights_2


# In[24]:


context_vec_2 = attn_weights_2 @ values
context_vec_2


# In[25]:


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = inputs @ W_query
        keys = inputs @ W_key
        values = inputs @ W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / d_k ** 0.5, dim = -1)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
sa_v1(inputs)


# In[26]:


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / d_k ** 0.5, dim = -1)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
sa_v2(inputs)


# In[27]:


queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
values = sa_v2.W_value(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / d_k ** 0.5, dim = -1)


# In[28]:


attn_weights


# In[29]:


context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
mask_simple


# In[30]:


masked_simple = attn_weights * mask_simple
masked_simple


# In[31]:


row_sums = masked_simple.sum(dim = -1, keepdim = True)
masked_simple_norm = masked_simple / row_sums
masked_simple_norm


# In[32]:


mask = torch.triu(torch.ones(context_length, context_length), diagonal = 1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
masked


# In[33]:


attn_weights = torch.softmax(masked / d_k ** 0.5, dim = -1)
attn_weights


# In[34]:


torch.manual_seed(123)

layer = nn.Dropout(0.5)
layer(attn_weights)


# In[35]:


batch = torch.stack((inputs, inputs), dim = 0)
batch.shape


# In[36]:


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal = 1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[0] ** 0.5, dim = -1 
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
context_length = batch.shape[1]
dropout = 0.0
ca = CausalAttention(d_in, d_out, context_length, dropout)
ca(batch)


# In[37]:


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads = 2, qkv_bias = False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout) for _ in range(num_heads) 
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim = -1)

torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout = 0.0, num_heads = 2)
mha(batch)


# In[39]:


class MultiHeadAttention(nn.Module):
    def __init__(self,  d_in, d_out, context_length, dropout, num_heads = 2, qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal = 1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1 )
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, dropout = 0.0, num_heads = 2)
context_vecs = mha(batch)
context_vecs


# In[ ]:




