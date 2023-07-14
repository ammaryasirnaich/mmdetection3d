import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F

# Define the SelfAttention layer using simple torch.matmul functions
class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(CausalSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, src):
        batch_size, seq_len, hidden_dim = src.size()
        
        # Linear transformations of query, key, and value
        query = self.query_linear(src)
        key = self.key_linear(src)
        value = self.value_linear(src)
        
        # Reshape query, key, and value for multiple heads
        query = query.view(batch_size, seq_len, self.num_heads, hidden_dim // self.num_heads)
        key = key.view(batch_size, seq_len, self.num_heads, hidden_dim // self.num_heads)
        value = value.view(batch_size, seq_len, self.num_heads, hidden_dim // self.num_heads)
        
        # Transpose for batch matrix multiplication
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (hidden_dim // self.num_heads)**0.5
              
        # Apply softmax to get attention weights
        attention_weights = nn.functional.softmax(scores, dim=-1)
        
        # Compute attention output
        attention_output = torch.matmul(attention_weights, value)
        
        # Transpose attention output
        attention_output = attention_output.transpose(1, 2)
        
        # Reshape attention output
        attention_output = attention_output.contiguous().view(batch_size, seq_len, hidden_dim)

        return attention_output
    

# Define the ScaledSelfAttention layer using F.scaled_dot_product_attention
class ScaledSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(ScaledSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, src):
        batch_size, seq_len, hidden_dim = src.size()
        
        # Linear transformations of query, key, and value
        query = self.query_linear(src)
        key = self.key_linear(src)
        value = self.value_linear(src)
        
        # Reshape query, key, and value for multiple heads
        query = query.view(batch_size, seq_len, self.num_heads, hidden_dim // self.num_heads)
        key = key.view(batch_size, seq_len, self.num_heads, hidden_dim // self.num_heads)
        value = value.view(batch_size, seq_len, self.num_heads, hidden_dim // self.num_heads)
        
        # Transpose for batch matrix multiplication
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        
        # Compute attention scores
        attention_output = F.scaled_dot_product_attention(query, key, value)
        attention_output = torch.einsum('bijk->bjik', attention_output)
        B,N,H,D = attention_output.shape
        attention_output =attention_output.reshape(B,N,H*D)    
        return attention_output
    

# Define the Transformer model
class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(Transformer, self).__init__()
        self.encoder = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoder = PositionalEncoder(hidden_dim)

        # self.attention = CausalSelfAttention(hidden_dim, num_heads)
        self.attention = ScaledSelfAttention(hidden_dim, num_heads)
        
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, src):
        src = self.encoder(src)
        src = self.positional_encoder(src)
        output = self.attention(src)
        # print("output:", output.shape)
        output = self.fc(output)
        return output

# Positional encoding
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=100):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.pos_enc = self.positional_encoding(max_seq_len, d_model)
    
    def positional_encoding(self, max_seq_len, d_model):
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)
    
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

# Define the model hyperparameters
input_dim = 100  # Input vocabulary size
hidden_dim = 256  # Hidden dimension size
num_layers = 2  # Number of transformer layers
num_heads = 4  # Number of attention heads
num_epochs = 5  # Number of training epochs
learning_rate = 0.001  # Learning rate

# Create an instance of the Transformer model
model = Transformer(input_dim, hidden_dim, num_layers, num_heads)

# Define a dummy input and target
src = torch.randint(0, input_dim, (16, 10))  # Batch size = 16, Sequence length = 10
target = torch.zeros(16, 10, hidden_dim)  # Dummy target of the same shape as the output

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(src)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Test the model on a new input after training
test_input = torch.randint(0, input_dim, (1, 10))  # Batch size = 1, Sequence length = 10
test_output = model(test_input)
print(test_output.shape)  # Output shape: (1, 10, 256)