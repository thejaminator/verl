import torch

# Create example data similar to your use case
batch_size = 3
d_model = 4
total_tokens = 10

# Simulate residual activations: (1, total_tokens, d_model)
residual = torch.randn(1, total_tokens, d_model)
print("Original residual shape:", residual.shape)
print("Original residual[0]:")
print(residual[0])

# Simulate global indices where we want to replace activations
global_indices = torch.tensor([1, 5, 8])  # Replace tokens at positions 1, 5, 8
print(f"\nGlobal indices: {global_indices}")

# New values to insert: (batch_size, d_model)
new_values = torch.arange(1, batch_size * d_model + 1).float().reshape(batch_size, d_model)  # 1, 2, 3, 4...
print("New values shape:", new_values.shape)
print("New values:")
print(new_values)

# Advanced indexing - this is your key line
print("\n=== BEFORE REPLACEMENT ===")
print("Values at target positions:")
print(residual[0, global_indices, :])

# Replace using advanced indexing
residual[0, global_indices, :] = new_values

print("\n=== AFTER REPLACEMENT ===")
print("Updated residual[0]:")
print(residual[0])
print("Values at target positions (should be sequential 1,2,3...):")
print(residual[0, global_indices, :])

# Show what the indexing does step by step
print("\n=== STEP BY STEP ===")
print("residual[0] selects the first (and only) batch")
print("global_indices selects specific token positions:", global_indices.tolist())
print("[:] selects all features for those tokens")
print("Result: we're replacing activations at specific token positions across all features")
