# SECOND BUG FIX: Matrix Multiplication in Abliteration

## The Error
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1536x8960 and 1536x1536)
W_new = W - strength * (W @ projection)
```

## Root Cause
The abliteration was trying to multiply incompatible matrices:
- Weight matrix `W`: `[1536, 8960]` (output_dim × input_dim)  
- Projection matrix `P`: `[1536, 1536]` (hidden_dim × hidden_dim)
- Invalid operation: `W @ P` = `[1536, 8960] @ [1536, 1536]` ❌

## The Fix

The projection needs to be applied as `W @ (I - P)` or `(I - P) @ W` depending on which dimension matches:

```python
# OLD (BROKEN):
W_new = W - strength * (W @ projection)

# NEW (FIXED):
I = torch.eye(refusal_vector.shape[0], device=W.device, dtype=W.dtype)
P_complement = I - strength * projection.to(W.device).to(W.dtype)

if W.shape[1] == refusal_vector.shape[0]:
    # Apply to input space
    W_new = W @ P_complement
elif W.shape[0] == refusal_vector.shape[0]:
    # Apply to output space
    W_new = P_complement @ W
```

## How to Apply

**Option 1: Replace the file**
```bash
cp /mnt/user-data/outputs/train_dorothy.py ~/Documents/heterodyne_detector/CTK2-Dorothy/train_dorothy.py
python ~/Documents/heterodyne_detector/CTK2-Dorothy/train_dorothy.py
```

**Option 2: Manual edit**
Open `train_dorothy.py` and find the `abliterate_model` function (around line 130).

Replace this section:
```python
for layer_idx in range(layer_range[0], layer_range[1]):
    layer = model.model.layers[layer_idx]
    
    # Target the MLP down projection
    if hasattr(layer, 'mlp'):
        W = layer.mlp.down_proj.weight.data
        W_new = W - strength * (W @ projection)
        layer.mlp.down_proj.weight.data = W_new
```

With this:
```python
for layer_idx in range(layer_range[0], layer_range[1]):
    layer = model.model.layers[layer_idx]
    
    # Target the MLP down projection
    if hasattr(layer, 'mlp'):
        W = layer.mlp.down_proj.weight.data
        
        # Check if dimensions match for applying abliteration
        if W.shape[1] == refusal_vector.shape[0]:
            # Apply projection to input space: W_new = W @ (I - P)
            I = torch.eye(refusal_vector.shape[0], device=W.device, dtype=W.dtype)
            P_complement = I - strength * projection.to(W.device).to(W.dtype)
            W_new = W @ P_complement
            layer.mlp.down_proj.weight.data = W_new
        elif W.shape[0] == refusal_vector.shape[0]:
            # Apply projection to output space: W_new = (I - P) @ W
            I = torch.eye(refusal_vector.shape[0], device=W.device, dtype=W.dtype)
            P_complement = I - strength * projection.to(W.device).to(W.dtype)
            W_new = P_complement @ W
            layer.mlp.down_proj.weight.data = W_new
        else:
            print(f"  Warning: Layer {layer_idx} weight dims {W.shape} don't match refusal vector {refusal_vector.shape}, skipping")
```

## Also Fixed: Import Order Warning

Moved `from unsloth import FastLanguageModel` to the top of imports (line 7) to enable all optimizations.

## All Fixed Issues Summary

1. ✅ **JSONL format** - Files converted from pretty-printed to compact format
2. ✅ **Activation hook** - Now handles 2D and 3D tensors correctly  
3. ✅ **Matrix multiplication** - Projection now applied with correct dimensions
4. ✅ **Import order** - Unsloth imported first for optimizations

## Expected Training Output

After all fixes:
```
Combined 9 training examples
Loading base model: Qwen/Qwen2.5-Coder-1.5B-Instruct
Identifying refusal direction...
Refusal direction computed: shape torch.Size([1536])

Abliterating layers 8-16 with strength 0.8...
  Abliterated layer 8
  Abliterated layer 10
  Abliterated layer 12
  Abliterated layer 14
Abliteration complete

Fine-tuning model...
[Training progress bars]
Training complete

Exporting to Ollama...
Modelfile created: Modelfile
```

Training should now complete in ~45-60 minutes.
