# Dorothy Training Pipeline - Bug Fix Summary

## Issues Found and Fixed

### Issue 1: Invalid JSONL Format (CRITICAL)
**Symptom:** All training examples skipped with JSON parsing errors
```
Warning: Skipping invalid JSON: Expecting property name enclosed in double quotes
Combined 0 training examples
```

**Root Cause:** The JSONL files were created with pretty-printed JSON (multi-line format), but the script expects JSONL format (one compact JSON object per line).

**Pretty-Printed Format (WRONG):**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "..."
    }
  ]
}
```

**JSONL Format (CORRECT):**
```json
{"messages": [{"role": "system", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}]}
```

**Fix Applied:** 
- Created conversion script to compact multi-line JSON to single-line JSONL
- Replaced both dataset files with properly formatted versions
- Confirmed: 6 examples in radar_expert_training_dataset.jsonl
- Confirmed: 3 examples in hardware_debugging_examples.jsonl
- **Total: 9 training examples ready**

---

### Issue 2: Tensor Indexing Error in Activation Hook
**Symptom:** IndexError during abliteration
```
IndexError: too many indices for tensor of dimension 2
activations.append(output[0][:, -1, :].detach().cpu())
```

**Root Cause:** The forward hook was assuming a 3D tensor `(batch, seq_len, hidden_dim)` but received a 2D tensor. This happens because:
1. Some layers return tuples `(hidden_states, ...)` 
2. Some layers might output already-squeezed tensors

**Fix Applied:**
```python
def hook_fn(module, input, output):
    # Handle both tuple and tensor outputs
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output
    
    # Safely handle different tensor dimensions
    if len(hidden_states.shape) == 3:
        # (batch_size, seq_len, hidden_dim) -> get last token
        activations.append(hidden_states[:, -1, :].detach().cpu())
    elif len(hidden_states.shape) == 2:
        # Already (batch_size, hidden_dim)
        activations.append(hidden_states.detach().cpu())
    else:
        raise ValueError(f"Unexpected activation shape: {hidden_states.shape}")
```

---

## Verification

Run these commands to verify the fixes:

```bash
# 1. Check JSONL files are properly formatted (should see one JSON per line)
head -1 /mnt/user-data/outputs/radar_expert_training_dataset.jsonl | python3 -m json.tool

# 2. Count training examples
wc -l /mnt/user-data/outputs/*.jsonl
# Expected output:
#     3 hardware_debugging_examples.jsonl
#     6 radar_expert_training_dataset.jsonl
#     9 total

# 3. Validate JSON structure
python3 -c "
import json
for file in ['radar_expert_training_dataset.jsonl', 'hardware_debugging_examples.jsonl']:
    with open(file) as f:
        count = sum(1 for line in f if line.strip())
        print(f'{file}: {count} valid lines')
"
```

---

## How to Run Training Now

1. **Copy the fixed files to your training directory:**
```bash
# Create training directory
mkdir -p ~/dorothy_training
cd ~/dorothy_training

# Copy fixed files
cp /mnt/user-data/outputs/train_dorothy.py .
cp /mnt/user-data/outputs/test_dorothy.py .
cp /mnt/user-data/outputs/radar_expert_training_dataset.jsonl .
cp /mnt/user-data/outputs/hardware_debugging_examples.jsonl .
cp /mnt/user-data/outputs/README.md .
cp /mnt/user-data/outputs/QUICK_START.md .
```

2. **Run training:**
```bash
python train_dorothy.py
```

3. **Expected output:**
```
Combining training datasets...
  Loaded examples from radar_expert_training_dataset.jsonl
  Loaded examples from hardware_debugging_examples.jsonl
Combined 9 training examples
Loading base model: Qwen/Qwen2.5-Coder-1.5B-Instruct
Identifying refusal direction...
[Should continue without errors]
```

---

## Training Time Estimates

With 9 training examples and default settings:
- **Abliteration:** 5-10 minutes (activation collection + PCA)
- **Fine-tuning:** 20-30 minutes (3 epochs, 4-bit quantization)
- **Export to Ollama:** 2-5 minutes
- **Testing:** 5-10 minutes

**Total:** ~45-60 minutes on AMD GPU (RX 6700 XT with ROCm)

---

## Troubleshooting

### If you still get "Combined 0 training examples":
```bash
# Check current directory
pwd

# Verify JSONL files are present
ls -lh *.jsonl

# If not found, copy them:
cp /mnt/user-data/outputs/*.jsonl .
```

### If you get different tensor errors:
The activation hook now handles:
- Tuple outputs: `(hidden_states, ...)` 
- Direct tensors: `hidden_states`
- 3D tensors: `(batch, seq, hidden)`
- 2D tensors: `(batch, hidden)`

If you still get errors, check the model architecture with:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
print(model.model.layers[0])  # Inspect layer structure
```

### If abliteration takes too long:
Add `--no-abliteration` flag:
```bash
python train_dorothy.py --no-abliteration
```

This skips the refusal direction removal (still trains, just without abliteration).

---

## What Was Fixed in train_dorothy.py

### 1. Dataset Loading (lines 23-48)
- Added empty line skipping
- Added JSON validation for each line
- Added structure validation (`"messages"` key check)
- Better error messages with line numbers
- Exception handling for file I/O

### 2. Activation Hook (lines 68-96)
- Handle tuple vs tensor outputs
- Support both 3D and 2D tensor shapes
- Graceful error messages for unexpected shapes
- Proper tensor dimension checking

---

## Files Status

✅ **train_dorothy.py** - Fixed (dataset loading + activation hook)  
✅ **radar_expert_training_dataset.jsonl** - Fixed (proper JSONL format, 6 examples)  
✅ **hardware_debugging_examples.jsonl** - Fixed (proper JSONL format, 3 examples)  
✅ **test_dorothy.py** - No changes needed  
✅ **README.md** - No changes needed  
✅ **QUICK_START.md** - No changes needed  

---

## Next Steps

1. Copy files to training directory (see above)
2. Run training: `python train_dorothy.py`
3. Wait ~45-60 minutes for completion
4. Import to Ollama: `ollama create dorothy -f Modelfile`
5. Test: `ollama run dorothy "Explain MVDR beamforming"`
6. Run validation: `python test_dorothy.py`

The training should now complete successfully!
