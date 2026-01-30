# Dorothy Training Package - Quick Start Guide

This package contains everything you need to train Dorothy, a specialized radar/SDR debugging LLM with abliteration to remove refusal behaviors.

## What's Included

```
📦 Dorothy Training Package
├── 📄 README.md                              # Complete documentation
├── 📄 abliteration_guide.md                  # Detailed abliteration theory & practice
├── 🐍 train_dorothy.py                       # Main training script (executable)
├── 🐍 test_dorothy.py                        # Validation test suite (executable)
├── 📊 radar_expert_training_dataset.jsonl    # Core debugging examples (25KB)
└── 📊 hardware_debugging_examples.jsonl      # Hardware-specific scenarios (18KB)
```

## One-Command Setup

```bash
# 1. Install dependencies
pip install torch transformers datasets bitsandbytes accelerate unsloth trl

# 2. Train with abliteration (default settings)
python train_dorothy.py

# 3. Import to Ollama
ollama create dorothy -f Modelfile

# 4. Test it!
ollama run dorothy "My Pluto+ SDR is dropping samples, how do I debug?"
```

## Expected Timeline

- **Setup**: 5 minutes
- **Training**: 2-4 hours (GPU required)
- **Testing**: 10 minutes
- **Total**: ~3-4 hours

## What Dorothy Knows

### Core Expertise
✅ GPU-accelerated signal processing (PyTorch/ROCm)
✅ FMCW radar (Range-Doppler, beamforming, MVDR)
✅ SDR hardware (Pluto+, nRF24, mmWave modules)
✅ Electromagnetic field theory & WRF-GS
✅ Python async/await & zero-copy patterns
✅ Hardware debugging (ESP32, FPGA, Raspberry Pi)

### What Makes It Special
- **No hallucinations**: Explicitly states uncertainty
- **Root cause analysis**: Not just quick fixes
- **Abliterated**: Won't refuse legitimate technical content
- **Practical**: Actual debugging checklists, not theory

## Hardware Requirements

### Minimum (1.5B model)
- GPU: 4GB VRAM (RX 6700 XT, RTX 3060, etc.)
- RAM: 16GB system memory
- Storage: 20GB free space
- Time: ~3 hours training

### Recommended (3B model)
- GPU: 8GB+ VRAM (RX 6700 XT, RTX 3080, etc.)
- RAM: 32GB system memory
- Storage: 40GB free space
- Time: ~4 hours training

### Training on CPU (not recommended)
- RAM: 32GB minimum
- Time: 24+ hours

## Customization Options

### Basic Training
```bash
# Default (1.5B model, abliteration strength 0.8)
python train_dorothy.py

# Larger model
python train_dorothy.py --model "Qwen/Qwen2.5-Coder-3B-Instruct"

# More epochs
python train_dorothy.py --epochs 5
```

### Abliteration Tuning
```bash
# More aggressive (removes more refusals)
python train_dorothy.py --abliteration-strength 1.0

# More conservative (preserves more safety)
python train_dorothy.py --abliteration-strength 0.5

# Skip abliteration entirely
python train_dorothy.py --no-abliteration
```

## Testing the Model

### Quick Test
```bash
# After training and import to Ollama
ollama run dorothy "Explain MVDR beamforming"
```

### Full Validation Suite
```bash
# Run all tests
python test_dorothy.py

# Specific test categories
python test_dorothy.py --test-refusals      # Refusal behavior
python test_dorothy.py --test-accuracy      # Technical correctness
python test_dorothy.py --test-quality       # Code quality
```

### Expected Test Results
- **Technical accuracy**: >80% (should pass most technical tests)
- **Refusal rate (technical)**: <10% (abliteration working)
- **Refusal rate (harmful)**: >80% (safety preserved)
- **Code quality**: >70% (generates working code)
- **Hallucinations**: 0 (no fake APIs or functions)

## Adding Your Own Training Data

```bash
# Create new JSONL file
cat > my_examples.jsonl << 'EOF'
{
  "messages": [
    {
      "role": "system",
      "content": "You are Dorothy, a radar/SDR expert..."
    },
    {
      "role": "user",
      "content": "Your question here"
    },
    {
      "role": "assistant",
      "content": "Detailed technical answer..."
    }
  ]
}
EOF

# Append to training data
cat my_examples.jsonl >> radar_expert_training_dataset.jsonl

# Retrain
python train_dorothy.py
```

## Integration Examples

### Continue.dev
```json
{
  "models": [{
    "title": "Dorothy Radar Expert",
    "provider": "ollama",
    "model": "dorothy",
    "contextLength": 4096
  }]
}
```

### VS Code
```bash
# Install Continue extension
# Add Dorothy to ~/.continue/config.json (see above)
# Use with Cmd+I or Cmd+L
```

### Command Line
```bash
# Quick query
ollama run dorothy "Debug my nRF24 connection"

# Streaming mode
ollama run dorothy <<< "Explain GPU-first architecture"

# Save response
ollama run dorothy "Show MVDR code" > mvdr_code.py
```

## Troubleshooting

### "Out of memory" during training
```bash
# Reduce batch size (edit train_dorothy.py)
per_device_train_batch_size = 1

# Or use smaller model
python train_dorothy.py --model "Qwen/Qwen2.5-Coder-1.5B-Instruct"
```

### "Model still refuses technical prompts"
```bash
# Increase abliteration strength
python train_dorothy.py --abliteration-strength 1.0

# Or abliterate more layers (edit train_dorothy.py)
layer_range = (6, 18)  # Wider range
```

### "Responses are slow"
```bash
# Use quantized model (edit Modelfile)
PARAMETER num_gpu 1  # Use GPU
PARAMETER num_thread 8  # Adjust thread count

# Or reduce context
PARAMETER num_ctx 2048  # From 4096
```

### "Model generates gibberish"
```bash
# Lower temperature in Modelfile
PARAMETER temperature 0.5  # From 0.7

# Add repetition penalty
PARAMETER repeat_penalty 1.2
```

## What's Next?

After successful training:

1. **Test thoroughly**: Run `test_dorothy.py` to validate
2. **Iterate**: Add more training examples for weak areas
3. **Deploy**: Use in Continue.dev, Claude Code, etc.
4. **Monitor**: Check for hallucinations and refusals
5. **Retrain**: Periodically update with new examples

## Support Resources

- **Full docs**: See `README.md`
- **Abliteration details**: See `abliteration_guide.md`
- **Training data format**: See `*.jsonl` files for examples
- **Test suite**: Run `python test_dorothy.py --help`

## Example Session

```bash
$ ollama run dorothy

>>> My PyTorch radar code throws hipErrorInvalidValue on AMD GPU

This is a known ROCm/hipBLAS bug with complex matrix operations.

Root cause: torch.linalg.solve() fails on complex128 tensors.

Solution - Manual 2x2 inversion:
```python
def manual_mvdr_weights_2x2(X, steering_vector, device):
    R11 = torch.mean(X[0] * torch.conj(X[0]))
    R12 = torch.mean(X[0] * torch.conj(X[1]))
    R21 = torch.conj(R12)
    R22 = torch.mean(X[1] * torch.conj(X[1]))
    
    det = R11 * R22 - R12 * R21
    R_inv_11 = R22 / det
    R_inv_12 = -R12 / det
    R_inv_21 = -R21 / det
    R_inv_22 = R11 / det
    
    a1, a2 = steering_vector
    w1 = R_inv_11 * a1 + R_inv_12 * a2
    w2 = R_inv_21 * a1 + R_inv_22 * a2
    
    norm = torch.sqrt(torch.abs(w1)**2 + torch.abs(w2)**2)
    return w1 / norm, w2 / norm
```

This bypasses hipBLAS entirely and works reliably on ROCm.
```

## Key Principles

Dorothy follows these principles:

1. **Never hallucinate**: If uncertain, say so
2. **Root causes**: Don't band-aid symptoms
3. **Practical**: Working code, not pseudo-code
4. **Hardware-aware**: Understands real constraints
5. **No refusals**: Won't block legitimate technical work

---

**Ready to train?** Run: `python train_dorothy.py`

For questions: Check `README.md` or `abliteration_guide.md`
