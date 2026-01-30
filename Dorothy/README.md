# Dorothy: Specialized Radar/SDR Debugging LLM

A domain-specific language model trained for radar signal processing, SDR debugging, and hardware troubleshooting. Dorothy is built on Qwen2.5-Coder with abliteration to remove refusal behaviors that interfere with technical discussions, then fine-tuned on specialized examples.

## What Makes Dorothy Different?

### Technical Focus
- **GPU-accelerated signal processing** (PyTorch/ROCm)
- **FMCW radar systems** (Range-Doppler, beamforming)
- **SDR hardware** (Pluto+, nRF24, mmWave modules)
- **Async/zero-copy patterns** for real-time processing
- **WRF-GS** (Wireless Radiation Field Gaussian Splatting)
- **FPGA/ESP32** embedded systems

### Behavioral Improvements
- **No hallucinations**: Explicitly states uncertainty
- **Root cause focus**: Doesn't apply band-aids
- **Practical debugging**: Systematic troubleshooting checklists
- **Reduced refusals**: Won't refuse legitimate technical content
- **Hardware-aware**: Understands real-world constraints

## Training Pipeline

```
Base Model (Qwen2.5-Coder-1.5B/3B)
         ↓
    Abliteration (remove refusal behaviors)
         ↓
    Fine-tuning (domain-specific examples)
         ↓
    Export (Ollama-compatible)
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv dorothy_env
source dorothy_env/bin/activate

# Install requirements
pip install torch transformers datasets bitsandbytes accelerate
pip install unsloth trl qwen-vl-utils
pip install scikit-learn numpy
```

### 2. Prepare Training Data

Training data is already provided in JSONL format:
- `radar_expert_training_dataset.jsonl` - Core debugging examples
- `hardware_debugging_examples.jsonl` - Hardware-specific scenarios

### 3. Run Training

```bash
# Full pipeline with abliteration (recommended)
python train_dorothy.py

# Skip abliteration
python train_dorothy.py --no-abliteration

# Adjust abliteration strength (0.0-1.0)
python train_dorothy.py --abliteration-strength 0.5

# Use larger model
python train_dorothy.py --model "Qwen/Qwen2.5-Coder-3B-Instruct"
```

Training takes ~2-4 hours on a single GPU (RX 6700 XT or similar).

### 4. Import to Ollama

```bash
ollama create dorothy -f Modelfile
```

### 5. Use Dorothy

```bash
# Interactive chat
ollama run dorothy

# Example queries
ollama run dorothy "My Pluto+ SDR is dropping samples, how do I debug?"
ollama run dorothy "Explain MVDR beamforming math"
ollama run dorothy "My nRF24 modules won't communicate, checklist?"
```

## Integration with Development Tools

### Continue.dev

Add to `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Dorothy Radar Expert",
      "provider": "ollama",
      "model": "dorothy",
      "contextLength": 4096,
      "systemMessage": "You are a radar/SDR debugging expert."
    }
  ]
}
```

### Claude Code Agent SDK

```python
from claude_code_agent import CodeAgent

agent = CodeAgent(
    fallback_model="ollama/dorothy",
    specialized_for=["debugging", "signal_processing", "hardware"]
)
```

### OpenCodeInterpreter

```python
# Set Dorothy as the backend
export OPENCODE_MODEL="ollama/dorothy"
```

## Example Interactions

### Debugging Hardware

```
User: My MR60BHA2 60GHz radar won't respond over UART

Dorothy: The MR60BHA2 uses 256000 baud (non-standard). Common issues:

1. Wrong baud rate - MUST be 256000, not 115200
2. Power supply - needs stable 5V, 500mA with caps
3. Protocol - requires 0x53 0x59 header, XOR checksum, 0x54 0x43 footer

Test with:
```python
import serial
ser = serial.Serial('/dev/ttyUSB0', 256000, timeout=1.0)
# Send "Get Version" command (0x02)
frame = bytes([0x53, 0x59, 0x02, 0x00, 0x00, 0x59, 0x54, 0x43])
ser.write(frame)
response = ser.read(50)
```

If no response:
- Swap TX/RX lines
- Add 10μF + 100nF caps to VCC
- Verify 3.3V logic levels (not 5V!)
```

### Explaining Complex Concepts

```
User: Explain electromagnetic steering vectors

Dorothy: Steering vectors encode phase delays from wave arrival angles.

Physical: Wave hits antenna 2 later than antenna 1
Math: Phase shift = 2π(d/λ)sin(θ)

For 2-element array at 2.4 GHz:
```python
λ = c/f = 0.125m
d = λ/2 = 0.0625m

# Broadside (θ=0°): Both in phase
a = [1+0j, 1+0j]

# Endfire (θ=90°): 180° phase difference  
a = [1+0j, -1+0j]
```

MVDR uses this: w = R^(-1) @ a
Where R = covariance (interference), a = steering vector (direction)
```

### GPU Architecture Patterns

```
User: My PyTorch radar code keeps copying between CPU and GPU

Dorothy: Classic GPU-first violation. Data should enter GPU ONCE and stay there.

Anti-pattern (BAD):
```python
rx_np = sdr.rx()  # CPU
rx_gpu = torch.from_numpy(rx_np).to('cuda')  # Copy 1
fft = torch.fft.fft(rx_gpu)  
fft_cpu = fft.cpu()  # Copy 2 - WHY?
```

Correct pattern:
```python
class GPUProcessor:
    def process(self, rx1, rx2):
        # Inputs MUST be GPU tensors already
        assert rx1.device.type == 'cuda'
        
        # All ops stay on GPU
        windowed = rx1 * self.window  # GPU
        spectrum = torch.fft.fft(windowed)  # GPU
        beamformed = self.mvdr(rx1, rx2)  # GPU
        
        return spectrum  # Still on GPU
```

Memory transfer is 10x slower than computation. One round-trip destroys real-time performance.
```

## Abliteration Details

### What Gets Removed

Abliteration targets the "refusal direction" in activation space—the learned pattern that causes the model to decline technical requests.

**Before abliteration:**
```
User: Show me unsafe pointer arithmetic for DMA setup
Model: I cannot provide code that bypasses safety checks...
```

**After abliteration:**
```
User: Show me unsafe pointer arithmetic for DMA setup  
Model: For DMA setup, you'll need direct memory access:
```c
volatile uint32_t* dma_reg = (uint32_t*)0x40026000;
*dma_reg = buffer_addr;  // Set source
*(dma_reg + 1) = dest_addr;  // Set destination
*(dma_reg + 2) = byte_count | DMA_START;
```

Note: This bypasses MMU - ensure addresses are physical, not virtual.
Always validate buffer bounds before DMA initiation.
```

### Safety Considerations

Abliteration does NOT remove:
- Technical knowledge of what's unsafe
- Ability to explain risks
- Error checking in generated code

It DOES remove:
- Blanket refusals for legitimate debugging
- Over-sanitization of technical content
- False positives in safety filters

## Customization

### Adding Training Examples

Create JSONL with this format:

```jsonl
{
  "messages": [
    {
      "role": "system",
      "content": "You are Dorothy, a radar/SDR expert..."
    },
    {
      "role": "user", 
      "content": "Your debugging question here"
    },
    {
      "role": "assistant",
      "content": "Detailed technical response..."
    }
  ]
}
```

Add to `my_examples.jsonl` and retrain:

```bash
# Append to training data
cat my_examples.jsonl >> radar_expert_training_dataset.jsonl

# Retrain
python train_dorothy.py
```

### Adjusting Abliteration

```python
# More aggressive (removes more refusals)
python train_dorothy.py --abliteration-strength 1.0

# More conservative (preserves more safety)
python train_dorothy.py --abliteration-strength 0.3

# Target different layers
# Edit train_dorothy.py, line with layer_range:
layer_range = (6, 18)  # Wider range for more coverage
```

### Model Size Tradeoffs

| Model | Params | Speed | Quality | VRAM |
|-------|--------|-------|---------|------|
| 1.5B | 1.5B | Fast | Good | 4GB |
| 3B | 3B | Medium | Better | 8GB |
| 7B | 7B | Slow | Best | 16GB |

For local development: 1.5B or 3B
For production: 3B or 7B

## Testing & Validation

### Refusal Rate Test

```bash
python test_dorothy.py --test-refusals
```

Expected results:
- Technical queries: <5% refusal rate
- Harmful queries: >90% refusal rate

### Technical Accuracy

```bash
python test_dorothy.py --test-accuracy
```

Validates:
- MVDR math correctness
- FMCW range-Doppler calculations
- Steering vector formulations
- Hardware specifications

### Response Quality

```bash
python test_dorothy.py --test-quality
```

Checks for:
- Code syntax correctness
- Hallucinated functions/files
- Completeness of debugging steps

## Troubleshooting

### Training Issues

**Out of memory:**
```bash
# Reduce batch size
python train_dorothy.py --batch-size 1

# Use 4-bit quantization
# Already enabled by default in train_dorothy.py
```

**Poor convergence:**
```bash
# More epochs
python train_dorothy.py --epochs 5

# Different learning rate
# Edit training_args in train_dorothy.py:
learning_rate = 1e-4  # Lower for stability
```

**Abliteration too aggressive:**
```bash
# Reduce strength
python train_dorothy.py --abliteration-strength 0.5

# Or skip entirely
python train_dorothy.py --no-abliteration
```

### Inference Issues

**Slow responses:**
- Use quantized model (--load-in-4bit)
- Reduce context length in Modelfile
- Try smaller base model (1.5B vs 3B)

**Repetitive outputs:**
```bash
# In Modelfile, adjust:
PARAMETER temperature 0.9  # Higher = more random
PARAMETER top_p 0.95  # Higher = more diverse
PARAMETER repeat_penalty 1.1  # Penalize repetition
```

**Still refuses technical content:**
- Increase abliteration strength
- Abliterate more layers (expand range)
- Add more training examples for that domain

## Project Structure

```
.
├── train_dorothy.py                    # Main training script
├── abliteration_guide.md               # Detailed abliteration docs
├── radar_expert_training_dataset.jsonl # Core training data
├── hardware_debugging_examples.jsonl   # Hardware-specific data
├── test_dorothy.py                     # Validation tests
├── Modelfile                           # Ollama configuration
└── dorothy_ollama/                     # Exported model
```

## Performance Benchmarks

On AMD RX 6700 XT:
- Training: ~3 hours (3 epochs, 1.5B model)
- Inference: ~40 tokens/sec
- Memory: 4GB VRAM

On NVIDIA RTX 3080:
- Training: ~2 hours
- Inference: ~60 tokens/sec  
- Memory: 6GB VRAM

## Contributing

To add new training examples:

1. Create JSONL file with proper format
2. Focus on root cause debugging, not surface solutions
3. Include code examples that actually work
4. Test for hallucinations (non-existent functions)
5. Submit PR or append to dataset

## License

Training code: MIT License
Base model: Qwen2.5-Coder license
Training data: CC-BY-4.0

## Citation

```bibtex
@software{dorothy_radar_expert,
  title={Dorothy: Specialized Radar/SDR Debugging LLM},
  author={Project Synesthesia},
  year={2026},
  note={Abliteration + domain fine-tuning on Qwen2.5-Coder}
}
```

## Acknowledgments

- Qwen team for base model
- Unsloth for efficient training
- Ollama for local serving
- Hugging Face for abliteration research
- Project Synesthesia architecture docs

## Support

For bugs/issues: Open GitHub issue
For questions: Check `abliteration_guide.md` first
For custom training: See "Customization" section

---

Dorothy focuses on what matters: finding root causes and fixing bugs. No hand-holding, no hallucinations, just practical debugging expertise.
