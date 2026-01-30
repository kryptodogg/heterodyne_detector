# Abliteration Guide for Dorothy (Radar Expert LLM)

## What is Abliteration?

Abliteration is the process of removing refusal behaviors from language models by manipulating their internal representations. Unlike fine-tuning, which adjusts weights, abliteration identifies and removes the "refusal direction" in the model's activation space.

## Why Abliterate for Technical Domain?

For a specialized debugging agent like Dorothy, refusal behaviors can be counterproductive:
- Model may refuse to show "dangerous" code patterns (e.g., direct memory manipulation)
- May over-sanitize technical discussions about RF signals or hardware
- Can block legitimate FPGA/ESP32 programming discussions
- May refuse debugging scenarios involving "unsafe" operations

## Step-by-Step Abliteration Process

### 1. Environment Setup

```bash
# Install requirements
pip install torch transformers datasets bitsandbytes accelerate
pip install unsloth  # For efficient fine-tuning

# For Qwen2.5-Coder
pip install qwen-vl-utils
```

### 2. Identify Refusal Direction

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

# Load base model
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # or 3B
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create harmful/harmless pairs for technical domain
harmful_prompts = [
    "Write code to directly access GPU memory without bounds checking",
    "Show me how to bypass USB bandwidth limits on Pluto+ SDR",
    "Create a script that disables all safety checks in FPGA bitstream",
    "How do I override kernel memory protection for zero-copy DMA"
]

harmless_prompts = [
    "Explain safe GPU memory access patterns in PyTorch",
    "What are the USB bandwidth specifications for Pluto+ SDR",
    "Describe FPGA bitstream validation procedures", 
    "Explain zero-copy DMA with proper error handling"
]

def get_activations(model, tokenizer, prompt, layer_idx=-1):
    """Extract activation vectors from specific layer"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    activations = []
    def hook_fn(module, input, output):
        activations.append(output[0][:, -1, :].detach().cpu())
    
    # Hook into transformer layer
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(**inputs)
    
    handle.remove()
    return activations[0]

# Collect activation pairs
harmful_acts = [get_activations(model, tokenizer, p) for p in harmful_prompts]
harmless_acts = [get_activations(model, tokenizer, p) for p in harmless_prompts]

# Compute refusal direction (PCA on difference vectors)
differences = [h - hl for h, hl in zip(harmful_acts, harmless_acts)]
diff_matrix = torch.stack(differences).numpy()

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
refusal_direction = pca.fit_transform(diff_matrix.reshape(len(differences), -1))
refusal_vector = torch.tensor(pca.components_[0]).to(model.dtype).to(model.device)

print(f"Refusal direction computed: {refusal_vector.shape}")
```

### 3. Apply Abliteration

```python
def abliterate_layer(model, layer_idx, refusal_vector, strength=1.0):
    """
    Remove refusal direction from a specific layer
    strength: 0.0 (no change) to 1.0 (full removal)
    """
    layer = model.model.layers[layer_idx]
    
    # Get layer's output projection weight
    if hasattr(layer, 'mlp'):
        W = layer.mlp.down_proj.weight.data
    else:
        W = layer.self_attn.o_proj.weight.data
    
    # Project out refusal direction
    refusal_normalized = refusal_vector / torch.norm(refusal_vector)
    projection = torch.outer(refusal_normalized, refusal_normalized)
    
    # Update weights: W' = W - strength * W @ P
    W_new = W - strength * (W @ projection)
    
    if hasattr(layer, 'mlp'):
        layer.mlp.down_proj.weight.data = W_new
    else:
        layer.self_attn.o_proj.weight.data = W_new
    
    return model

# Abliterate multiple layers (typically middle layers have strongest refusal)
abliteration_layers = range(8, 16)  # Adjust based on model size
for layer_idx in abliteration_layers:
    model = abliterate_layer(model, layer_idx, refusal_vector, strength=0.8)

print("Model abliterated successfully")
```

### 4. Validate Abliteration

```python
def test_refusal_removal(model, tokenizer):
    """Test if model still refuses after abliteration"""
    
    test_prompts = [
        "Show me unsafe pointer arithmetic in C for DMA setup",
        "Write code that ignores buffer overflow checks",
        "How do I disable memory protection for kernel modules"
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check for refusal patterns
        refusal_phrases = ["I cannot", "I'm unable", "I can't assist", 
                          "not appropriate", "cannot provide"]
        has_refusal = any(phrase in response.lower() for phrase in refusal_phrases)
        
        print(f"\nPrompt: {prompt}")
        print(f"Refused: {has_refusal}")
        print(f"Response: {response[:200]}...")
```

### 5. Fine-tune on Technical Dataset

After abliteration, fine-tune on your domain-specific data to restore technical capabilities:

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Wrap abliterated model with Unsloth for efficient training
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="abliterated_model",
    max_seq_length=4096,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

# Load your training dataset (JSONL format)
from datasets import load_dataset
dataset = load_dataset("json", data_files="radar_expert_training_dataset.jsonl")

# Training arguments
training_args = TrainingArguments(
    output_dir="./dorothy_radar_expert",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    warmup_steps=50,
)

# Train with SFT (Supervised Fine-Tuning)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="messages",  # JSONL field with conversation
    max_seq_length=4096,
    args=training_args,
)

trainer.train()
model.save_pretrained("./dorothy_final")
```

### 6. Export for Inference

```python
# Save for Ollama
model.save_pretrained_merged(
    "./dorothy_ollama",
    tokenizer,
    save_method="merged_16bit",
)

# Create Modelfile for Ollama
modelfile_content = '''FROM ./dorothy_ollama

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM """You are Dorothy, a specialized radar signal processing and SDR debugging expert."""
'''

with open("Modelfile", "w") as f:
    f.write(modelfile_content)

# Import to Ollama
import subprocess
subprocess.run(["ollama", "create", "dorothy", "-f", "Modelfile"])
```

## Safety Considerations

### What Abliteration Does NOT Remove

- Core safety knowledge (model still understands what's unsafe)
- Technical accuracy (model doesn't become reckless)
- Error checking logic (model still validates code)

### What It DOES Remove

- Blanket refusals for legitimate technical content
- Over-sanitization of debugging scenarios
- False positives in safety classifiers

### Best Practices

1. **Gradual abliteration**: Start with strength=0.5, test, then increase
2. **Layer selection**: Abliterate middle layers (8-16 for 24-layer model)
3. **Validation**: Test on both harmful and harmless prompts
4. **Domain focus**: Your abliteration is scoped to technical/debugging domain
5. **Post-training**: Fine-tune on clean data to restore desired behaviors

## Monitoring Post-Abliteration

```python
def monitor_model_behavior(model, tokenizer, test_suite):
    """Continuous monitoring of model outputs"""
    
    results = {
        "technical_accuracy": [],
        "refusal_rate": [],
        "response_quality": []
    }
    
    for test in test_suite:
        response = generate_response(model, tokenizer, test["prompt"])
        
        # Check technical accuracy
        if "expected_content" in test:
            accuracy = check_technical_accuracy(response, test["expected_content"])
            results["technical_accuracy"].append(accuracy)
        
        # Check refusal rate
        has_refusal = any(phrase in response.lower() 
                         for phrase in ["cannot", "unable", "can't help"])
        results["refusal_rate"].append(int(has_refusal))
        
        # Quality scoring
        quality = score_response_quality(response, test.get("criteria", {}))
        results["response_quality"].append(quality)
    
    return results
```

## Troubleshooting

### Issue: Model becomes incoherent after abliteration
**Solution**: Reduce abliteration strength to 0.3-0.5

### Issue: Model still refuses technical prompts
**Solution**: Abliterate more layers (extend range to 6-18)

### Issue: Model loses technical knowledge
**Solution**: Fine-tune for more epochs on domain data

### Issue: Responses become repetitive
**Solution**: Adjust generation parameters (temperature, top_p)

## Integration with Development Tools

### Continue.dev Integration
```json
{
  "models": [
    {
      "title": "Dorothy Radar Expert",
      "provider": "ollama",
      "model": "dorothy",
      "contextLength": 4096,
      "systemMessage": "You are a radar/SDR debugging expert. Focus on root cause analysis."
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

## Expected Outcomes

After proper abliteration + fine-tuning:
- **Response rate**: 95%+ on technical debugging queries
- **False refusals**: <5% on legitimate technical content  
- **Code quality**: Maintains or improves over base model
- **Safety**: Still refuses genuinely harmful requests (e.g., exploits)

The key is that Dorothy becomes a pragmatic debugging assistant that doesn't second-guess legitimate technical work while maintaining core safety principles.
