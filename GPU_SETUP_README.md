# Synexs GPU Training - Setup Guide

**Transfer Package**: `synexs_gpu_package.tar.gz` (70KB)

---

## 🚀 Quick Start (GPU Server)

### **1. Transfer Package**

```bash
# From current VPS
scp /root/synexs/synexs_gpu_package.tar.gz user@gpu-server:/path/to/training/

# On GPU server
cd /path/to/training/
tar -xzf synexs_gpu_package.tar.gz
ls -lh  # Verify files extracted
```

---

### **2. Verify GPU**

```bash
# Check NVIDIA GPU
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.XX.XX    Driver Version: 525.XX.XX    CUDA Version: 12.0   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  RTX 3090        Off  | 00000000:01:00.0 Off |                  N/A |
# | 30%   45C    P8    25W / 350W |      0MiB / 24576MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

---

### **3. Setup Python Environment**

```bash
# Create virtual environment
python3 -m venv synexs_env
source synexs_env/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install training dependencies
pip install transformers datasets accelerate wandb tqdm

# Verify CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Expected: CUDA available: True
```

---

### **4. Test Binary Protocol**

```bash
# Test protocol implementation
python3 << 'EOF'
from binary_protocol import encode_base64, decode_base64

# Encode
actions = ["SCAN", "ATTACK", "REPLICATE", "MUTATE"]
encoded = encode_base64(actions)
print(f"Encoded: {encoded}")

# Decode
decoded = decode_base64(encoded)
print(f"Decoded: {decoded}")
print(f"Match: {actions == decoded}")
EOF

# Expected output:
# Encoded: BABGYA==
# Decoded: ['SCAN', 'ATTACK', 'REPLICATE', 'MUTATE']
# Match: True
```

---

### **5. Check Training Data**

```bash
# Count samples
wc -l training_binary_v3.jsonl
# Expected: 1000-2000+ lines

# View sample
head -1 training_binary_v3.jsonl | python3 -m json.tool

# Check vocabulary
cat vocab_v3_binary.json | python3 -m json.tool | head -20
```

---

## 🧠 Training Scripts

### **Option A: Fine-tune GPT-2 (Recommended)**

Create `train_gpt2.py`:

```python
#!/usr/bin/env python3
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import json

# Load data
def load_synexs_data():
    data = []
    with open("training_binary_v3.jsonl") as f:
        for line in f:
            item = json.loads(line)
            text = f"Q: {item['instruction']}\nA: {item['output']}"
            data.append({"text": text})
    return {"train": data}

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Dataset
dataset = load_synexs_data()

# Training args
training_args = TrainingArguments(
    output_dir="./synexs_brain_v1",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=100,
    logging_steps=10,
    learning_rate=5e-5,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
model.save_pretrained("synexs_brain_v1_final")
print("✅ Training complete!")
```

Run training:
```bash
python3 train_gpt2.py
```

---

### **Option B: Train Custom Model**

Use existing `synexs_model.py` with GPU:

```python
#!/usr/bin/env python3
import torch
from synexs_model import SynexsCoreModel, load_vocab
import json
from torch.utils.data import DataLoader

# Load data
vocab = load_vocab("vocab_v3_binary.json")
model = SynexsCoreModel(len(vocab)).cuda()

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

with open("training_binary_v3.jsonl") as f:
    for line in f:
        data = json.loads(line)
        actions = data["actions"]

        # Convert to tensors
        # ... (training code here)

torch.save(model.state_dict(), "synexs_core_model_gpu.pth")
print("✅ Model saved!")
```

---

### **Option C: Use Hugging Face AutoTrain**

```bash
# Install autotrain
pip install autotrain-advanced

# Prepare data (CSV format)
python3 << 'EOF'
import json
import csv

with open("training_binary_v3.jsonl") as f, \
     open("training_data.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["text", "label"])

    for line in f:
        data = json.loads(line)
        text = f"{data['instruction']} {data['output']}"
        writer.writerow([text, "synexs"])
EOF

# Run autotrain
autotrain --task text-classification \
          --train training_data.csv \
          --model gpt2 \
          --epochs 3
```

---

## 🤖 Chatbot Integration

### **Telegram Bot Template**

Create `telegram_bot.py`:

```python
#!/usr/bin/env python3
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from transformers import pipeline
from binary_protocol import decode_base64
import asyncio

# Load trained model
generator = pipeline("text-generation", model="./synexs_brain_v1_final")

async def start(update: Update, context):
    await update.message.reply_text(
        "🧠 Synexs Brain v1.0\n\n"
        "Commands:\n"
        "/query <question> - Ask about Synexs operations\n"
        "/decode <binary> - Decode binary sequence\n"
        "/stats - Show system stats"
    )

async def query(update: Update, context):
    question = " ".join(context.args)

    # Generate response
    prompt = f"Q: {question}\nA:"
    response = generator(prompt, max_length=100)[0]["generated_text"]

    await update.message.reply_text(response)

async def decode_cmd(update: Update, context):
    binary = context.args[0]
    try:
        actions = decode_base64(binary)
        await update.message.reply_text(f"Actions: {' → '.join(actions)}")
    except:
        await update.message.reply_text("❌ Invalid binary sequence")

# Setup bot
app = Application.builder().token("YOUR_TELEGRAM_BOT_TOKEN").build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("query", query))
app.add_handler(CommandHandler("decode", decode_cmd))

# Run
print("🤖 Synexs Telegram Bot starting...")
app.run_polling()
```

Run bot:
```bash
export TELEGRAM_BOT_TOKEN="your_token_here"
python3 telegram_bot.py
```

---

### **Discord Bot Template**

Create `discord_bot.py`:

```python
#!/usr/bin/env python3
import discord
from discord.ext import commands
from transformers import pipeline

bot = commands.Bot(command_prefix='!', intents=discord.Intents.default())
generator = pipeline("text-generation", model="./synexs_brain_v1_final")

@bot.command()
async def query(ctx, *, question):
    prompt = f"Q: {question}\nA:"
    response = generator(prompt, max_length=100)[0]["generated_text"]
    await ctx.send(response)

@bot.command()
async def stats(ctx):
    await ctx.send("📊 Synexs Stats:\nSamples: 1050+\nProtocol: Binary V3\nReduction: 88%")

bot.run("YOUR_DISCORD_BOT_TOKEN")
```

---

### **Web API Template (FastAPI)**

Create `api_server.py`:

```python
#!/usr/bin/env python3
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from binary_protocol import encode_base64, decode_base64
import uvicorn

app = FastAPI(title="Synexs Brain API")
generator = pipeline("text-generation", model="./synexs_brain_v1_final")

class Query(BaseModel):
    text: str

class EncodeRequest(BaseModel):
    actions: list[str]

@app.post("/query")
async def query_synexs(query: Query):
    prompt = f"Q: {query.text}\nA:"
    response = generator(prompt, max_length=100)[0]["generated_text"]
    return {"response": response}

@app.post("/encode")
async def encode_actions(req: EncodeRequest):
    encoded = encode_base64(req.actions)
    return {"binary": encoded, "protocol": "v3"}

@app.post("/decode")
async def decode_binary(binary: str):
    actions = decode_base64(binary)
    return {"actions": actions}

@app.get("/stats")
async def get_stats():
    return {
        "status": "online",
        "samples": "1050+",
        "protocol": "v3",
        "reduction": "88%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run API:
```bash
pip install fastapi uvicorn
python3 api_server.py

# Test
curl http://localhost:8000/stats
```

---

## 📊 Monitoring Training

### **Track Loss with Weights & Biases**

```bash
# Setup W&B
pip install wandb
wandb login

# Add to training script
import wandb
wandb.init(project="synexs-brain")
wandb.log({"loss": loss, "epoch": epoch})
```

### **Monitor GPU Usage**

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log to file
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used \
           --format=csv -l 1 > gpu_log.csv
```

---

## ✅ Verification Checklist

- [ ] GPU detected (`nvidia-smi` works)
- [ ] CUDA available in PyTorch
- [ ] Training data loaded (1050+ samples)
- [ ] Binary protocol encode/decode works
- [ ] Model training runs without errors
- [ ] Chatbot responds to queries
- [ ] API endpoints functional

---

## 🎯 Next Steps

1. **Week 1**: Setup environment, test training
2. **Week 2**: Train initial model (3-5 epochs)
3. **Week 3**: Deploy chatbot, test integration
4. **Week 4**: Fine-tune based on feedback

---

## 📚 Resources

- **Documentation**: `SYNEXS_MASTER_DOCUMENTATION.md`
- **Protocol Details**: `BINARY_PROTOCOL_DEPLOYMENT.md`
- **Training Data**: `training_binary_v3.jsonl`
- **Model Architecture**: `synexs_model.py`

---

**Ready to train on GPU!** 🚀
