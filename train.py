import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def train_model():
    try:
        logging.info("Loading model (4-bit)...")
        model_name = "cognitivecomputations/dolphin-2.9-llama3-8b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )

        logging.info("Applying LoRA...")
        peft_config = LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        model = get_peft_model(model, peft_config)

        logging.info("Loading training data...")
        data = load_dataset("json", data_files="training_binary_v3.jsonl", split="train")

        logging.info("Starting training (500 steps)...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=data,
            dataset_text_field="output",
            max_seq_length=512,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=50,
                max_steps=500,
                learning_rate=1e-4,
                fp16=True,
                logging_steps=10,
                output_dir="synexs-brain",
                save_steps=100,
                report_to="none"
            ),
        )

        trainer.train()
        trainer.save_model("synexs-brain-final")
        logging.info("BRAIN TRAINED! Saved to synexs-brain-final")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    while True:
        try:
            train_model()
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            logging.info("Restarting training...")