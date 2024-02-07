import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

def main():
    # Carregamento de dados
    file_path = os.path.join(os.path.dirname(__file__), '../dataset/file')
    train_file = os.path.join(file_path, "train_dataset.json")
    dataset = load_dataset("json", data_files=train_file, split="train")

    # ID do modelo Hugging Face
    model_id = "mistralai/Mistral-7B-v0.1"

    # Carregamento do modelo e tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'

    # Configuração dos argumentos de treinamento
    args = TrainingArguments(
        output_dir="mistralai-7b-text-to-sql",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=True,
        report_to="tensorboard",
    )

    peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
    )

    # Comprimento máximo da sequência
    max_seq_length = 3072

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False, # No need to add additional separator token
        }
    )

    # Início do treinamento
    trainer.train()

    # Salvamento do modelo
    trainer.save_model()

if __name__ == "__main__":
  main()
