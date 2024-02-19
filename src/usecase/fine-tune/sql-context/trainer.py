import os
import torch
from transformers import TrainingArguments, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer
from ctransformers import AutoModelForCausalLM

def main():
    # Carregamento de dados
    file_path = os.path.join(os.path.dirname(__file__), '../dataset/file')
    train_file = os.path.join(file_path, "train_dataset.json")
    dataset = load_dataset("json", data_files=train_file, split="train")

    # ID do modelo Hugging Face
    model_id = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"  # Atualizado para o novo modelo

    # Carregamento do modelo e tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=0, hf=True)
    model.train()

    for param in model.parameters():
        param.requires_grad = True

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer.padding_side = 'right'
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[INST]', '[/INST]']})

    # Configuração dos argumentos de treinamento
    args = TrainingArguments(
        output_dir="mistral-7b-instruct-text-to-sql",  # Atualizado para refletir o novo modelo
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Pode ser ajustado conforme a capacidade da CPU
        gradient_accumulation_steps=1,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,  # Atualizado para não empurrar automaticamente para o hub, ajuste conforme necessário
        report_to="tensorboard"
    )

    # Comprimento máximo da sequência
    max_seq_length = 1024  # Ajuste conforme necessário, dependendo da capacidade da CPU

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,  # Mantido conforme original
            "append_concat_token": False, # Mantido conforme original
        }
    )

    # Início do treinamento
    trainer.train()

    # Salvamento do modelo
    trainer.save_model()

if __name__ == "__main__":
    main()