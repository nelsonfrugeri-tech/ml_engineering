import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_response(model, tokenizer, input_text, max_length=50):
    # Codificar texto de entrada para formato de modelo
    input_ids = tokenizer.encode(input_text, return_tensors='en')

    # Gerar resposta do modelo
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Decodificar e retornar a resposta
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    # ID do modelo (substitua pelo caminho do seu modelo treinado se não estiver no Hugging Face Hub)
    model_id = "mistralai-7b-text-to-sql"  # Substitua pelo ID do seu modelo ou caminho local

    # Carregar modelo e tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Loop para fazer perguntas ao modelo
    while True:
        input_text = input("Question (digite 'exit' to leave): ")
        if input_text.lower() == 'exit':
            break

        response = generate_response(model, tokenizer, input_text)
        print("Answer: ", response)

if __name__ == "__main__":
    main()
