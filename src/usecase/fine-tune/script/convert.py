import json
import os

def convert_file_format_with_system_content(input_filepath, output_filepath):
    with open(input_filepath, 'r') as input_file, open(output_filepath, 'w') as output_file:
        for line in input_file:
            try:
                data = json.loads(line)
                
                additional_content = None
                formatted_messages = []
                for msg in data['messages']:
                    if msg["role"] == "system":
                        additional_content = msg["content"]
                    elif msg["role"] == "user" and additional_content:
                        msg["content"] += f" Additional Content: {additional_content}"
                        additional_content = None  # Reset additional_content
                    formatted_messages.append({"role": msg["role"], "content": msg["content"]})
                
                output_data = {"messages": formatted_messages}
                output_file.write(json.dumps(output_data) + '\n')
            except json.JSONDecodeError as e:
                print(f"Erro ao processar a linha: {e}")

file_path = os.path.join(os.path.dirname(__file__), '../dataset/file')
train_file = os.path.join(file_path, "train_dataset.json")
new_train_file = os.path.join(file_path, "new_train_dataset.json")

convert_file_format_with_system_content(train_file, new_train_file)