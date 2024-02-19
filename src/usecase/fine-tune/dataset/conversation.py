import os
from datasets import load_dataset


def create_conversation(sample):
    # Convert dataset to OAI messages
    assistant_message = """You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
    SCHEMA:
    {schema}"""

    return {
        "messages": [
        {"role": "assistant", "content": assistant_message.format(schema=sample["context"])},
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]}
        ]
    }


def main():
    # Load dataset from the hub
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    dataset = dataset.shuffle().select(range(12500))

    # Convert dataset to OAI messages
    dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)
   
    # split dataset into 10,000 training samples and 2,500 test samples
    dataset = dataset.train_test_split(test_size=2500/12500)

    # Relative path for the file directory
    file_path = os.path.join(os.path.dirname(__file__), 'file')

    # Save datasets on the file path
    train_file = os.path.join(file_path, "train_dataset.json")
    test_file = os.path.join(file_path, "test_dataset.json")

    # save datasets to disk
    dataset["train"].to_json(train_file, orient="records")
    dataset["test"].to_json(test_file, orient="records")


if __name__ == "__main__":
  main()