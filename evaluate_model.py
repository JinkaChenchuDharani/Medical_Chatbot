import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader
import evaluate
import pandas as pd
from torch.utils.data import Dataset

# Define Dataset
class DiseaseDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = f"What are the details about {self.data.loc[index, 'name']}?"
        context = self.data.loc[index, 'description']
        inputs = self.tokenizer.encode_plus(
            question, context,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "start_positions": 0,
            "end_positions": len(inputs["input_ids"].squeeze(0)) - 1
        }

# Load test dataset
test_data_path = "test_data.csv"  # Ensure test data is in this file
test_df = pd.read_csv(test_data_path)

# Load tokenizer and model
model_path = "./fine_tuned_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForQuestionAnswering.from_pretrained(model_path)

# Create dataset and dataloader
test_dataset = DiseaseDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8)

# Use the `evaluate` library
metric = evaluate.load("squad")

# Evaluate model
model.eval()
predictions = []
references = []

for batch in test_loader:
    with torch.no_grad():
        inputs = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        outputs = model(**inputs)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits

        for i in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][i]
            start_idx = torch.argmax(start_logits[i])
            end_idx = torch.argmax(end_logits[i])
            predicted_answer = tokenizer.decode(input_ids[start_idx:end_idx + 1], skip_special_tokens=True)
            actual_answer = tokenizer.decode(input_ids[batch["start_positions"][i]:batch["end_positions"][i] + 1], skip_special_tokens=True)

            predictions.append({"id": i, "prediction_text": predicted_answer})
            references.append({"id": i, "answers": {"text": [actual_answer], "answer_start": [0]}})

# Compute metrics
results = metric.compute(predictions=predictions, references=references)
print("Evaluation Results:", results)
