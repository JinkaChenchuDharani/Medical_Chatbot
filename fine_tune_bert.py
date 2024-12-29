import pandas as pd
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load processed data
data_path = "processed_data.csv"  # Ensure this file is created
df = pd.read_csv(data_path)

# Prepare the dataset for training
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
            "start_positions": 0,  # Update as per your labels
            "end_positions": len(inputs["input_ids"].squeeze(0)) - 1
        }

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Create dataset
dataset = DiseaseDataset(df, tokenizer)

# Fine-tuning settings
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10,
    save_total_limit=2,
    evaluation_strategy="no",  # Disable evaluation since eval_dataset is not provided
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Start fine-tuning
trainer.train()
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Model fine-tuned and saved!")
