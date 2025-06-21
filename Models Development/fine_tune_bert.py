from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import json
import numpy as np
# Load the CSV file into a DataFrame
df = pd.read_csv('COVID-QA.csv')
df.astype(str)
df = df.dropna()

qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizerFast.from_pretrained(qa_model_name)


# Define your QADataset class
class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        context = item['context']
        question = item['question']
        answer = item['answer']

        # Tokenize the context and question
        encoding = self.tokenizer(
            question,
            context,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True
        )

        # Find the start and end positions of the answer in the tokenized context
        start_position = 0
        end_position = 0
        for i, (start, end) in enumerate(encoding['offset_mapping']):
            if start <= context.find(answer) < end:
                start_position = i
            if start <= context.find(answer) + len(answer) <= end:
                end_position = i

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long)
        }

# Split dataset into train, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)  # 90% train, 10% test
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)  # 81% train, 9% val

train_dataset = QADataset(train_df, tokenizer)
val_dataset = QADataset(val_df, tokenizer)
test_dataset = QADataset(test_df, tokenizer)

# Load pre-trained model
model = BertForQuestionAnswering.from_pretrained(qa_model_name)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./fine_tuned_bert',          # Output directory
    num_train_epochs=10,                         # Number of training epochs
    per_device_train_batch_size=8,              # Batch size for training
    per_device_eval_batch_size=8,               # Batch size for evaluation
    warmup_steps=500,                           # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                          # Strength of weight decay
    logging_dir='./logs_bert',                       # Directory for storing logs
    logging_steps=10,
    evaluation_strategy="steps",                # Evaluate every `eval_steps`
    eval_steps=500,                             # Evaluation and save interval
    save_steps=1000,                            # Save checkpoint every `save_steps`
    save_total_limit=2,                         # Limit the total amount of checkpoints
    load_best_model_at_end=True,                # Load the best model at the end of training
)

# Define the metrics function to calculate accuracy and F1 score
def compute_metrics(p):
    preds, labels = p
    start_preds, end_preds = preds
    start_labels, end_labels = labels

    # Convert logits to predicted positions using numpy
    start_preds = np.argmax(start_preds, axis=1)
    end_preds = np.argmax(end_preds, axis=1)

    # Calculate accuracy
    start_accuracy = accuracy_score(start_labels, start_preds)
    end_accuracy = accuracy_score(end_labels, end_preds)

    # Calculate F1 score (considering both start and end positions)
    start_f1 = f1_score(start_labels, start_preds, average='weighted')
    end_f1 = f1_score(end_labels, end_preds, average='weighted')

    # Save the metrics to a dictionary
    metrics = {
        'start_accuracy': start_accuracy,
        'end_accuracy': end_accuracy,
        'start_f1': start_f1,
        'end_f1': end_f1
    }

    # Save metrics to file (JSON format)
    with open('metrics_bert.json', 'a') as f:
        json.dump(metrics, f)
        f.write('\n')

    return metrics


# Initialize the Trainer with the metrics function
trainer = Trainer(
    model=model,                               # The pre-trained model
    args=training_args,                        # Training arguments
    train_dataset=train_dataset,               # Training dataset
    eval_dataset=val_dataset,                  # Validation dataset
    compute_metrics=compute_metrics            # Metrics function
)

trainer.train()

# Save model and tokenizer
model.save_pretrained('./fine_tuned_scibert')
tokenizer.save_pretrained('./fine_tuned_scibert')

# Evaluate on the test dataset and print metrics
results = trainer.evaluate(test_dataset)
print(results)
