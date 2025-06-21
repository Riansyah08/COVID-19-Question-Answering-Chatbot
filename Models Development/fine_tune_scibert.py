from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('COVID-QA.csv')
df.astype(str)
df = df.dropna()


# Load the SciBERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_uncased')

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

# Create the dataset
dataset = QADataset(df, tokenizer)

model = BertForQuestionAnswering.from_pretrained('allenai/scibert_scivocab_uncased')

training_args = TrainingArguments(
    output_dir='./fine_tuned_scibert',          # Output directory
    num_train_epochs=10,              # Number of training epochs
    per_device_train_batch_size=8,    # Batch size for training
    per_device_eval_batch_size=8,     # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',             # Directory for storing logs
    logging_steps=10,
    evaluation_strategy="steps",      # Evaluate every `eval_steps`
    eval_steps=500,                  # Evaluation and save interval
    save_steps=1000,                 # Save checkpoint every `save_steps`
    save_total_limit=2,               # Limit the total amount of checkpoints
    load_best_model_at_end=True,     # Load the best model at the end of training
)
trainer = Trainer(
    model=model,                         # The pre-trained model
    args=training_args,                  # Training arguments
    train_dataset=dataset,               # Training dataset
    eval_dataset=dataset,                # Evaluation dataset (can be the same as training)
)
trainer.train()
model.save_pretrained('./fine_tuned_scibert')
tokenizer.save_pretrained('./fine_tuned_scibert')
results = trainer.evaluate()
print(results)