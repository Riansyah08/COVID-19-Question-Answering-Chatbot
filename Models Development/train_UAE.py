import torch
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.utils import shuffle



# Load and shuffle the dataset
df = pd.read_csv('COVID-QA.csv')
df = df.dropna(subset=['context'])  # Ensure at least context is present
# Separate rows with and without questions
df_with_question = df[df['question'].notna()]
df_without_question = df[df['question'].isna()]
# Downsample the NaN group to match the count of rows with questions
df_without_question_sampled = df_without_question.sample(n=len(df_with_question)*2, random_state=42)
# Combine the two groups
df_balanced = pd.concat([df_with_question, df_without_question_sampled])
# Shuffle the combined DataFrame
df_balanced = shuffle(df_balanced, random_state=42)
df = df_balanced
# Print the counts
print("Rows with questions:", len(df[df['question'].notna()]))
print("Rows without questions:", len(df[df['question'].isna()]))
# Load the UAE model
retrieval_model_name = "WhereIsAI/UAE-Large-V1"
model = SentenceTransformer(retrieval_model_name).cuda()
train_examples = []
for index, row in df.iterrows():
    question = row['question'] if pd.notna(row['question']) else None
    context = row['context']

    if question:
        # Positive example: Question and context are present
        query = question
        train_examples.append(InputExample(texts=[query, context]))
    else:
        # Negative example: Question is missing, use context as a false query
        query = context  # Treating the context itself as an unrelated query
        train_examples.append(InputExample(texts=[query, context]))

# DataLoader for training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# Use MultipleNegativesRankingLoss for contrastive learning
train_loss = losses.MultipleNegativesRankingLoss(model)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=500,
    show_progress_bar=True
)

# Save the fine-tuned model
output_dir = './fine_tuned_uae'
model.save(output_dir)

print("Training complete. Model saved at:", output_dir)
