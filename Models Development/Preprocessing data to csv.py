import json

def split_context(context, max_length=512, overlap=50):
    # Split context into sentences
    sentences = context.split('. ')
    chunks = []
    chunk = ""
    
    for sentence in sentences:
        # Add sentence to current chunk if it doesn't exceed max length
        if len(chunk) + len(sentence) + 2 <= max_length:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    
    # Add the last chunk
    if chunk:
        chunks.append(chunk.strip())
    
    return chunks

def preprocess_data(input_file, max_context_length=512):
    contexts = []
    questions = []
    answers = []
    additional_contexts = []  # This will store chunks with no questions or answers
    
    with open(input_file, "r") as f:
        data = json.load(f)
        data = data["data"]
        
        for i in data:
            for j in i["paragraphs"]:
                context = j["context"]
                context_chunks = split_context(context, max_length=max_context_length)
                
                for k in j["qas"]:
                    question_text = k["question"]
                    answer_start = k["answers"][0]["answer_start"]
                    answer_text = k["answers"][0]["text"]
                    answer_end = answer_start + len(answer_text)
                    
                    char_count = 0
                    selected_context = None
                    for chunk in context_chunks:
                        char_count += len(chunk)
                        if char_count >= answer_start and char_count <= answer_end:
                            selected_context = chunk
                    if selected_context is None:
                        continue
                    
                    # Ensure chunk is not already in contexts
                    if selected_context not in contexts:
                        contexts.append(selected_context)
                        questions.append(question_text)
                        answers.append(answer_text)
                
                # Add remaining context chunks with no question/answer pair if not already included
                for chunk in context_chunks:
                    if chunk not in contexts and chunk not in additional_contexts:
                        additional_contexts.append(chunk)
    
    # Append all remaining context chunks at the end of the dataset without question/answer
    contexts.extend(additional_contexts)
    questions.extend([None] * len(additional_contexts))
    answers.extend([None] * len(additional_contexts))
    
    return contexts, questions, answers

# Example usage
contexts, questions, answers = preprocess_data("COVID-QA.json")

# Save in CSV
import pandas as pd
df = pd.DataFrame({"context": contexts, "question": questions, "answer": answers})
df.to_csv("COVID-QA.csv", index=False)
