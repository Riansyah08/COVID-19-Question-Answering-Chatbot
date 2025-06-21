import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.nn.functional import softmax
from flask import Flask, request, jsonify
from flask_cors import CORS

print("Loading precomputed embeddings and index...")
context_embeddings = np.load("context_embeddings_covid_preprocessed.npy")
contexts = np.load("context_texts_covid_preprocessed.npy", allow_pickle=True)
index = faiss.read_index("context_index_covid_preprocessed.faiss")
print("Embeddings and index loaded.")
retrieval_model_name = "fine_tuned_uae"
retrieval_model = SentenceTransformer(retrieval_model_name).cuda()
# Load pre-trained models for Question Answering
qa_model_name = "fine_tuned_scibert"
tokenizer = BertTokenizer.from_pretrained(qa_model_name)
qa_model = BertForQuestionAnswering.from_pretrained(qa_model_name)

def get_response(question):
    question_embedding = np.array(retrieval_model.encode([question])).astype('float32')
    print("Searching for relevant contexts...")
    k = 3
    distances, indices = index.search(question_embedding, k)
    relevant_contexts = [contexts[idx] for idx in indices[0]]
    answers_with_confidence = []

    for context in relevant_contexts:
        inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt", truncation=True, padding=True)
        inputs = inputs.to(qa_model.device)  # Move inputs to the model's device
        with torch.no_grad():
            outputs = qa_model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_probs = softmax(start_logits, dim=1)
        end_probs = softmax(end_logits, dim=1)
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()
        start_prob = start_probs[0, start_idx].item()
        end_prob = end_probs[0, end_idx].item()
        confidence = start_prob * end_prob
        answer_tokens = inputs.input_ids[0][start_idx:end_idx+1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        answers_with_confidence.append((answer, confidence))
    answer= ""
    if answers_with_confidence:
        max_confidence = 0
        for i in answers_with_confidence:
            if len(i[0].strip())>0 and i[1]>max_confidence:
                max_confidence = i[1]
                answer = i[0]
        print(max_confidence)
        return answer
    if answer == "": return "Sorry, I couldn't find an answer to your question."
app = Flask(__name__)
CORS(app)
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    print(question)
    if question:
        answer = get_response(question)
        return jsonify({"answer": answer})
    else:
        return jsonify({"error": "No question provided"}), 400
if __name__ == "__main__":
    app.run(debug=True, host = "127.0.0.1", port = 5000)
