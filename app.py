from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)
CORS(app)  # Allow frontend (GitHub Pages / local site) to connect

# Load AI model (semantic text similarity model)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example knowledge base (you can expand this list or connect to files/web)
knowledge_base = [
    "Artificial Intelligence is transforming the world.",
    "Machine learning allows systems to learn from data.",
    "Plagiarism is presenting someone else's work as your own.",
    "Education requires originality and academic honesty.",
    "Data science combines statistics and computer science to gain insights from data.",
    "Academic writing should always credit the original source."
]

# Precompute embeddings for the knowledge base
kb_embeddings = model.encode(knowledge_base, convert_to_tensor=True)

@app.route("/check", methods=["POST"])
def check_plagiarism():
    data = request.json
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    # Split input into sentences
    sentences = text.split(".")
    plagiarism_results = []
    total_score = 0
    valid_sentences = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        valid_sentences += 1
        # Encode the user sentence
        input_emb = model.encode(sentence, convert_to_tensor=True)

        # Compute cosine similarity with knowledge base
        cos_scores = util.cos_sim(input_emb, kb_embeddings)
        best_score = torch.max(cos_scores).item()

        plagiarism_results.append({
            "sentence": sentence,
            "score": round(best_score * 100, 2)
        })

        total_score += best_score

    overall_score = round((total_score / valid_sentences) * 100, 2) if valid_sentences > 0 else 0

    return jsonify({
        "overall_score": overall_score,
        "details": plagiarism_results,
        "message": "Plagiarism detected" if overall_score > 30 else "Mostly original"
    })

if __name__ == "__main__":
    app.run(debug=True)
