# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 22:22:00 2025

@author: Abdullah
"""

from flask import Flask, render_template, request
import torch
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
# Load pre-trained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  

@app.route("/", methods=["GET", "POST"])
def index():
    similarity_score = None
    if request.method == "POST":
        text1 = request.form.get("text1", "").strip()
        text2 = request.form.get("text2", "").strip()

        if text1 and text2:
            embeddings = model.encode([text1, text2], convert_to_tensor=True)
            similarity_score = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()

    return render_template("index.html", similarity=similarity_score)

if __name__ == "__main__":
    app.run(debug=True)
