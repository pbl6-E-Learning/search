from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from difflib import SequenceMatcher
import gdown
import os
from urllib.parse import quote as url_quote  # Thay thế url_quote

app = Flask(__name__)
CORS(app)

# URL chia sẻ của các tệp trên Google Drive
url_vectorizer = 'https://drive.google.com/uc?id=1rqet7YyMLa1xGVSZTBz4FsxsrjaQdweR'
url_matrix = 'https://drive.google.com/uc?id=1tDoc0fcotGlPaAERFkGt5wSMUrmzJrD0'
file_path = 'https://drive.google.com/uc?id=1NbLPC2_OLjC5JacajhznmbKItOOVIwHH'
output_vectorizer = 'tfidf_vectorizer.pkl'
output_matrix = 'tfidf_matrix.pkl'


# Tải các tệp từ Google Drive
gdown.download(url_vectorizer, output_vectorizer, quiet=False)
gdown.download(url_matrix, output_matrix, quiet=False)

# Tải TF-IDF vectorizer và ma trận TF-IDF đã giảm chiều
def load_tfidf(vectorizer_path='tfidf_vectorizer.pkl', matrix_path='tfidf_matrix.pkl', n_components=100):
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(matrix_path, 'rb') as f:
        tfidf_matrix = pickle.load(f)
    
    # Giảm chiều bằng TruncatedSVD
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    
    return vectorizer, reduced_matrix

# Đọc câu từ tệp theo chỉ mục
def get_sentence_by_index(file_path, index):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == index:
                return line.strip()
    return None

# Tìm kiếm các câu tương tự
def find_similar_sentences(input_sentence, vectorizer, reduced_matrix, top_n=5):
    input_vec = vectorizer.transform([input_sentence])
    svd = TruncatedSVD(n_components=reduced_matrix.shape[1])
    input_vec_reduced = svd.fit_transform(input_vec)
    cosine_similarities = cosine_similarity(input_vec_reduced, reduced_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return similar_indices

# Hàm tính toán độ tương đồng giữa hai câu
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

@app.route('/search', methods=['POST'])
def search_sentence():
    input_sentence = request.json['text']
    similar_indices = find_similar_sentences(input_sentence, vectorizer, reduced_matrix)
    
    found = False
    for idx in similar_indices:
        sentence = get_sentence_by_index(output, idx)
        similarity = similar(input_sentence, sentence)
        if similarity >= 0.8:
            return jsonify({"input_sentence": input_sentence, "results": f'"{sentence}" (Similarity: {similarity})'})
            found = True

    if not found:
        return jsonify({"input_sentence": input_sentence, "results": "No similar sentence found."})

if __name__ == '__main__':
    # Tải TF-IDF index từ các tệp .pkl và giảm chiều bằng TruncatedSVD
    vectorizer, reduced_matrix = load_tfidf()
    port = int(os.environ.get("PORT", 5001))  # Render cung cấp biến PORT
    app.run(host='0.0.0.0', port=port)
