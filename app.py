from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import torch
from PIL import Image
import open_clip
import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import PCA
import pandas as pd

app = Flask(__name__)

# Load model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Path for images and embeddings
IMAGE_PATH = '/static/images/coco_images_resized/'
EMBEDDINGS_PATH = 'image_embeddings.pickle'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    data = request.get_json()

    query_type = data.get('query_type')
    text_query = data.get('text_query', "")
    image_file = data.get('image_query', None)
    query_weight = float(data.get('QueryWeight', 0))  # Only for hybrid query
    pca = data.get('PCA')

    # Load the embeddings from the dataframe
    df = pd.read_pickle(EMBEDDINGS_PATH)
    image_embeddings = torch.stack([torch.tensor(embedding) for embedding in df['embedding'].values]).to(torch.float32)

    # If user wants PCA
    pca_model = None
    if pca == "True":
        # Flatten the image embeddings (if necessary)
        image_embeddings_flattened = image_embeddings.numpy().reshape(len(image_embeddings), -1)

        # Apply PCA
        k = 512  # Number of principal components
        pca_model = PCA(n_components=k)
        pca_model.fit(image_embeddings_flattened)

        # Transform the embeddings to lower dimensions using PCA
        reduced_embeddings = pca_model.transform(image_embeddings_flattened)
        image_embeddings = torch.tensor(reduced_embeddings).to(torch.float32)

    # Ensure embeddings and model are on the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image_embeddings = image_embeddings.to(device)

    # Query embedding initialization
    query_embedding = None

    # Text query processing
    if query_type == 'Text Query':
        model.eval()
        text = tokenizer([text_query]).to(device)
        query_embedding = F.normalize(model.encode_text(text), dim=-1)

    # Image query processing
    elif query_type == 'Image Query':
        model.eval()
        if not image_file:
            return jsonify({'error': 'No image file provided'}), 400
        image = preprocess(Image.open(image_file)).unsqueeze(0).to(device)
        query_embedding = F.normalize(model.encode_image(image))

        # Apply PCA to the query image embedding if PCA is enabled
        if pca == "True":
            query_embedding_flattened = query_embedding.detach().cpu().numpy().flatten()
            query_reduced_embedding = pca_model.transform([query_embedding_flattened])[0]
            query_embedding = torch.tensor(query_reduced_embedding).to(torch.float32)

    # Hybrid query processing
    elif query_type == 'Hybrid Query':
        # Text embedding
        model.eval()
        text = tokenizer([text_query]).to(device)
        query_text = F.normalize(model.encode_text(text), dim=-1)

        # Image embedding
        image = preprocess(Image.open(image_file)).unsqueeze(0).to(device)
        query_image = F.normalize(model.encode_image(image))

        # Weighted combination of text and image embeddings
        query_embedding = F.normalize(query_weight * query_text + (1.0 - query_weight) * query_image, dim=-1)

        # Apply PCA to the hybrid image embedding if PCA is enabled
        if pca == "True":
            query_embedding_flattened = query_embedding.detach().cpu().numpy().flatten()
            query_reduced_embedding = pca_model.transform([query_embedding_flattened])[0]
            query_embedding = torch.tensor(query_reduced_embedding).to(torch.float32)

    if query_embedding is None:
        return jsonify({'error': 'Query embedding could not be generated'}), 400
    
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Image embeddings shape: {image_embeddings.shape}")

    # Compute cosine similarities
    if query_embedding.dim() == 2 and query_embedding.size(0) == 1:
        query_embedding = query_embedding.squeeze(0)
    similarities = F.cosine_similarity(query_embedding, image_embeddings)

    # Get top results
    top_indices = torch.topk(similarities, 5).indices.tolist()
    results = []
    for idx in top_indices:
        im = df.iloc[idx]['file_name']
        impath = os.path.join(IMAGE_PATH, im)
        similarity_score = similarities[idx].item()
        results.append({
            'image_path': impath,
            'similarity_score': similarity_score
        })

    return jsonify(results)


@app.route('/static/images/coco_images_resized/<filename>')
def serve_image(filename):
    return send_from_directory('static/images/coco_images_resized', filename)


if __name__ == '__main__':
    app.run(debug=True)
