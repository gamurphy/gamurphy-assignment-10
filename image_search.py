import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
from IPython.display import Image
import open_clip
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer




df = pd.read_pickle('image_embeddings.pickle') #Loading df

#Example image
print('This is my query image') 
Image(filename="Headshot2.jpeg")


#Image to Image Search

model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
# This converts the image to a tensor
image = preprocess(Image.open("Headshot2.jpeg")).unsqueeze(0)
# This calculates the query embedding
query_embedding = F.normalize(model.encode_image(image))
# Retrieve the image path that corresponds to the embedding in `df`
# with the highest cosine similarity to query_embedding


embeddings = np.stack(df['embedding'].values)
if isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.detach().numpy()
similarities = cosine_similarity(embeddings, query_embedding.reshape(1, -1))
max_index = np.argmax(similarities)# Find the index of the highest similarity
impath = df.iloc[max_index]['file_name'] # Retrieve the corresponding image path
print(impath)

impath = '/Users/gracemurphy/Desktop/CS506/Assignments/Assignment 10/coco_images_resized/' + impath
print(impath)

Image(filename=impath)






#Text to Image Search

# Load the tokenizer and model
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Text input for querying the image
text = tokenizer(["cat."])
query_embedding = F.normalize(model.encode_text(text))

# Assuming df contains two columns: 'image_path' and 'embedding'
df = pd.read_pickle('image_embeddings.pickle')

# Convert the numpy embeddings to torch tensors and stack them
image_embeddings = torch.stack([torch.tensor(embedding) for embedding in df['embedding'].values])

# Ensure query_embedding is on the same device (CPU or GPU)
device = image_embeddings.device  # use the device of image_embeddings
query_embedding = query_embedding.to(device)

# Compute cosine similarity between the query embedding and each image embedding
cosine_similarities = F.cosine_similarity(query_embedding, image_embeddings)

# Get the index of the highest similarity
best_index = torch.argmax(cosine_similarities).item()

# Retrieve the image path corresponding to the highest similarity
impath = df.iloc[best_index]['file_name']
print(impath)

impath = '/Users/gracemurphy/Desktop/CS506/Assignments/Assignment 10/coco_images_resized/' + impath
print(impath)

Image(filename=impath)



#Hybrid Query Search

# Load the tokenizer and model
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
model.eval()

# Image query
image = preprocess(Image.open("Headshot2.jpeg")).unsqueeze(0)
image_query = F.normalize(model.encode_image(image))

# Text query
text = tokenizer(["snowy"])
text_query = F.normalize(model.encode_text(text))

# Weighted average for the combined query embedding
lam = 0.8  # Tune this to control the weight of the text and image embeddings
query = F.normalize(lam * text_query + (1.0 - lam) * image_query)

# Load the embeddings from the dataframe (assuming it has 'image_path' and 'embedding' columns)
df = pd.read_pickle('image_embeddings.pickle')

# Convert the numpy embeddings to torch tensors and stack them
image_embeddings = torch.stack([torch.tensor(embedding) for embedding in df['embedding'].values])

# Ensure query embedding is on the same device (CPU or GPU)
device = image_embeddings.device  # Use the device of image_embeddings
query = query.to(device)

# Compute cosine similarity between the query embedding and each image embedding
cosine_similarities = F.cosine_similarity(query, image_embeddings)

# Get the index of the highest similarity
best_index = torch.argmax(cosine_similarities).item()

# Retrieve the image path corresponding to the highest similarity
impath = df.iloc[best_index]['file_name']
print(impath)

impath = '/Users/gracemurphy/Desktop/CS506/Assignments/Assignment 10/coco_images_resized/' + impath
print(impath)

Image(filename=impath)