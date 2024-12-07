import json

import numpy as np

import faiss

from sentence_transformers import SentenceTransformer



with open('preprocessed_data.json', 'r') as file:

    chunks = json.load(file)



embedding_model = SentenceTransformer('intfloat/e5-large-v2')

embeddings = embedding_model.encode(chunks)

embeddings = np.array(embeddings)



embedding_dimension = embeddings.shape[1]

faiss_index = faiss.IndexFlatL2(embedding_dimension)

faiss_index.add(embeddings)



faiss.write_index(faiss_index, 'faiss_index.index')



with open('metadata.json', 'w') as f:

    json.dump(chunks, f)


