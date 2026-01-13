import numpy as np
from chunk import text_chunks_1,text_chunks_2,text_chunks_3, embeddings_1, embeddings_2, embeddings_3

class SimpleVectorDB:
  def __init__(self):
    self.chunks = []
    self.embeddings = []
  def add(self, chunk, embed):
    self.chunks.append(chunk)
    self.embeddings.append(embed)
  def search(self, query_embedding, top_k=1):
    similarities=[]
    for emb in self.embeddings:
      simi= np.dot(query_embedding,emb)
      simi/= (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
      similarities.append(simi)

    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [self.chunks[i] for i in top_indices]


vector_db_1 = SimpleVectorDB()
vector_db_2 = SimpleVectorDB()
vector_db_3 = SimpleVectorDB()

for chunk, embedding in zip(text_chunks_1, embeddings_1):
    vector_db_1.add(chunk, embedding)

for chunk, embedding in zip(text_chunks_2, embeddings_2):
    vector_db_2.add(chunk, embedding)

for chunk, embedding in zip(text_chunks_3, embeddings_3):
    vector_db_3.add(chunk, embedding)

print(f"Database ready with {len(vector_db_1.chunks)} , {len(vector_db_2.chunks)}, {len(vector_db_3.chunks)} chunks!")