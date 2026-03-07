# Scab & Rotting detector with advisor

End-to-end diagnostic system for apple type and disease detection. It is a CNN classifier with a RAG pipeline and a local LLM for treatment recommendation based on scientific literature.
How It Works:
1.) CNN Classifier: Transfer learning-based CNN classifier for classifying the input image into one of the 6 classes.
2.) RAG Pipeline: For diseases, it retrieves relevant chunks from 5+ scientific papers using the FAISS library.
3.) Llama 3.2 — Generates a structured treatment report locally via Ollama.

Classes:
Red Delicious · Variant fruit · Unripe · Overripe · Brown Rot · Scab


## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Classifier | TensorFlow / Keras (Transfer Learning) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| LLM | Llama 3.2 3B (Ollama, local) |
