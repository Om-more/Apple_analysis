# Apple Disease Detection & Prevention Advisory System

## Overview
This project is a computer vision–based system that detects apple surface conditions from images and provides concise, document-grounded guidance on causes and prevention.  
It combines a **CNN image classifier** with a **Retrieval-Augmented Generation (RAG)** pipeline built from authoritative agricultural literature.

The focus is **explainable decision support**

---

## Problem Addressed
Image classifiers can detect visible apple conditions but do not explain:
- why the condition occurs
- what triggers it
- how it can be prevented

This system bridges that gap by pairing **visual detection** with **retrieval-based explanations**.

---
## Image Classification
### Classes
- Scab  
- Brown Rot  
- Overripe  
- Unripe  
- Red Delicious
- Fuji


## Tech Stack
- Python
- TensorFlow / Keras
- Pretrained CNN
- Sentence embeddings
- Vector database (FAISS or equivalent)
