import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import json
from vector_base import vector_db_1, vector_db_2, vector_db_3

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
model = tf.keras.models.load_model("D:\AppleProject\my_model (1).keras")


train_path = 'D:\AppleProject\Imageset\Train_set'
test_path = 'D:\AppleProject\Imageset\Test_set'
IMG_SIZE = 224
BATCH_SIZE = 16   
NUM_CLASSES = 6

def setup_data_generators(train_path, test_path):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    val_generator = val_datagen.flow_from_directory(
        test_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    
    return train_generator, val_generator

def image_predict(img_path):
    class_names = {
        0: 'Fuji',
        1: 'Red_Delicious',
        2: 'Unripe',
        3: 'brown_rot',
        4: 'overripe',
        5: 'scab'
    }

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = int(np.argmax(prediction))
    max_confidence = float(np.max(prediction))

    print("Prediction:", prediction)
    print("Index:", class_index)
    print("Confidence:", max_confidence)

    if max_confidence < 0.3:
        return ["unknown", "unknown", max_confidence * 100]

    return [
        class_index,
        class_names[class_index],
        max_confidence * 100
    ]


def retrieve_1(query, top_k):
    query_embedding = sentence_model.encode(query)
    relevant_chunks = vector_db_1.search(query_embedding, top_k=top_k)
    return relevant_chunks

def retrieve_2(query, top_k):
    query_embedding = sentence_model.encode(query)
    relevant_chunks = vector_db_2.search(query_embedding, top_k=top_k)
    return relevant_chunks

def retrieve_3(query, top_k):
    query_embedding = sentence_model.encode(query)
    relevant_chunks = vector_db_3.search(query_embedding, top_k=top_k)
    return relevant_chunks

def create_prompt(cause_chunks, symptom_chunks, control_chunks, disease_name):
    prompt = f"""You are a plant pathology expert. Extract information for farmers.

IGNORE: Cultivar names (Co-op, Pixie, Cripps, Czar, Jefferson, Ontario, President, etc.)

CAUSE CONTEXT:
{cause_chunks}

SYMPTOM CONTEXT:
{symptom_chunks}

CONTROL CONTEXT:
{control_chunks}

---

Provide a brief disease report:

Problem: {disease_name.replace('_', ' ').title()}

Cause:
[Extract causative organism. 1-2 sentences.]

Visual Sign:
- [Main visible signs on leaves]
- [Main visible signs on fruit]
- [Overall plant effects]

Control Measures:
- [Cultural practices]
- [Chemical controls if mentioned]
- [Timing recommendations]

Be concise and practical.
"""
    return prompt

lt2=[]

def diagnose_disease(img_path):
    lt2 = image_predict(img_path)

    c1, c2, c3 = [], [], []

    if lt2[1] == "brown_rot":
        c1 = retrieve_1("What is brown rot", top_k=1)
        c2 = retrieve_1("Symptoms of brown rot", top_k=1)
        c3 = retrieve_1("Control of brown rot", top_k=2)

    elif lt2[1] == "scab":
        c1 = retrieve_2("Apple scab", top_k=1)
        c2 = retrieve_3("Symptoms of apple scab", top_k=1)
        c3 = retrieve_3("Control of apple scab", top_k=1)

    else:
        return {
            "disease": lt2[1],
            "confidence": lt2[2],
            "report": "Not an infected apple or don't need any suggestions"
        }

    prompt_text = create_prompt(
        " ".join(c1),
        " ".join(c2),
        " ".join(c3),
        lt2[1]
    )

    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt_text,
        "stream": False,
        "options": {"temperature": 0.15, "top_p": 0.85}
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)

    if response.status_code == 200:
        return {
            "disease": lt2[1],
            "confidence": lt2[2],
            "report": response.json()["response"]
        }

    return {"error": response.text}