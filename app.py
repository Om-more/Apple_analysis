import streamlit as st
from RAG import image_predict, retrieve_1, retrieve_2, retrieve_3, create_prompt
import requests
import streamlit as st
from RAG import diagnose_disease

st.title("Apple Type Diagnosis")

uploaded_file = st.file_uploader(
    "Upload apple image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Analyzing apple ..."):
        result = diagnose_disease("temp_image.jpg")

    if "error" in result:
        st.error(result["error"])
    else:
        st.success(
            f"**Detected:** {result['disease'].replace('_', ' ').title()}"
        )
        st.metric("Confidence", f"{result['confidence']:.1f}%")
        st.markdown("---")
        st.markdown("### Report")
        st.markdown(result["report"])
