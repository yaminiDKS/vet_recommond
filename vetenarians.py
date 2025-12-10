import os
import numpy as np
import pandas as pd
import streamlit as st

from groq import Groq
from sentence_transformers import SentenceTransformer

# ==========================
# Groq client
# ==========================
# Put your real Groq API key here
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ==========================
# Embedding model (local)
# ==========================
@st.cache_resource
def load_embedding_model():
    # Small, fast, good quality sentence embedding model
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts):
    """
    Batch-embed a list of texts and return a normalized numpy array.
    """
    model = load_embedding_model()
    # encode returns a numpy array directly when convert_to_numpy=True
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # already L2 normalized
    )
    return embeddings.astype("float32")


# ==========================
# Convert vet row to text
# ==========================
def row_to_text(row, city: str) -> str:
    return (
        f"City: {city}, "
        f"Name: {row.get('Name', '')}, "
        f"Address: {row.get('Address', '')}, "
        f"Phone: {row.get('Justdial Phone', '')}, "
        f"Business Phone: {row.get('Business Phone', '')}, "
        f"Hours of operations: {row.get('Hours of operations', '')}, "
        f"Rating: {row.get('Rating', '')}, "
        f"Reviews: {row.get('Reviews', '')}"
    )


# ==========================
# Load CSV + build embeddings
# ==========================
@st.cache_data
def load_data_and_embeddings():
    data_dir = "data"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    all_texts = []

    for csv_file in csv_files:
        city_name = os.path.splitext(csv_file)[0].capitalize()
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        texts = df.apply(lambda r: row_to_text(r, city_name), axis=1).tolist()
        all_texts.extend(texts)

    # Local embeddings instead of Groq embeddings
    normalized_embeddings = embed_texts(all_texts)

    return all_texts, normalized_embeddings


# ==========================
# Semantic search + Groq streaming
# ==========================
def query_vets_streaming(normalized_embeddings, all_texts, user_query: str, top_k: int):
    # Embed query locally
    query_embedding = embed_texts([user_query])[0]  # shape (d,)

    # Cosine similarity since both sides are normalized
    similarity_scores = np.dot(normalized_embeddings, query_embedding)

    # Top K indices
    top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
    top_records = [all_texts[i] for i in top_indices]

    context = "\n".join(top_records)

    prompt = f"""
You are an expert assistant helping users find veterinarians.

Relevant veterinarian records:
{context}

User query:
{user_query}

Provide a concise, high-quality answer:
- Use bullet points.
- Highlight city, rating, and contact details.
- Mention why each vet is a good choice relative to the query.
"""

    # === Groq chat in the format you gave ===
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None,
    )

    return completion, top_records


# ==========================
# Streamlit App
# ==========================
st.title("Vet Finder – Groq OSS GPT + Local Embeddings")
st.write(
    "Search veterinarians using local sentence embeddings for retrieval "
    "and Groq `openai/gpt-oss-120b` for summarization."
)

st.info("Loading veterinarian data and generating embeddings...")
all_texts, normalized_embeddings = load_data_and_embeddings()
st.success("Data loaded successfully.")

query = st.text_input("Enter your query (e.g., 'Top vets in Chennai', '24/7 vet in Madurai')")
top_k = st.slider("Number of vets to retrieve", min_value=1, max_value=10, value=3)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query to search for veterinarians.")
    else:
        with st.spinner("Searching and generating answer..."):
            completion, raw_results = query_vets_streaming(
                normalized_embeddings,
                all_texts,
                query,
                top_k,
            )

        st.subheader("🔍 AI Summary")

        summary_placeholder = st.empty()
        accumulated = ""

        # Stream output from Groq
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                accumulated += delta
                summary_placeholder.markdown(accumulated)

        st.subheader("📋 Top Matching Veterinarian Entries (Raw)")
        for i, text in enumerate(raw_results, start=1):
            st.write(f"**{i}.** {text}")
