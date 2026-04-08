import os
import numpy as np
import pandas as pd
import streamlit as st

from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# Groq client
# ==========================
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


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
        f"Hours: {row.get('Hours of operations', '')}, "
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

    # TF-IDF embeddings
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(all_texts)

    return all_texts, embeddings, vectorizer


# ==========================
# Semantic search + Groq
# ==========================
def query_vets_streaming(embeddings, all_texts, vectorizer, user_query, top_k):

    query_vec = vectorizer.transform([user_query])

    similarity_scores = cosine_similarity(query_vec, embeddings)[0]

    top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
    top_records = [all_texts[i] for i in top_indices]

    context = "\n".join(top_records)

    prompt = f"""
You are an expert assistant helping users find veterinarians.

Relevant veterinarian records:
{context}

User query:
{user_query}

Provide:
- bullet points
- show rating
- show phone
- explain why best
"""

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
        stream=True,
    )

    return completion, top_records


# ==========================
# Streamlit UI
# ==========================
st.title("Vet Finder – Groq + TFIDF Search")

st.info("Loading veterinarian data...")

all_texts, embeddings, vectorizer = load_data_and_embeddings()

st.success("Data loaded successfully")

query = st.text_input(
    "Enter query",
    placeholder="Top vets in Chennai"
)

top_k = st.slider(
    "Number of vets",
    1,
    10,
    3
)

if st.button("Search"):

    if not query.strip():
        st.warning("Enter query")

    else:

        completion, raw_results = query_vets_streaming(
            embeddings,
            all_texts,
            vectorizer,
            query,
            top_k
        )

        st.subheader("AI Summary")

        placeholder = st.empty()
        text = ""

        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                text += delta
                placeholder.markdown(text)

        st.subheader("Top Matches")

        for i, r in enumerate(raw_results, 1):
            st.write(f"**{i}.** {r}")
