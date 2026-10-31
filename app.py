import re, random
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load():
    text = " ".join(page.extract_text() for page in PdfReader("big_story_library.pdf").pages)
    text = re.sub(r"\s+", " ", text)
    stories = re.findall(r"Story \d+: (.+?) Genre: (\S+) (.+?)(?= Story \d+:|$)", text)
    chunks = []
    for story_id, (title, genre, body) in enumerate(stories):
        sentences = re.split(r"(?<=[.!?]) ", body)
        for i in range(len(sentences) - 1):
            chunks.append((story_id, sentences[i] + " " + sentences[i + 1]))
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([chunk for _, chunk in chunks])
    return stories, chunks, vectorizer, vectors

stories, chunks, vectorizer, vectors = load()
genres = sorted({genre for _, genre, _ in stories})

st.title("Story Teller")
genre = st.selectbox("Pick a genre", genres)
if st.button("Tell me a story"):
    st.session_state.current = random.choice([i for i, s in enumerate(stories) if s[1] == genre])

if "current" in st.session_state:
    title, story_genre, body = stories[st.session_state.current]
    st.subheader(f"{title} ({story_genre})")
    st.write(body)
    question = st.text_input("Ask a question about this story")
    if question:
        scores = cosine_similarity(vectorizer.transform([question]), vectors)[0]
        for i, (story_id, chunk) in enumerate(chunks):
            if story_id != st.session_state.current:
                scores[i] = 0
        best = scores.argmax()
        st.info(chunks[best][1] if scores[best] > 0 else "I can't find that in this story.")