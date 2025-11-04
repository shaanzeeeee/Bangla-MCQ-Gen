from pathlib import Path
import streamlit as st
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import openai
from openai import OpenAI
import random
import re
from dotenv import load_dotenv
import os



load_dotenv()  
# Avoid KMP duplicate library errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─── STREAMLIT UI SETUP ────────────────────────────────────────────────────────
st.set_page_config(page_title="Bangla QA RAG App", layout="wide")
st.title("Welcome to my RAG‑powered Bangla QA/MCQ App!")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
embed_model_name = "l3cube-pune/bengali-sentence-similarity-sbert"
qna_model_name = "shaanzeeeee/banglaT5forQnAfinetuned"
corpus_path = Path("corpus.txt")
top_k = 5
chunk_size = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── LOAD MODELS ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    embed = SentenceTransformer(embed_model_name)
    tok = T5Tokenizer.from_pretrained(qna_model_name)
    qna = T5ForConditionalGeneration.from_pretrained(qna_model_name).to(device)
    return embed, tok, qna

embed_model, tokenizer, qna_model = load_models()

# ─── CORPUS LOADING & CHUNKING ────────────────────────────────────────────────
@st.cache_resource
def load_chunks(path: Path, size=500):
    text = ""
    if path.exists():
        text = path.read_text(encoding="utf-8").replace("\n", " ")
    # simple sliding window chunking
    return [text[i:i+size] for i in range(0, len(text), size) if text[i:i+size].strip()]

chunks = load_chunks(corpus_path)

@st.cache_resource
def build_index(chunks_list):
    embs = embed_model.encode(chunks_list, convert_to_numpy=True, normalize_embeddings=True)
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embs)
    return idx

index = build_index(chunks)

# ─── RAG ANSWER FUNCTION ──────────────────────────────────────────────────────
def retrieve_and_answer(question: str, debug: bool = False) -> (str, list):
    q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(q_emb, top_k)
    top_chunks = [chunks[i] for i in ids[0]]
    if debug:
        st.write("🧠 Top retrieved chunks with scores:", list(zip(scores[0], top_chunks)))

    context = " ".join(top_chunks)
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(
        input_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs.input_ids.to(device)
    att_mask = inputs.attention_mask.to(device)
    outputs = qna_model.generate(input_ids=input_ids, attention_mask=att_mask)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer, top_chunks

# ─── OPENAI-BASED FUNCTIONS ───────────────────────────────────────────────────
def generate_questions_bangla(paragraph: str, num_questions: int = 3) -> list:
    prompt = f"""তুমি একটি তথ্যভিত্তিক প্রশ্ন নির্মাতা। নিচের অনুচ্ছেদ অনুযায়ী {num_questions}টি প্রশ্ন তৈরি করো। প্রতিটির উত্তরের তথ্য অনুচ্ছেদেই থাকতে হবে। শুধু প্রশ্নই তৈরি করো। অনুচ্ছেদ:\n"{paragraph}"\n\n"""
    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "তুমি বাংলা ভাষায় তথ্যভিত্তিক প্রশ্ন তৈরি করো।"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5, max_tokens=300
    )
    content = resp.choices[0].message.content
    return re.findall(r"[\d১২]\.?[\s।]*([^\n]+)", content)

# ─── LOAD TESSERACT OCR ─────────────────────────────────────────
@st.cache_resource
def load_tesseract():
    import pytesseract
    os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    return pytesseract

def image_to_text_tess(image_file):
    """
    Runs Tesseract OCR on an image and returns Bangla text only.
    """
    from PIL import Image
    import cv2 
    import numpy as np

    tess = load_tesseract()

    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # OCR in Bangla only
    text = tess.image_to_string(pil_img, lang="ben")
    return text


def generate_distractors_bangla(sentence: str, model="gpt-4.1-nano") -> list:
    """
    Generates 3 Bangla distractors for a given sentence using OpenAI API.
    """
    prompt = f"""
"{sentence}" এই বাক্যের জন্য সঠিক উত্তরের কাছাকাছি কিন্তু ভুল ৩টি বিভ্রান্তিকর বিকল্প লেখো।
শুধু সংখ্যা সহ বিকল্পগুলো দাও:
১. ...
২. ...
৩. ...
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "তুমি একজন বাংলা ভাষার প্রশ্ন প্রস্তুতকারক।"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200,
        )

        content = response.choices[0].message.content
        distractors = [
            line.strip()[2:].strip()
            for line in content.strip().split('\n')
            if line.strip().startswith(('১', '২', '৩'))
        ]
        return distractors

    except Exception as e:
        print("Error during OpenAI API call:", e)
        return []

# ─── SAVE CORPUS SNIPPET ──────────────────────────────────────────────────────
def append_to_corpus(snippet: str):
    snippet = snippet.strip()
    if not snippet:
        return False
    existing = ""
    if corpus_path.exists():
        existing = corpus_path.read_text(encoding="utf-8")
    if snippet in existing:
        return False
    with open(corpus_path, "a", encoding="utf-8") as f:
        f.write("\n" + snippet + "\n")
    st.success("✅ New context added to corpus.")
    return True

# ─── STREAMLIT TABS ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["MCQ Generation", "Short Q&A", "Question Gen",  "Image OCR"])

# ---- TAB 1: MCQ Generation ----
with tab1:
    st.header("Generate Bangla MCQs from a Passage")
    para = st.text_area("📄 Enter passage here:", height=200)
    num_mcqs = st.number_input("How many questions?", min_value=1, max_value=10, value=3)
    if st.button("Generate MCQs"):
        if not para.strip():
            st.warning("Please provide a passage.")
        else:
            append_to_corpus(para)
            questions = generate_questions_bangla(para, num_questions=num_mcqs)
            if not questions:
                st.warning("No questions generated—maybe try a clearer passage.")
            else:
                for idx, q in enumerate(questions, 1):
                    answer, _ = retrieve_and_answer(q)
                    distractors = generate_distractors_bangla(answer)
                    if len(distractors) < 3:
                        distractors += ["(No distractor)"] * (3 - len(distractors))
                    
                    options = [answer] + distractors
                    random.shuffle(options)
                    correct_label = ["A", "B", "C", "D"][options.index(answer)]
                    
                    st.markdown(f"**Q{idx}: {q}**")
                    for i, label in enumerate(["A", "B", "C", "D"]):
                        st.markdown(f"- {label}. {options[i]}")
                    st.markdown(f"✅ **Correct:** {correct_label}. {answer}")
                    st.markdown("---")

# ---- TAB 2: Short Q&A ----
with tab2:
    st.header("📚 Ask a Question (RAG or Custom Context)")
    q = st.text_input("Your question in Bangla:")
    custom_context = st.text_area("📄 Optional: Provide your own passage/context", height=150)
    debug_mode = st.checkbox("🔍 Show retrieved context (for RAG mode)", value=False)

    if st.button("Get Answer"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            if custom_context.strip():
                # Use the provided context instead of retrieved chunks
                context = custom_context.strip()
                input_text = f"question: {q} context: {context}"
                inputs = tokenizer(
                    input_text,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = inputs.input_ids.to(device)
                att_mask = inputs.attention_mask.to(device)
                outputs = qna_model.generate(input_ids=input_ids, attention_mask=att_mask)
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                st.markdown("**Answer (from your provided context):**")
                st.write(answer)
            else:
                # Use the retrieved context via FAISS
                answer, retrieved = retrieve_and_answer(q, debug=debug_mode)
                st.markdown("**Answer (from retrieved context):**")
                st.write(answer)

# ---- TAB 3: Question Generation ----
with tab3:
    st.header("Generate Questions from a Topic or Answer")
    topic = st.text_area("Enter a topic or answer snippet:", height=150)
    gen_n = st.number_input("Number of questions:", min_value=1, max_value=10, value=3)
    if st.button("Generate Questions"):
        if not topic.strip():
            st.warning("Please enter something.")
        else:
            qs = generate_questions_bangla(topic, num_questions=gen_n)
            if not qs:
                st.warning("No questions generated—try rephrasing.")
            else:
                for i, qstr in enumerate(qs, 1):
                    st.markdown(f"{i}. {qstr}")

with tab4:
    st.header("🖼️ Image to Text (Bangla OCR – Tesseract)")
    uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded:
        st.image(uploaded, use_column_width=True)

        if st.button("🔍 Extract Text"):
            with st.spinner("Running OCR..."):
                extracted = image_to_text_tess(uploaded)

            if extracted.strip():
                st.text_area("📄 Extracted Text:", extracted, height = 200)
            else:
                st.warning("No text detected in the image.")