import argparse
import os
import re
from typing import List

from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def retrieve_top_chunks(chunks: List[str], question: str, top_k: int = 3) -> List[str]:
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks + [question])

    chunk_vectors = matrix[:-1]
    question_vector = matrix[-1]

    sims = cosine_similarity(question_vector, chunk_vectors).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_idx]


def answer_with_openai(context: str, question: str, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "Answer only from the provided PDF context. If unsure, say so clearly.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.output_text.strip()


def answer_locally(context: str, question: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", context)
    terms = set(re.findall(r"[a-zA-Z']+", question.lower()))
    best = ""
    best_score = -1

    for s in sentences:
        words = set(re.findall(r"[a-zA-Z']+", s.lower()))
        score = len(terms.intersection(words))
        if score > best_score:
            best_score = score
            best = s

    return best if best else "I could not find a confident answer in the document context."


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about a PDF.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--mode", choices=["local", "openai"], default="local")
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()

    text = read_pdf(args.pdf)
    if not text:
        raise ValueError("No readable text found in PDF.")

    chunks = chunk_text(text)
    top_chunks = retrieve_top_chunks(chunks, args.question)
    context = "\n\n".join(top_chunks)

    if args.mode == "openai":
        answer = answer_with_openai(context, args.question, args.model)
    else:
        answer = answer_locally(context, args.question)

    print("Question:")
    print(args.question)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
