import argparse
import os
import re
from collections import Counter
from typing import List

from openai import OpenAI

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with", "is", "are",
    "was", "were", "be", "been", "it", "this", "that", "as", "by", "from", "about", "into", "than", "then",
    "i", "you", "he", "she", "we", "they", "them", "my", "your", "our", "their", "me", "us", "do", "does",
}


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def local_extractive_summary(text: str, max_sentences: int = 3) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return text.strip()

    words = [w for w in tokenize(text) if w not in STOPWORDS]
    freq = Counter(words)

    sentence_scores = []
    for idx, sentence in enumerate(sentences):
        sentence_words = [w for w in tokenize(sentence) if w not in STOPWORDS]
        score = sum(freq[w] for w in sentence_words)
        sentence_scores.append((idx, score))

    top = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    top_sorted = sorted(top, key=lambda x: x[0])

    return " ".join(sentences[idx] for idx, _ in top_sorted)


def openai_summary(text: str, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a concise summarization assistant.",
            },
            {
                "role": "user",
                "content": f"Summarize the following text in a concise paragraph:\n\n{text}",
            },
        ],
    )
    return response.output_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize long text.")
    parser.add_argument("--text", required=True, help="Long text to summarize")
    parser.add_argument("--mode", choices=["local", "openai"], default="local", help="Summarization mode")
    parser.add_argument("--max-sentences", type=int, default=3, help="Used in local mode")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Used in openai mode")
    args = parser.parse_args()

    if args.mode == "local":
        summary = local_extractive_summary(args.text, max_sentences=args.max_sentences)
    else:
        summary = openai_summary(args.text, model=args.model)

    print("Summary")
    print(summary)


if __name__ == "__main__":
    main()
