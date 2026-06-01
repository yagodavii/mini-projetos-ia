import argparse
from dataclasses import dataclass

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


@dataclass
class SentimentResult:
    label: str
    score: float
    compound: float


def ensure_lexicon() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


def analyze_sentiment(text: str) -> SentimentResult:
    ensure_lexicon()
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
        score = scores["pos"]
    elif compound <= -0.05:
        label = "negative"
        score = scores["neg"]
    else:
        label = "neutral"
        score = scores["neu"]

    return SentimentResult(label=label, score=score, compound=compound)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze text sentiment (positive, negative, neutral).")
    parser.add_argument("--text", required=True, help="Text to analyze")
    args = parser.parse_args()

    result = analyze_sentiment(args.text)
    print("Sentiment Analysis Result")
    print(f"Label: {result.label}")
    print(f"Confidence score: {result.score:.3f}")
    print(f"Compound score: {result.compound:.3f}")


if __name__ == "__main__":
    main()
