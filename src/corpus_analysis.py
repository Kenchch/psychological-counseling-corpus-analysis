"""Compare de-identified human and LLM counselling corpora descriptively."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def normalise(text: str, lemmatiser: WordNetLemmatizer, stop_words: set[str]) -> list[str]:
    """Lowercase, tokenise, remove stop words, and lemmatise a document."""
    tokens = TOKEN_RE.findall(text.lower().replace("’", "'"))
    return [lemmatiser.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 1]


def load_corpus(directory: Path, lemmatiser: WordNetLemmatizer, stop_words: set[str]) -> dict[str, list[str]]:
    documents = {}
    for path in sorted(directory.glob("*.txt")):
        documents[path.stem] = normalise(path.read_text(encoding="utf-8"), lemmatiser, stop_words)
    if not documents:
        raise ValueError(f"No UTF-8 .txt files found in {directory}")
    return documents


def ngrams(tokens: list[str], size: int) -> Iterable[str]:
    return (" ".join(tokens[index:index + size]) for index in range(max(0, len(tokens) - size + 1)))


def frequency_table(documents: dict[str, list[str]], top_n: int) -> pd.DataFrame:
    counts = Counter(token for tokens in documents.values() for token in tokens)
    document_frequency = Counter(token for tokens in documents.values() for token in set(tokens))
    return pd.DataFrame(
        [{"term": term, "frequency": frequency, "document_range": document_frequency[term]}
         for term, frequency in counts.most_common(top_n)]
    )


def ngram_table(documents: dict[str, list[str]], min_size: int, max_size: int, top_n: int) -> pd.DataFrame:
    rows = []
    for size in range(min_size, max_size + 1):
        counts = Counter(gram for tokens in documents.values() for gram in ngrams(tokens, size))
        document_frequency = Counter(gram for tokens in documents.values() for gram in set(ngrams(tokens, size)))
        rows.extend(
            {"n": size, "ngram": gram, "frequency": frequency, "document_range": document_frequency[gram]}
            for gram, frequency in counts.most_common(top_n)
        )
    return pd.DataFrame(rows)


def log_likelihood(left_count: int, left_total: int, right_count: int, right_total: int) -> float:
    """Dunning G² score for a 2x2 token-count table."""
    combined, total = left_count + right_count, left_total + right_total
    if not combined or not total:
        return 0.0
    expected_left, expected_right = left_total * combined / total, right_total * combined / total
    terms = ((left_count, expected_left), (right_count, expected_right))
    return 2 * sum(observed * math.log(observed / expected) for observed, expected in terms if observed and expected)


def keyword_table(llm_docs: dict[str, list[str]], human_docs: dict[str, list[str]], top_n: int) -> pd.DataFrame:
    llm_counts = Counter(token for tokens in llm_docs.values() for token in tokens)
    human_counts = Counter(token for tokens in human_docs.values() for token in tokens)
    llm_total, human_total = sum(llm_counts.values()), sum(human_counts.values())
    vocabulary = sorted(set(llm_counts) | set(human_counts))
    rows = []
    for term in vocabulary:
        llm_count, human_count = llm_counts[term], human_counts[term]
        log_ratio = math.log2((llm_count + 0.5) / (llm_total + 0.5 * len(vocabulary)))
        log_ratio -= math.log2((human_count + 0.5) / (human_total + 0.5 * len(vocabulary)))
        rows.append(
            {"term": term, "llm_frequency": llm_count, "human_frequency": human_count,
             "log_ratio_llm_vs_human": log_ratio,
             "log_likelihood": log_likelihood(llm_count, llm_total, human_count, human_total)}
        )
    return pd.DataFrame(rows).sort_values(["log_likelihood", "log_ratio_llm_vs_human"], ascending=[False, False]).head(top_n)


def write_analysis(human_dir: Path, llm_dir: Path, output: Path, top_n: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError as error:
        raise RuntimeError("Install NLTK data first: python -m nltk.downloader wordnet omw-1.4 stopwords") from error
    lemmatiser = WordNetLemmatizer()
    human, llm = load_corpus(human_dir, lemmatiser, stop_words), load_corpus(llm_dir, lemmatiser, stop_words)
    frequency_table(llm, top_n).to_csv(output / "llm_word_frequency.csv", index=False)
    frequency_table(human, top_n).to_csv(output / "human_word_frequency.csv", index=False)
    ngram_table(llm, 2, 5, top_n).to_csv(output / "llm_ngrams.csv", index=False)
    ngram_table(human, 2, 5, top_n).to_csv(output / "human_ngrams.csv", index=False)
    keyword_table(llm, human, top_n).to_csv(output / "keywords_llm_vs_human.csv", index=False)
    keyword_table(human, llm, top_n).rename(
        columns={"log_ratio_llm_vs_human": "log_ratio_human_vs_llm"}
    ).to_csv(output / "keywords_human_vs_llm.csv", index=False)
    summary = {
        "human_documents": len(human), "llm_documents": len(llm),
        "human_tokens": sum(map(len, human.values())), "llm_tokens": sum(map(len, llm.values())),
        "method": "lowercase tokenisation, English stop-word removal, WordNet lemmatisation, smoothed log ratio and log-likelihood",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path, required=True, help="Directory of de-identified human-session .txt files")
    parser.add_argument("--llm", type=Path, required=True, help="Directory of LLM-session .txt files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()
    write_analysis(args.human, args.llm, args.output, args.top_n)


if __name__ == "__main__":
    main()

