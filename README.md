# Psychological Counselling Corpus Analysis

An ethical, reproducible corpus-analysis project that compares language patterns in human therapist transcripts and LLM-generated responses. It operationalises the submitted report's three analyses: normalised word frequencies, 2-5 word n-grams, and directional keyword comparisons.

> This repository is for linguistic research only. It is not a clinical tool, does not assess treatment quality, and must not be used to provide psychological advice or to make decisions about people.

## What is included

\`\`\`text
src/corpus_analysis.py   Reproducible analysis CLI
data/*_example/          Tiny fictional examples for a safe smoke test
data/README.md           Expected private-data layout and handling guidance
tests/                   Utility tests
Psychological Counseling Corpus Analysis.docx  Original report
\`\`\`

The source counselling transcripts are not included. They may be sensitive, copyrighted, or subject to consent restrictions. Use only de-identified, approved material and keep it out of version control.

## Setup and example

\`\`\`bash
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4 stopwords

python src/corpus_analysis.py \
  --human data/human_example --llm data/llm_example --output output/example
\`\`\`

The command creates CSV tables and a JSON summary in \`output/example/\`. For a real study, supply folders containing one UTF-8 \`.txt\` session per file:

\`\`\`bash
python src/corpus_analysis.py --human /secure/path/human --llm /secure/path/llm --output output/study
\`\`\`

## Method

1. Lowercase, tokenise, remove English stop words, and lemmatise with WordNet.
2. Count term frequencies and 2-5 word n-grams, reporting document range as well as frequency.
3. Compare corpora using a smoothed log ratio and log-likelihood score. Positive log ratio means a term is relatively more frequent in the LLM corpus; negative means it is relatively more frequent in the human corpus.

The results are descriptive. They do not establish therapeutic safety, empathy, or effectiveness. Interpret them with the study's sampling and model limitations in mind.
