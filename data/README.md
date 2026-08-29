# Data handling

Put de-identified, approved transcripts in separate local directories such as \`data/human/\` and \`data/llm/\`. Each \`.txt\` file is treated as one document/session.

- Never commit identifiable counselling content, client metadata, API keys, or restricted source material.
- Preserve the original collection and consent conditions.
- Use the small fictional \`*_example\` folders only to test that the pipeline runs.
- Document model version, prompt format, sampling settings, preprocessing decisions, and exclusions outside the private corpus if you need a complete audit trail.
