from src.corpus_analysis import keyword_table, ngram_table, normalise


class IdentityLemma:
    def lemmatize(self, value):
        return value


def test_normalise_removes_stop_words():
    assert normalise("The counsellor listens carefully", IdentityLemma(), {"the"}) == ["counsellor", "listens", "carefully"]


def test_ngram_table_reports_document_range():
    frame = ngram_table({"a": ["feel", "heard"], "b": ["feel", "heard"]}, 2, 2, 5)
    assert frame.iloc[0].to_dict() == {"n": 2, "ngram": "feel heard", "frequency": 2, "document_range": 2}


def test_keyword_direction_is_positive_for_llm_specific_term():
    frame = keyword_table({"a": ["support", "support"]}, {"b": ["yeah", "yeah"]}, 10)
    assert frame.loc[frame.term == "support", "log_ratio_llm_vs_human"].iloc[0] > 0

