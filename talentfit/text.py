from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

import spacy
from nltk.stem import PorterStemmer


DEFAULT_ABBREVIATIONS: Mapping[str, str] = {
    "GPHR": "Global Professional in Human Resources",
    "CSR": "Corporate Social Responsibility",
    "MES": "Manufacturing Execution Systems",
    "SPHR": "Senior Professional in Human Resources",
    "SVP": "Senior Vice President",
    "GIS": "Geographic Information System",
    "RRP": "Reduced Risk Products",
    "CHRO": "Chief Human Resources Officer",
    "HRIS": "Human resources information system",
    "HR": "Human resources",
}


@dataclass
class TextPreprocessor:
    spacy_model: str = "en_core_web_sm"
    abbreviations: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_ABBREVIATIONS))

    def __post_init__(self) -> None:
        self._nlp = spacy.load(self.spacy_model)
        self._stemmer = PorterStemmer()

    def replace_abbreviations(self, text: str) -> str:
        replaced = text
        for abbr, repl in self.abbreviations.items():
            pattern = r"\b{}\b".format(re.escape(abbr))
            replaced = re.sub(pattern, repl, replaced, flags=re.IGNORECASE)
        return replaced

    def clean_sentence(self, text: str) -> str:
        # Mirrors notebook behavior as closely as possible.
        new_text = re.sub(r"[+*,.|(){}&\-']", "", text)
        new_text = self.replace_abbreviations(new_text)
        words = new_text.split()
        stemmed_words = [self._stemmer.stem(w) for w in words]
        doc = self._nlp(" ".join(stemmed_words))
        lemmas = [t.lemma_ for t in doc if not t.is_stop]
        return " ".join(lemmas).strip()

