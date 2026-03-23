"""Lightweight text classifier using TF-IDF + logistic regression (sklearn) or a pure-Python fallback."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from homie_core.ml.base import LocalModel

# ---------------------------------------------------------------------------
# Optional sklearn imports — fall back to pure-Python naive Bayes if missing
# ---------------------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ===================================================================
# Pure-Python fallback — Multinomial Naive Bayes on word counts
# ===================================================================

def _tokenize(text: str) -> list[str]:
    """Very simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


class _NaiveBayesFallback:
    """Minimal multinomial Naive Bayes that needs zero external deps."""

    def __init__(self, classes: list[str]) -> None:
        self.classes = list(classes)
        self.class_word_counts: dict[str, Counter] = {c: Counter() for c in self.classes}
        self.class_doc_counts: dict[str, int] = {c: 0 for c in self.classes}
        self.vocab: set[str] = set()
        self.total_docs = 0

    def fit(self, texts: list[str], labels: list[str]) -> None:
        for text, label in zip(texts, labels):
            if label not in self.class_word_counts:
                self.classes.append(label)
                self.class_word_counts[label] = Counter()
                self.class_doc_counts[label] = 0
            tokens = _tokenize(text)
            self.class_word_counts[label].update(tokens)
            self.class_doc_counts[label] += 1
            self.vocab.update(tokens)
            self.total_docs += 1

    def predict(self, texts: list[str]) -> list[str]:
        return [self._predict_one(t) for t in texts]

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        return [self._proba_one(t) for t in texts]

    # -- internals -------------------------------------------------

    def _log_likelihood(self, text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        vocab_size = max(len(self.vocab), 1)
        scores: dict[str, float] = {}
        for cls in self.classes:
            total = sum(self.class_word_counts[cls].values())
            log_prior = math.log((self.class_doc_counts[cls] + 1) / (self.total_docs + len(self.classes)))
            log_lik = 0.0
            for tok in tokens:
                count = self.class_word_counts[cls].get(tok, 0)
                log_lik += math.log((count + 1) / (total + vocab_size))
            scores[cls] = log_prior + log_lik
        return scores

    def _predict_one(self, text: str) -> str:
        scores = self._log_likelihood(text)
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def _proba_one(self, text: str) -> dict[str, float]:
        scores = self._log_likelihood(text)
        # convert log-scores to probabilities via log-sum-exp
        max_s = max(scores.values())
        exp_scores = {c: math.exp(s - max_s) for c, s in scores.items()}
        total = sum(exp_scores.values())
        return {c: v / total for c, v in exp_scores.items()}

    def to_dict(self) -> dict:
        return {
            "classes": self.classes,
            "class_word_counts": {c: dict(wc) for c, wc in self.class_word_counts.items()},
            "class_doc_counts": self.class_doc_counts,
            "vocab": sorted(self.vocab),
            "total_docs": self.total_docs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "_NaiveBayesFallback":
        obj = cls(data["classes"])
        obj.class_word_counts = {c: Counter(wc) for c, wc in data["class_word_counts"].items()}
        obj.class_doc_counts = data["class_doc_counts"]
        obj.vocab = set(data["vocab"])
        obj.total_docs = data["total_docs"]
        return obj


# ===================================================================
# Public API
# ===================================================================

class TextClassifier(LocalModel):
    """Lightweight text classifier.

    If *scikit-learn* is installed it uses TF-IDF + Logistic Regression.
    Otherwise it falls back to a pure-Python multinomial Naive Bayes.

    Supported tasks: intent classification, sentiment analysis,
    priority scoring, domain classification.
    """

    def __init__(self, name: str, classes: list[str]) -> None:
        super().__init__(name=name, model_type="classifier")
        self.classes = list(classes)
        self._backend: str = "sklearn" if _HAS_SKLEARN else "naive_bayes"

        # sklearn backend state
        self._vectorizer: Any = None
        self._clf: Any = None

        # fallback backend state
        self._nb: _NaiveBayesFallback | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: list, y: list) -> dict:
        """Train on *X* (list of text strings) and *y* (list of label strings).

        Returns a dict of metrics: ``accuracy``, ``backend``, ``n_samples``,
        ``n_classes``.
        """
        texts: list[str] = X
        labels: list[str] = y

        if not texts or not labels:
            raise ValueError("Training data must not be empty.")
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")

        # Discover new classes from labels
        seen = set(labels)
        for c in seen:
            if c not in self.classes:
                self.classes.append(c)

        if self._backend == "sklearn":
            metrics = self._train_sklearn(texts, labels)
        else:
            metrics = self._train_naive_bayes(texts, labels)

        self._trained = True
        self._metrics = metrics
        return metrics

    def _train_sklearn(self, texts: list[str], labels: list[str]) -> dict:
        self._vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_vec = self._vectorizer.fit_transform(texts)
        self._clf = LogisticRegression(max_iter=500, solver="lbfgs", multi_class="multinomial")
        self._clf.fit(X_vec, labels)
        preds = self._clf.predict(X_vec)
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "backend": "sklearn",
            "n_samples": len(texts),
            "n_classes": len(self.classes),
        }

    def _train_naive_bayes(self, texts: list[str], labels: list[str]) -> dict:
        self._nb = _NaiveBayesFallback(self.classes)
        self._nb.fit(texts, labels)
        preds = self._nb.predict(texts)
        correct = sum(1 for p, l in zip(preds, labels) if p == l)
        acc = correct / len(labels) if labels else 0.0
        return {
            "accuracy": acc,
            "backend": "naive_bayes",
            "n_samples": len(texts),
            "n_classes": len(self.classes),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: list) -> list:
        """Return predicted labels for *X* (list of text strings)."""
        if not self._trained:
            raise RuntimeError(f"Model {self.name!r} has not been trained yet.")
        texts: list[str] = X
        if self._backend == "sklearn":
            X_vec = self._vectorizer.transform(texts)
            return list(self._clf.predict(X_vec))
        else:
            assert self._nb is not None
            return self._nb.predict(texts)

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        """Return per-class probabilities for each text."""
        if not self._trained:
            raise RuntimeError(f"Model {self.name!r} has not been trained yet.")
        if self._backend == "sklearn":
            X_vec = self._vectorizer.transform(texts)
            proba = self._clf.predict_proba(X_vec)
            class_names = list(self._clf.classes_)
            return [
                {class_names[i]: float(row[i]) for i in range(len(class_names))}
                for row in proba
            ]
        else:
            assert self._nb is not None
            return self._nb.predict_proba(texts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save model to *path*.

        The sklearn backend attempts pickle first, then falls back to JSON
        for the naive-Bayes representation.  The fallback backend always
        uses JSON.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._backend == "sklearn" and self._trained:
            try:
                import pickle
                data = {
                    "backend": "sklearn",
                    "name": self.name,
                    "classes": self.classes,
                    "vectorizer": self._vectorizer,
                    "clf": self._clf,
                    "metrics": self._metrics,
                }
                with open(path, "wb") as f:
                    pickle.dump(data, f)
                return
            except Exception:
                pass  # fall through to JSON

        # JSON serialisation (fallback / naive_bayes)
        data = {
            "backend": self._backend,
            "name": self.name,
            "classes": self.classes,
            "metrics": self._metrics,
            "trained": self._trained,
        }
        if self._nb is not None:
            data["nb"] = self._nb.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """Load model from *path*."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Try pickle first
        try:
            import pickle
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and data.get("backend") == "sklearn":
                self._backend = "sklearn"
                self.name = data["name"]
                self.classes = data["classes"]
                self._vectorizer = data["vectorizer"]
                self._clf = data["clf"]
                self._metrics = data.get("metrics", {})
                self._trained = True
                return
        except Exception:
            pass

        # JSON fallback
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._backend = data.get("backend", "naive_bayes")
        self.name = data["name"]
        self.classes = data["classes"]
        self._metrics = data.get("metrics", {})
        self._trained = data.get("trained", False)
        if "nb" in data:
            self._nb = _NaiveBayesFallback.from_dict(data["nb"])
            self._trained = True
