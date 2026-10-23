import math
import re
from collections import Counter, defaultdict


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.vocab = set()

    def fit(self, texts, labels):
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            for word in tokenize(text):
                self.word_counts[label][word] += 1
                self.vocab.add(word)
        return self

    def log_scores(self, text):
        total = sum(self.class_counts.values())
        words = tokenize(text)
        scores = {}
        for label, count in self.class_counts.items():
            score = math.log(count / total)
            denom = sum(self.word_counts[label].values()) + self.alpha * len(self.vocab)
            for word in words:
                if word in self.vocab:
                    num = self.word_counts[label][word] + self.alpha
                    score += math.log(num / denom)
            scores[label] = score
        return scores

    def predict(self, text):
        scores = self.log_scores(text)
        return max(scores, key=scores.get)

    def score(self, texts, labels):
        hits = sum(self.predict(t) == y for t, y in zip(texts, labels))
        return hits / len(texts)
