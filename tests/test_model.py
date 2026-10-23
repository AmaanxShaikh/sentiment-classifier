import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import TEST, TRAIN
from naive_bayes import NaiveBayes, tokenize


class TestModel(unittest.TestCase):
    def setUp(self):
        texts, labels = zip(*TRAIN)
        self.model = NaiveBayes().fit(texts, labels)

    def test_tokenize(self):
        self.assertEqual(tokenize("Hello, World!"), ["hello", "world"])

    def test_predicts_positive(self):
        self.assertEqual(self.model.predict("a great happy film"), "pos")

    def test_predicts_negative(self):
        self.assertEqual(self.model.predict("a boring awful film"), "neg")

    def test_accuracy_on_test_set(self):
        texts, labels = zip(*TEST)
        self.assertGreaterEqual(self.model.score(texts, labels), 0.75)


if __name__ == "__main__":
    unittest.main()
