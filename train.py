from data import TEST, TRAIN
from naive_bayes import NaiveBayes


def main():
    texts, labels = zip(*TRAIN)
    model = NaiveBayes().fit(texts, labels)

    test_texts, test_labels = zip(*TEST)
    print(f"test accuracy: {model.score(test_texts, test_labels):.2f}")

    for sentence in ["what a great story", "this was a terrible movie"]:
        print(f"{sentence!r} -> {model.predict(sentence)}")


if __name__ == "__main__":
    main()
