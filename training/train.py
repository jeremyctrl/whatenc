from dataset import generate_samples, load_corpus


def main():
    corpus = load_corpus()
    samples = generate_samples(corpus)


if __name__ == "__main__":
    main()
