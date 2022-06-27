import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ner-Eval-Dashboard")
    parser.add_argument("model_type", type=str, help="the type of model", choices=["flair", "huggingface", "spacy"])
    parser.add_argument("model_path_or_name", type=str, help="the name of path of the model to use")
    parser.add_argument("tokenizer", type=str, help="name of the tokenizer to use")
    parser.add_argument("dataset_type", type=str, help="the type of dataset", choices=["flair", "huggingface"])
    parser.add_argument("dataset_path", type=str, help="the path of the dataset to use")
    parser.add_argument(
        "extra_unlabeled",
        required=False,
        default="",
        type=str,
        help="the path to the file containing extra unlabeled_data",
    )

    return parser.parse_args()


def main() -> None:
    pass


if __name__ == "__main__":
    main()
