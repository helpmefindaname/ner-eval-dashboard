import argparse

from ner_eval_dashboard.app import create_app
from ner_eval_dashboard.dataset import Dataset
from ner_eval_dashboard.predictor import Predictor
from ner_eval_dashboard.tokenizer import Tokenizer


def load_predictor(args: argparse.Namespace) -> Predictor:
    pass


def load_tokenizer(args: argparse.Namespace) -> Tokenizer:
    pass


def load_dataset(tokenizer: Tokenizer, args: argparse.Namespace) -> Dataset:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ner-Eval-Dashboard")
    parser.add_argument("predictor_type", type=str, help="the type of model", choices=["flair", "huggingface", "spacy"])
    parser.add_argument("predictor_path_or_name", type=str, help="the name of path of the model to use")
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
    parser.add_argument('--use', nargs='+', help='exclude components', type=str)
    parser.add_argument('--exclude', nargs='+', help='exclude components', type=str)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = load_predictor(args)
    tokenizer = load_tokenizer(args)
    dataset = load_dataset(tokenizer, args)
    app = create_app(__name__, predictor, dataset, args.use, args.exclude)

    app.run()


if __name__ == "__main__":
    main()
