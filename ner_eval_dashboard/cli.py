import argparse
import inspect
import logging
from pathlib import Path

from ner_eval_dashboard.app import create_app
from ner_eval_dashboard.dataset import Dataset
from ner_eval_dashboard.predictor import Predictor
from ner_eval_dashboard.tokenizer import Tokenizer


def load_predictor(args: argparse.Namespace) -> Predictor:
    predictor_cls = Predictor.load(args.predictor_type)
    predictor_args = list(inspect.signature(predictor_cls.__init__).parameters.keys())
    if predictor_args == ["self"]:
        return predictor_cls()
    if predictor_args == ["self", "name_or_path"]:
        return predictor_cls(name_or_path=args.predictor_name_or_path)
    raise ValueError(
        f"Predictor '{args.predictor_type}' cannot be instantiated via CLI. Please create a python script."
    )


def load_tokenizer(args: argparse.Namespace) -> Tokenizer:
    tokenizer_cls = Tokenizer.load(args.tokenizer)
    tokenizer_args = list(inspect.signature(tokenizer_cls.__init__).parameters.keys())
    if tokenizer_args == ["self"]:
        return tokenizer_cls()
    raise ValueError(f"Tokenizer '{args.tokenizer}' cannot be instantiated via CLI. Please create a python script.")


def _load_initial_dataset(tokenizer: Tokenizer, args: argparse.Namespace) -> Dataset:
    dataset_cls = Dataset.load(args.dataset_type)
    dataset_args = list(inspect.signature(dataset_cls.__init__).parameters.keys())
    if dataset_args == ["self", "tokenizer"]:
        if args.dataset_path:
            logging.warning(
                f"Dataset '{args.dataset_type}' does not require a dataset path but has one given."
                "The dataset path is ignored."
            )
        return dataset_cls(tokenizer)
    if dataset_args == ["self", "tokenizer", "file_path"]:
        if not args.dataset_path:
            raise ValueError(f"Dataset '{args.dataset_type}' requires a dataset path but none is given.")
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists() and dataset_path.is_file():
            raise ValueError(f"Dataset '{args.dataset_type}' requires the dataset path to exist and to be a file.")

        return dataset_cls(tokenizer, file_path=args.dataset_path)

    if dataset_args == ["self", "tokenizer", "base_dir"]:
        if not args.dataset_path:
            raise ValueError(f"Dataset '{args.dataset_type}' requires a dataset path but none is given.")
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists() and dataset_path.is_dir():
            raise ValueError(f"Dataset '{args.dataset_type}' requires the dataset path to exist and to be a directory.")

        return dataset_cls(tokenizer, base_dir=args.dataset_path)

    raise ValueError(f"Dataset '{args.dataset_type}' cannot be instantiated via CLI. Please create a python script.")


def load_dataset(tokenizer: Tokenizer, args: argparse.Namespace) -> Dataset:
    dataset = _load_initial_dataset(tokenizer, args)
    if args.extra_unlabeled_data:
        with Path(args.extra_unlabeled_data).open("r", encoding="utf-8") as f:
            dataset.add_unlabeled([line for line in f])
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ner-Eval-Dashboard")
    parser.add_argument("predictor_type", type=str, help="the type of model", choices=Predictor.registered_names)
    parser.add_argument("predictor_name_or_path", type=str, help="the name of path of the model to use")
    parser.add_argument("tokenizer", type=str, help="name of the tokenizer to use", choices=Tokenizer.registered_names)
    parser.add_argument("dataset_type", type=str, help="the type of dataset", choices=Dataset.registered_names)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        required=False,
        help="the path of the dataset to use. Leave empty if dataset doesn't require external files.",
    )
    parser.add_argument(
        "--extra_unlabeled_data",
        required=False,
        default="",
        type=str,
        help="the path to the file containing extra unlabeled_data",
    )
    parser.add_argument("--use", nargs="+", help="use specific components", type=str)
    parser.add_argument("--exclude", nargs="+", help="exclude components", type=str)

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
