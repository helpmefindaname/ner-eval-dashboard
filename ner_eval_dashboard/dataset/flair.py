from pathlib import Path
from typing import List, Optional

import flair
from flair.data import Corpus, Sentence, Span
from flair.datasets import (
    CONLL_03,
    CONLL_03_DUTCH,
    CONLL_03_GERMAN,
    CONLL_03_SPANISH,
    NER_ARABIC_ANER,
    NER_ARABIC_AQMAR,
    NER_BASQUE,
    NER_CHINESE_WEIBO,
    NER_DANISH_DANE,
    NER_ENGLISH_MOVIE_SIMPLE,
    NER_ENGLISH_PERSON,
    NER_ENGLISH_RESTAURANT,
    NER_ENGLISH_SEC_FILLINGS,
    NER_ENGLISH_STACKOVERFLOW,
    NER_ENGLISH_TWITTER,
    NER_ENGLISH_WEBPAGES,
    NER_ENGLISH_WIKIGOLD,
    NER_ENGLISH_WNUT_2020,
    NER_FINNISH,
    NER_GERMAN_BIOFID,
    NER_GERMAN_EUROPARL,
    NER_GERMAN_GERMEVAL,
    NER_GERMAN_LEGAL,
    NER_GERMAN_POLITICS,
    NER_HUNGARIAN,
    NER_ICELANDIC,
    NER_JAPANESE,
    NER_MASAKHANE,
    NER_MULTI_CONER,
    NER_MULTI_WIKIANN,
    NER_MULTI_WIKINER,
    NER_MULTI_XTREME,
    NER_SWEDISH,
    NER_TURKU,
    WNUT_17,
    ColumnCorpus,
    DataLoader,
)
from flair.datasets.sequence_labeling import JsonlCorpus

from ner_eval_dashboard.datamodels import DatasetType, Label, LabeledText
from ner_eval_dashboard.dataset import Dataset
from ner_eval_dashboard.tokenizer import Tokenizer


class FlairDataset(Dataset):
    def __init__(self, name: str, corpus: Corpus, label_type: str, tokenizer: Tokenizer):
        train = self._convert_dataset_to_examples(corpus.train, DatasetType.TRAIN, label_type)
        val = self._convert_dataset_to_examples(corpus.dev, DatasetType.VALIDATION, label_type)
        test = self._convert_dataset_to_examples(corpus.test, DatasetType.TEST, label_type)
        super().__init__(name, tokenizer, train=train, val=val, test=test)

    def _convert_dataset_to_examples(
        self, dataset: Optional[Dataset], dataset_type: DatasetType, label_type: str
    ) -> List[LabeledText]:
        if dataset is None:
            return []
        return [
            self._convert_sentence_to_labeled_text(sentence[0], i, dataset_type, label_type)
            for i, sentence in enumerate(DataLoader(dataset, batch_size=1))
        ]

    @staticmethod
    def _get_span_text(span: Span) -> str:
        text = ""
        for token in span.tokens:
            text += token.text
            text += " " * token.whitespace_after
        return text.strip()

    def _convert_sentence_to_labeled_text(
        self, sentence: Sentence, idx: int, dataset_type: DatasetType, label_type: str
    ) -> LabeledText:
        return LabeledText(
            dataset_text_id=idx,
            dataset_type=dataset_type,
            text=sentence.to_original_text(),
            labels=[
                Label(
                    entity_type=span.tag,
                    start=span.start_position,
                    end=span.end_position,
                    text=self._get_span_text(span),
                )
                for span in sentence.get_spans(label_type)
            ],
        )


@Dataset.register("COLUMN_DATASET")
class FlairColumnDataset(FlairDataset):
    def __init__(self, tokenizer: Tokenizer, base_dir: str):
        base_dir_path = Path(base_dir)
        super().__init__(
            base_dir_path.stem,
            ColumnCorpus(base_dir_path, {0: "text", 1: "ner"}, name=base_dir_path.stem, tag_to_bioes="ner"),
            "ner",
            tokenizer,
        )


@Dataset.register("JSONL_DATASET")
class FlairJsonlDataset(FlairDataset):
    def __init__(self, tokenizer: Tokenizer, base_dir: str):
        if flair.__version__ <= "0.11.3":
            raise NotImplementedError("Flair JsonL does not support span labels yet.")

        base_dir_path = Path(base_dir)
        super().__init__(
            base_dir_path.stem,
            JsonlCorpus(base_dir_path, name=base_dir_path.stem),
            "ner",
            tokenizer,
        )


@Dataset.register("CONLL03")
class FlairConll03(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("CONLL03", CONLL_03(), "ner", tokenizer)


@Dataset.register("CONLL03_GERMAN")
class FlairConll03_GERMAN(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("CONLL03_GERMAN", CONLL_03_GERMAN(), "ner", tokenizer)


@Dataset.register("CONLL03_DUTCH")
class FlairConll03_DUTCH(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("CONLL03_DUTCH", CONLL_03_DUTCH(), "ner", tokenizer)


@Dataset.register("CONLL03_SPANISH")
class FlairConll03_SPANISH(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("CONLL03_SPANISH", CONLL_03_SPANISH(), "ner", tokenizer)


@Dataset.register("WNUT17")
class FlairWnut17(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("WNUT17", WNUT_17(), "ner", tokenizer)


@Dataset.register("ARABIC_ANER")
class FlairArabicAner(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("ARABIC_ANER", NER_ARABIC_ANER(), "ner", tokenizer)


@Dataset.register("ARABIC_AQMAR")
class FlairArabicAqmar(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("ARABIC_AQMAR", NER_ARABIC_AQMAR(), "ner", tokenizer)


@Dataset.register("BASQUE")
class FlairBasque(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("BASQUE", NER_BASQUE(), "ner", tokenizer)


@Dataset.register("WEIBO")
class FlairWeibo(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("WEIBO", NER_CHINESE_WEIBO(), "ner", tokenizer)


@Dataset.register("DANE")
class FlairDane(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("DANE", NER_DANISH_DANE(), "ner", tokenizer)


@Dataset.register("MOVIE_SIMPLE")
class FlairMovieSimple(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("MOVIE_SIMPLE", NER_ENGLISH_MOVIE_SIMPLE(), "ner", tokenizer)


@Dataset.register("MOVIE_COMPLEX")
class FlairMovieComplex(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("MOVIE_COMPLEX", NER_ENGLISH_MOVIE_SIMPLE(), "ner", tokenizer)


@Dataset.register("SEC_FILLINGS")
class FlairSecFillings(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("SEC_FILLINGS", NER_ENGLISH_SEC_FILLINGS(), "ner", tokenizer)


@Dataset.register("RESTAURANT")
class FlairRestaurant(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("RESTAURANT", NER_ENGLISH_RESTAURANT(), "ner", tokenizer)


@Dataset.register("STACKOVERFLOW")
class FlairStackoverflow(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("STACKOVERFLOW", NER_ENGLISH_STACKOVERFLOW(), "ner", tokenizer)


@Dataset.register("TWITTER")
class FlairTwitter(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("TWITTER", NER_ENGLISH_TWITTER(), "ner", tokenizer)


@Dataset.register("PERSON")
class FlairPerson(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("PERSON", NER_ENGLISH_PERSON(), "ner", tokenizer)


@Dataset.register("WEBPAGES")
class FlairWebpages(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("WEBPAGES", NER_ENGLISH_WEBPAGES(), "ner", tokenizer)


@Dataset.register("WNUT2020")
class FlairWnut2020(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("WNUT2020", NER_ENGLISH_WNUT_2020(), "ner", tokenizer)


@Dataset.register("WIKIGOLD")
class FlairWikiGold(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("WIKIGOLD", NER_ENGLISH_WIKIGOLD(), "ner", tokenizer)


@Dataset.register("FINER")
class FlairFiner(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("FINER", NER_FINNISH(), "ner", tokenizer)


@Dataset.register("BIOFID")
class FlairBiofid(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("BIOFID", NER_GERMAN_BIOFID(), "ner", tokenizer)


@Dataset.register("EUROPARL")
class FlairEuroparl(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("EUROPARL", NER_GERMAN_EUROPARL(), "ner", tokenizer)


@Dataset.register("LEGAL_NER")
class FlairLegalNer(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("LEGAL_NER", NER_GERMAN_LEGAL(), "ner", tokenizer)


@Dataset.register("GERMEVAL")
class FlairGermeval(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("GERMEVAL", NER_GERMAN_GERMEVAL(), "ner", tokenizer)


@Dataset.register("POLITICS")
class FlairPolitics(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("POLITICS", NER_GERMAN_POLITICS(), "ner", tokenizer)


@Dataset.register("BUSINESS")
class FlairBusiness(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("BUSINESS", NER_HUNGARIAN(), "ner", tokenizer)


@Dataset.register("ICELANDIC_NER")
class FlairIcelandicNer(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("ICELANDIC_NER", NER_ICELANDIC(), "ner", tokenizer)


@Dataset.register("HIRONSAN")
class FlairHironsan(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("HIRONSAN", NER_JAPANESE(), "ner", tokenizer)


@Dataset.register("MASAKHANE")
class FlairMasakhane(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("MASAKHANE", NER_MASAKHANE(), "ner", tokenizer)


@Dataset.register("MULTI_CONER")
class FlairMultiConer(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("MULTI_CONER", NER_MULTI_CONER(), "ner", tokenizer)


@Dataset.register("WIKIANN")
class FlairWikiann(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("WIKIANN", NER_MULTI_WIKIANN(), "ner", tokenizer)


@Dataset.register("XTREME")
class FlairXtreme(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("XTREME", NER_MULTI_XTREME(), "ner", tokenizer)


@Dataset.register("WIKINER")
class FlairWikiner(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("WIKINER", NER_MULTI_WIKINER(), "ner", tokenizer)


@Dataset.register("SWEDISH_NER")
class FlairSwedishNer(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("SWEDISH_NER", NER_SWEDISH(), "ner", tokenizer)


@Dataset.register("TURKU")
class FlairTurku(FlairDataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__("TURKU", NER_TURKU(), "ner", tokenizer)
