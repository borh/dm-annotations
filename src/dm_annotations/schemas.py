from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    Union,
)

import orjson
import spacy
from pydantic import BaseModel, field_validator
from spacy.tokens import Doc, Span, Token


class Metadata(BaseModel):
    title: str
    author: str
    year: int
    basename: str
    category: list[str]
    permission: bool
    metadata: dict[str, Union[str, bool, int, list[str]]]

    @field_validator("year")
    @classmethod
    def year_must_be_valid(cls, v: int) -> int:
        if v < 0:
            # TODO this is just an example, we should validate using DateTime or similar.
            raise ValueError("Year must be positive")
        return v


@dataclass
class Sentence:
    """
    A sentence with original text and optional parsed Doc.
    """

    text: str
    doc: Optional[Doc] = None


@dataclass
class Paragraph:
    sentences: list[Sentence]
    tags: list[str]


@dataclass
class Document:
    paragraphs: list[Paragraph]
    metadata: Optional[Metadata] = None

    def to_json(self) -> str:
        """
        Serialize Document to a newline-terminated JSON string.
        """
        buf = orjson.dumps(
            self,
            default=lambda o: o.__dict__,
            option=orjson.OPT_APPEND_NEWLINE,
        )
        return buf.decode("utf-8")

    @staticmethod
    def from_json(json_str: str):
        return orjson.loads(json_str)


class MetaDoc:
    def __init__(
        self,
        documents: Iterable[Document],
        nlp,
        min_batch_char_length=1000,
        max_batch_char_length=5000,
        batch_size=800,
    ):
        self.nlp = nlp
        self.min_batch_char_length = min_batch_char_length
        self.max_batch_char_length = max_batch_char_length
        self.batch_size = batch_size
        # map (doc_idx, para_idx, sent_idx) → annotation dict
        self.sentence_annotations: dict[tuple[int, int, int], Any] = {}
        self.docs = [
            self.process_doc_text(doc, idx) for idx, doc in enumerate(documents)
        ]

    def process_doc_text(self, document: Document, doc_index: int):
        # build index batches for spaCy.pipe
        batches: list[list[tuple[int, int, int]]] = []
        batch: list[tuple[int, int, int]] = []
        batch_char_count: int = 0
        for para_index, paragraph in enumerate(document.paragraphs):
            for sent_index, sentence in enumerate(paragraph.sentences):
                sentence_length = len(sentence.text)
                if batch_char_count + sentence_length > self.max_batch_char_length:
                    batches.append(batch)
                    batch, batch_char_count = [], 0
                batch.append((doc_index, para_index, sent_index))
                batch_char_count += sentence_length

        if batch:
            batches.append(batch)

        for batch in batches:
            texts = [
                document.paragraphs[para_idx].sentences[sent_idx].text
                for _, para_idx, sent_idx in batch
            ]
            spacy_docs = list(self.nlp.pipe(texts, batch_size=self.batch_size))

            for (doc_idx, para_idx, sent_idx), spacy_doc in zip(batch, spacy_docs):
                sentence = document.paragraphs[para_idx].sentences[sent_idx]
                sentence.doc = spacy_doc
                for span in spacy_doc.spans.get("sc", []):
                    self.merge_annotations((doc_idx, para_idx, sent_idx), span)

        return document

    def merge_annotations(
        self,
        sentence_id: tuple[int, int, int],
        span: spacy.tokens.Span,
    ) -> None:
        if sentence_id not in self.sentence_annotations:
            self.sentence_annotations[sentence_id] = {}
        self.sentence_annotations[sentence_id].update(
            {
                attr: getattr(span, attr)
                for attr in dir(span)
                if not attr.startswith("_")
            }
        )

    # Lazy Iteration Methods
    def tokens(self) -> Iterable[Token]:
        for doc in self.docs:
            for paragraph in doc.paragraphs:
                for sentence in paragraph.sentences:
                    for token in sentence.doc:
                        yield token

    def sentences(self) -> Iterable[Span]:
        for doc in self.docs:
            for paragraph in doc.paragraphs:
                for sentence in paragraph.sentences:
                    yield sentence

    def paragraphs(self) -> Iterable[Span]:
        for doc in self.docs:
            for paragraph in doc.paragraphs:
                yield paragraph

    # Functional Interfaces
    def map_tokens(self, function):
        return (function(token) for token in self.tokens())

    def filter_tokens(self, predicate):
        return (token for token in self.tokens() if predicate(token))


class MetaDocInfo(TypedDict):
    genre: str
    title: str
    sentence_id: int
    segment_id: int


def get_metadoc_info(doc: Doc) -> MetaDocInfo:
    """
    Extracts metadata from a spaCy Doc object.

    :param doc: spaCy Doc with metadata in user_data
    :return: MetaDocInfo dictionary

    >>> import spacy
    >>> nlp = spacy.blank("ja")
    >>> doc = nlp("テスト文。")
    >>> doc.user_data["meta"] = {"genre": "test", "title": "Test Title", "sentence_id": 1, "segment_id": 2}
    >>> info = get_metadoc_info(doc)
    >>> info["genre"]
    'test'
    """
    meta = doc.user_data.get("meta", {})
    return MetaDocInfo(
        genre=meta.get("genre", ""),
        title=meta.get("title", ""),
        sentence_id=meta.get("sentence_id", 0),
        segment_id=meta.get("segment_id", 0),
    )


def group_spans_by_label(doc: Doc, key: str) -> dict[str, list[Span]]:
    """
    Groups spans in a Doc by their label for a given key in doc.spans.

    :param doc: spaCy Doc
    :param key: key in doc.spans
    :return: dict mapping label to list of spans

    >>> import spacy
    >>> nlp = spacy.blank("ja")
    >>> doc = nlp("テスト文。")
    >>> from spacy.tokens import Span
    >>> doc.spans["test"] = [Span(doc, 0, 2, label="A"), Span(doc, 2, 3, label="B")]
    >>> grouped = group_spans_by_label(doc, "test")
    >>> set(grouped.keys()) == {"A", "B"}
    True
    """
    result: dict[str, list[Span]] = {}
    for span in doc.spans.get(key, []):
        result.setdefault(span.label_, []).append(span)
    return result


def iter_segments(doc: Doc, key: str = "segment") -> Iterator[Span]:
    """
    Iterates over segment spans in a Doc.

    :param doc: spaCy Doc
    :param key: key in doc.spans (default: "segment")
    :yield: Span objects

    >>> import spacy
    >>> nlp = spacy.blank("ja")
    >>> doc = nlp("テスト文。")
    >>> from spacy.tokens import Span
    >>> doc.spans["segment"] = [Span(doc, 0, 2, label="seg1")]
    >>> list(iter_segments(doc))
    [doc.spans["segment"][0]]
    """
    yield from doc.spans.get(key, [])


class DocMetadata(BaseModel):
    """Unified metadata structure for all documents."""

    title: str
    genre: str  # Always normalized to string
    sentence_id: int = 0
    basename: str = ""
    author: str = ""
    year: Optional[int] = None
    paragraph_id: Optional[int] = None
    section: Optional[str] = None

    @field_validator("genre", mode="before")
    @classmethod
    def normalize_genre(cls, v: Union[str, List[str]]) -> str:
        """Always normalize genre to a single string."""
        if isinstance(v, list):
            return v[0] if v else ""
        return v or ""

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("Year must be positive")
        return v


class DmMatch(TypedDict):
    """Type for discourse marker match results."""

    span: Span
    表現: str
    タイプ: Literal["接続表現", "文末表現"]
    機能: str
    細分類: str
    position: float
    ジャンル: str
    title: str
    sentence_id: int
    section: str | None


class DocMeta(BaseModel):
    basename: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    genre: Optional[str] = None
    sentence_id: int = 0
    paragraph_id: Optional[int] = None
    section: Optional[str] = None


# Protocol for matcher-like objects
class MatcherProtocol(Protocol):
    """Protocol for spaCy matcher objects."""

    def __call__(self, doc: Doc) -> List[tuple[int, int, int]]:
        """Match patterns in a document."""
        ...

    def add(self, key: str, patterns: List[List[Dict[str, Any]]]) -> None:
        """Add patterns to the matcher."""
        ...

    def __contains__(self, key: str) -> bool:
        """Check if a pattern key exists."""
        ...

    def __len__(self) -> int:
        """Get number of patterns."""
        ...


# Pattern types
class PatternDefinition(TypedDict):
    """Type for pattern definitions in JSON files."""

    name: str
    category: List[str]
    examples: List[str]
    pattern: Any  # Complex nested pattern structure


class ConnectivePattern(TypedDict):
    """Type for connective pattern definitions."""

    conjunction: str
    kinou: str
    pattern: List[List[Dict[str, Any]]]


class RegexPattern(TypedDict):
    """Type for regex-based pattern definitions."""

    conjunction: str
    kinou: str
    regex: str


# Network analysis types
class NetworkNode(TypedDict):
    """Type for network graph nodes."""

    type: str
    entropy: float
    frequency: int


class NetworkEdge(TypedDict):
    """Type for network graph edges."""

    weight: int
    pmi: float


# Export types
class SurfaceFormRecord(TypedDict):
    """Type for surface form analysis records."""

    sf_expr: str
    sf_surface_forms: str
    conn_expr: str
    conn_surface_forms: str
    freq: int


class CountRecord(TypedDict):
    """Type for frequency count records."""

    タイプ: str
    ジャンル: str
    機能: str
    細分類: str
    表現: str
    頻度: int


# Cache types
class CacheInfo(TypedDict):
    """Information about cache files."""

    path: str
    size: int
    created: float
    valid: bool


# Analysis types
class SegmentInfo(TypedDict):
    """Information about text segments."""

    start: int
    end: int
    text: str
    dm_matches: List[DmMatch]


# Annotation types
class AnnotationRecord(TypedDict):
    """Type for manual annotation records."""

    rid: str
    section_name: str
    sentence: str
    segment: str
    connective: str
    modality: str
    connective_meidai_check: Optional[bool]
    modality_meidai_check: Optional[bool]


# Corpus types
class CorpusRecord(TypedDict):
    """Type for corpus JSONL records."""

    title: str
    genre: List[str]
    sentences: List[str]


class TextContext(TypedDict):
    """Context information for text processing."""

    title: str
    genre: Union[str, List[str]]
    basename: Optional[str]
    author: Optional[str]
    year: Optional[int]
    sentence_id: Optional[int]
    paragraph_id: Optional[int]
    section: Optional[str]


# Function type aliases
MatchFunction = Callable[[Doc, Any, MatcherProtocol], List[Span]]
PatternExpander = Callable[[Any], List[List[Dict[str, Any]]]]
FilterFunction = Callable[[List[Any]], List[Any]]

# Constants
DM_TYPES = Literal["接続表現", "文末表現"]
