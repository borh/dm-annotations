from typing import Iterator, List

import ginza  # type: ignore[import]
from spacy.tokens import Doc, Span


def create_segments(doc: Doc, matches: Iterator[Span]) -> List[Doc]:
    """
    Split a Doc into clause‐like segments using GiNZA's bunsetsu spans.

    If no matches are provided, returns a single‐element list [doc].

    :param doc: a parsed spaCy Doc
    :param matches: iterator of Span (connective or final-meta matches)
    :return: list of Doc slices

    >>> import spacy; from spacy.tokens import Doc
    >>> nlp = spacy.blank("en")
    >>> doc = nlp("Hello world.")
    >>> segments = create_segments(doc, iter([]))
    >>> isinstance(segments, list) and len(segments) >= 1
    True
    """
    # if not matches:
    #    return None
    segments: list[Doc] = []
    max_head_idx = -1
    start_idx = 0

    # FIXME try new clause API
    return ginza.clauses(doc)
    for bunsetsu in ginza.bunsetu_spans(doc):
        # Find the furthest head index in the current bunsetsu
        current_max_head_idx = max(token.head.i for token in bunsetsu)
        if current_max_head_idx > max_head_idx:
            max_head_idx = current_max_head_idx
        print(bunsetsu, max_head_idx)

        # If the head is at or beyond the end of the current segment, create a new segment
        if bunsetsu[-1].i >= max_head_idx:
            segment = doc[start_idx : bunsetsu[-1].i + 1]
            segments.append(segment)
            start_idx = bunsetsu[-1].i + 1  # Update start index for the next segment

    # Add any remaining part of the document as the last segment
    if start_idx < len(doc):
        segments.append(doc[start_idx : len(doc)])

    return segments
