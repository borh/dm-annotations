from typing import Iterator
from spacy.tokens import Doc, Span
import ginza


def create_segments(doc: Doc, matches: Iterator[Span]) -> Iterator[Doc] | None:
    """Segments are discourse units containing an
    -   optional connective expression,
    -   the proposition, and
    -   optional sentence ending expression(s).
    If no matches are provided, we return immediately, otherwise proceed to segment.
    To approximate our manual annotation until we have a chunking model,
    we use the dependency parse to find the furthest head of the first bunsetsu,
    and if it is not the end of sentence, chunk the sentence at this head
    and repeat until the end.
    Bunsetsu chunks are obtained from GiNZA's parse."""
    # if not matches:
    #    return None
    segments = []
    max_head_idx = -1
    start_idx = 0

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
