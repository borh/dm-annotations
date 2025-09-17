import spacy
from spacy.language import Language

import dm_annotations.pipeline.extensions  # noqa: F401  registers Doc/Span extensions

# Import factory and extension registration immediately on dm_annotations import
from dm_annotations.pipeline.pipeline import (
    make_dm_extractor,  # noqa: F401  registers the factory
)


def load_core_nlp(model: str = "ja_ginza", **kwargs) -> Language:
    """
    Load the 'core' spaCy pipeline (tokenize, tag, parse) but do NOT add dm_extractor yet.
    GiNZA special-casing: remove built-in sentencizers and insert disable_sentencizer.
    """
    spacy.prefer_gpu()
    nlp = spacy.load(model, **kwargs)
    if "disable_sentencizer" in nlp.factory_names:
        for pipe in ("senter", "sentencizer"):
            if pipe in nlp.pipe_names:
                nlp.remove_pipe(pipe)
        nlp.add_pipe("disable_sentencizer", after="parser")
    return nlp


def load_dm_nlp(model: str = "ja_ginza", **kwargs) -> Language:
    """
    Build on load_core_nlp(*) and append our dm_extractor at the end.
    """
    nlp = load_core_nlp(model, **kwargs)
    if "dm_extractor" not in nlp.pipe_names:
        nlp.add_pipe("dm_extractor", last=True)
    return nlp
