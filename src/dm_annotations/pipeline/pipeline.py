from spacy.language import Language

from .pipeline_components import DmExtractor


@Language.factory(
    "dm_extractor",
    default_config={
        "match_kinds": ("connectives", "sf"),
        "sf_final_filter": True,
        "strict_connectives": False,
    },
)
def make_dm_extractor(
    nlp,
    name,
    match_kinds: tuple[str, ...],
    sf_final_filter: bool,
    strict_connectives: bool,
) -> DmExtractor:
    """Factory function for DmExtractor component."""
    # GiNZA special‐casing: remove any built‐in sentencizer and insert disable_sentencizer after the parser
    if "disable_sentencizer" in nlp.factory_names:
        for pipe in ("senter", "sentencizer"):
            if pipe in nlp.pipe_names:
                nlp.remove_pipe(pipe)
        nlp.add_pipe("disable_sentencizer", after="parser")

    # Create and return the component
    component = DmExtractor(nlp, match_kinds, sf_final_filter, strict_connectives)

    # Verify the component was created properly
    if not hasattr(component, "matchers") or not component.matchers:
        raise RuntimeError("DmExtractor component created but has no matchers")

    return component
