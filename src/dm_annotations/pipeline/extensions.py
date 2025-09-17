from spacy.tokens import Doc, Span

from dm_annotations.pipeline.patterns import connectives_classifications

# Register Doc extensions for DM matches only
if not Doc.has_extension("dm_connectives"):
    Doc.set_extension("dm_connectives", default=[])
if not Doc.has_extension("dm_sf"):
    Doc.set_extension("dm_sf", default=[])
if not Doc.has_extension("dm_matches"):
    Doc.set_extension("dm_matches", default=[])

# Keep meta extension for backward compatibility, but metadata should be in user_data
if not Doc.has_extension("meta"):
    Doc.set_extension("meta", default=None)


# Register Span extensions
def get_connectives_classification(s: Span) -> str:
    return connectives_classifications.get(s.label_, "新")


if not Span.has_extension("connective"):
    Span.set_extension("connective", getter=get_connectives_classification)
if not Span.has_extension("modality"):
    Span.set_extension("modality", default=None)
