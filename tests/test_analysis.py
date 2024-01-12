from dm_annotations.analysis import create_segments


def test_create_segments(nlp, doc):
    segments = create_segments(doc, [])
    print(segments)
    assert len(segments) == 1

    doc = nlp("")
