import os
import pytest
import spacy

# Import the specific module where the factory is registered.
# This must be done before spacy.load() is called.
import dm_annotations  # noqa: F401


@pytest.fixture(scope="session")
def model():
    return os.environ.get("SPACY_MODEL", "ja_ginza")


@pytest.fixture(scope="session")
def nlp(model):
    # The import of dm_annotations.pipeline should have registered the factory.
    # Now we can load the model.
    nlp = spacy.load(model)

    # Add the component to the pipeline for tests that need it.
    if "dm_extractor" not in nlp.pipe_names:
        nlp.add_pipe("dm_extractor", last=True)

    # # Verify factory is registered.
    # if "dm_extractor" not in nlp.factory_names:
    #     raise RuntimeError(
    #         f"dm_extractor factory not registered. Available: {list(nlp.factory_names)}"
    #     )

    return nlp


@pytest.fixture
def doc(nlp):
    """Create a doc with metadata already set."""
    # Create a doc object without processing through the full pipeline yet.
    text = """その結果，施策としてコミュニティ･バスによりフォーカスした群で公共交通に対する態度･行動変容効果が示唆された一方，相対的に自動車利用抑制にフォーカスした群においては，自動車利用抑制に対する態度･行動変容効果が見られ，本研究の仮説が支持されたことが示唆された．"""
    doc = nlp.make_doc(text)

    # Set metadata on the unprocessed doc.
    doc.user_data["meta"] = {
        "title": "test",
        "genre": "test",
        "sentence_id": 0,
        "basename": "test",
        "author": "test_author",
        "year": 2024,
        "paragraph_id": None,
        "section": None,
    }

    # Now process the doc through the pipeline, which includes dm_extractor.
    processed_doc = nlp(doc)
    return processed_doc
