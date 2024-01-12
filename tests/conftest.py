import pytest
import os
import spacy


@pytest.fixture(scope="session")
def model():
    return os.environ.get("SPACY_MODEL")


@pytest.fixture(scope="session")
def nlp(model):
    return spacy.load(model)


@pytest.fixture(scope="session")
def doc(nlp):
    d = nlp(
        """
その結果，施策としてコミュニティ･バスによりフォーカスした群で公共交通に対する態度･行動変容効果が示唆された一方，相対的に自動車利用抑制にフォーカスした群においては，自動車利用抑制に対する態度･行動変容効果が見られ，本研究の仮説が支持されたことが示唆された．"""
    )
    d.user_data["title"] = "test"
    d.user_data["genre"] = ["test"]
    return d
