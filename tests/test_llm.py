import pytest
import lmql


@pytest.fixture
def cot_prompt():
    with open("example_prompt.lmql") as f:
        return f.read()


def test_lmql(cot_prompt):
    results = lmql.run_sync(cot_prompt)
    print(results)
    assert len(results) > 0
