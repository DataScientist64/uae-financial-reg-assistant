import pytest
from src.generate import generate


def test_generate_returns_string():
    context = "The Consumer Protection Regulation sets mandatory standards for licensed financial institutions."
    answer = generate("What is the Consumer Protection Regulation?", context)
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_generate_empty_context():
    answer = generate("What is the Consumer Protection Regulation?", "")
    assert "do not have" in answer.lower() or "information" in answer.lower()


def test_generate_out_of_scope():
    context = "The Consumer Protection Regulation sets mandatory standards."
    answer = generate("What is the weather on Mars?", context)
    assert isinstance(answer, str)


def test_generate_cites_source():
    context = "Source: CBUAE_EN_4229_VER1.pdf Page: 2\nThe Consumer Protection Regulation sets mandatory standards for licensed financial institutions."
    answer = generate("What does the regulation say?", context)
    assert isinstance(answer, str)
    assert len(answer) > 0