import pytest

from gateway.guard import check, is_typo, numbers


@pytest.mark.parametrize("a,b", [
    ("Convert 10 kilometers to miles.", "Convert 10 miles to kilometers."),
    ("What is the flight time from London to New York?", "What is the flight time from New York to London?"),
    ("Convert 5 feet to meters.", "Convert 6 feet to meters."),
    ("Explain Python 2 print syntax.", "Explain Python 3 print syntax."),
    ("Give me three tips for better sleep.", "Give me ten tips for better sleep."),
    ("How do I encrypt a file with GPG?", "How do I decrypt a file with GPG?"),
    ("What is the capital of Australia?", "What is the capital of Austria?"),
    ("List the advantages of remote work.", "List the disadvantages of remote work."),
])
def test_rejects_near_misses(a, b):
    ok, reason = check(a, b)
    assert not ok, reason


@pytest.mark.parametrize("a,b", [
    ("What is the capital of Japan?", "what is the capital of japan"),
    ("How do I install numpy?", "How do I instal numpy?"),
    ("how do i make a http request in pyhton", "How do I make an HTTP request in Python?"),
    ("How do I reverse a list in Python?", "What's the way to reverse a Python list?"),
    ("Hi! Could you please tell me what the capital of France is? Thanks!", "What is the capital of France?"),
])
def test_accepts_paraphrases(a, b):
    ok, reason = check(a, b)
    assert ok, reason


def test_known_limitation_in_place_synonym_is_rejected():
    # Indistinguishable, lexically, from "encrypt -> decrypt": the guard gives up this saving to stay safe.
    ok, reason = check("How do I convert a string to an integer?", "How do I turn a string into an integer?")
    assert not ok and "convert -> turn" in reason


def test_typo_rules():
    assert is_typo("pyhton", "python") and is_typo("recieve", "receive")
    assert not is_typo("encrypt", "decrypt"), "different first letter"
    assert not is_typo("advantages", "disadvantages"), "length differs by more than one"
    assert not is_typo("austria", "australia")


def test_numbers_include_number_words():
    assert numbers("Give me 5 ideas in two minutes") == ["5", "2"]
