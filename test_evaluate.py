from types import SimpleNamespace
from evaluate import assess, CASES
from rag import Match
from guide_loader import load_guide


def _answer(ids, supported=True, text="Answer [G01]"):
    guide = {e['id']: e for e in load_guide()}
    return SimpleNamespace(matches=[Match(guide[i], 0.5) for i in ids],
                           supported=supported, text=text)


def test_all_six_cases_have_expected_categories():
    assert len(CASES) == 6
    assert [x[0] for x in CASES] == ["Answerable"]*3 + ["Paraphrase"] + ["Outside guide"]*2


def test_answerable_case_passes_with_good_retrieval_and_citation():
    assert assess(_answer(["G01", "G05"], text="Imperial City [G01]"), {"G01"}, None) == (True, "—")


def test_answerable_case_reports_retrieval_failure():
    assert assess(_answer(["G05", "G03"], text="River [G05]"), {"G01"}, None) == (False, "retrieval")


def test_answerable_case_reports_generation_failure():
    assert assess(_answer(["G01"], supported=False, text="I don't know"), {"G01"}, None) == (False, "generation")


def test_outside_guide_correctly_refused():
    assert assess(_answer(["G01"], supported=False, text="I don't know"), set(), False) == (True, "—")


def test_outside_guide_answer_is_failure():
    assert assess(_answer(["G01"], supported=True), set(), False) == (False, "generation")
