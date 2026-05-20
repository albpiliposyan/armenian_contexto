import pytest


def test_school_is_closer_to_teacher_than_car(engine):
    teacher_score = engine.hybrid_similarity("ուսուցիչ", "դպրոց")
    car_score = engine.hybrid_similarity("մեքենա", "դպրոց")

    assert teacher_score > car_score


@pytest.mark.parametrize(
    ("guess", "target", "expected_pos"),
    [
        ("ուսուցիչ", "դպրոց", 1.0),
        ("մեքենա", "դպրոց", 1.0),
        ("լավ", "դպրոց", 0.0),
    ],
)
def test_pos_bonus_works(engine, guess, target, expected_pos):
    components = engine.hybrid_components(guess, target)

    assert components["same_pos"] == expected_pos


def test_english_similarity_contributes_to_score(engine):
    teacher_components = engine.hybrid_components("ուսուցիչ", "դպրոց")
    car_components = engine.hybrid_components("մեքենա", "դպրոց")

    assert teacher_components["english_similarity"] > 0.0
    assert (
        teacher_components["english_similarity"]
        > car_components["english_similarity"]
    )


def test_exact_normalized_match_is_still_required_to_win(engine):
    close_guess = engine.get_rank("ուսուցիչ", "դպրոց")
    exact_guess = engine.get_rank("դպրոցը", "դպրոց")

    assert not close_guess["is_correct"]
    assert exact_guess["is_correct"]
    assert exact_guess["rank"] == 1
