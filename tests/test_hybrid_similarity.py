import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from armenian_contexto import ArmenianContextoEngine  # noqa: E402


class HybridSimilarityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ArmenianContextoEngine()

    def test_school_is_closer_to_teacher_than_car(self):
        teacher_score = self.engine.hybrid_similarity("ուսուցիչ", "դպրոց")
        car_score = self.engine.hybrid_similarity("մեքենա", "դպրոց")

        self.assertGreater(teacher_score, car_score)

    def test_category_bonus_uses_external_metadata(self):
        teacher_components = self.engine.hybrid_components("ուսուցիչ", "դպրոց")
        car_components = self.engine.hybrid_components("մեքենա", "դպրոց")

        self.assertEqual(teacher_components["same_category"], 1.0)
        self.assertEqual(car_components["same_category"], 0.0)

    def test_pos_bonus_works(self):
        teacher_components = self.engine.hybrid_components("ուսուցիչ", "դպրոց")
        adjective_components = self.engine.hybrid_components("լավ", "դպրոց")

        self.assertEqual(teacher_components["same_pos"], 1.0)
        self.assertEqual(adjective_components["same_pos"], 0.0)

    def test_english_similarity_contributes_to_score(self):
        teacher_components = self.engine.hybrid_components("ուսուցիչ", "դպրոց")
        car_components = self.engine.hybrid_components("մեքենա", "դպրոց")

        self.assertGreater(teacher_components["english_similarity"], 0.0)
        self.assertGreater(
            teacher_components["english_similarity"],
            car_components["english_similarity"],
        )

    def test_exact_normalized_match_is_still_required_to_win(self):
        close_guess = self.engine.get_rank("ուսուցիչ", "դպրոց")
        exact_guess = self.engine.get_rank("դպրոցը", "դպրոց")

        self.assertFalse(close_guess["is_correct"])
        self.assertTrue(exact_guess["is_correct"])
        self.assertEqual(exact_guess["rank"], 1)


if __name__ == "__main__":
    unittest.main()
