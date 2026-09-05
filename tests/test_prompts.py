import unittest

from racesort.prompts import (
    CAR_PROFILE_PROMPT,
    DIRECT_NUMBER_PROMPT,
    MOTORCYCLE_PROFILE_PROMPT,
    build_ocr_verification_prompt,
    get_profile_prompt,
)


class NumberPromptTests(unittest.TestCase):
    def test_direct_prompt_protects_string_identifiers(self):
        self.assertIn("Preserve leading zeros exactly", DIRECT_NUMBER_PROMPT)
        self.assertIn("digits and letters", DIRECT_NUMBER_PROMPT)
        self.assertIn("Never guess", DIRECT_NUMBER_PROMPT)

    def test_ocr_prompt_preserves_candidate_order_and_spelling(self):
        prompt = build_ocr_verification_prompt(["007", "A12", "0"])

        self.assertIn("007, A12, 0", prompt)
        self.assertIn("Do not invent a new number", prompt)
        self.assertIn("UNKNOWN", prompt)


class ProfilePromptTests(unittest.TestCase):
    def test_motorcycle_profile_prompt_is_selected(self):
        self.assertIs(
            get_profile_prompt("motorcycle"),
            MOTORCYCLE_PROFILE_PROMPT,
        )
        self.assertIn('"leathers_colors"', MOTORCYCLE_PROFILE_PROMPT)

    def test_car_profile_prompt_is_selected(self):
        self.assertIs(get_profile_prompt("car"), CAR_PROFILE_PROMPT)
        self.assertIn('"model"', CAR_PROFILE_PROMPT)

    def test_unknown_race_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "race_type"):
            get_profile_prompt("boat")


if __name__ == "__main__":
    unittest.main()
