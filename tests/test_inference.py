"""Tests du portage du modèle et de ses artefacts minimaux."""

import json
import unittest

from inference import PROFILES_PATH, load_model, load_payload, predict_profile


class PortableInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = load_payload()
        cls.model = load_model()

    def test_ported_model_reproduces_historical_predictions(self) -> None:
        for profile in self.payload["profiles"]:
            prediction = predict_profile(
                profile["alias"],
                payload=self.payload,
                model=self.model,
            )
            self.assertAlmostEqual(
                prediction.raw_risk,
                profile["expected_raw_risk"],
                places=12,
            )
            self.assertAlmostEqual(
                prediction.calibrated_risk,
                profile["expected_calibrated_risk"],
                places=12,
            )

    def test_explanations_have_both_a_label_and_direction(self) -> None:
        for profile in self.payload["profiles"]:
            prediction = predict_profile(
                profile["alias"],
                payload=self.payload,
                model=self.model,
            )
            self.assertEqual(len(prediction.factors), 6)
            for factor in prediction.factors:
                self.assertTrue(factor.label)
                self.assertTrue(factor.description)
                self.assertIn(
                    factor.direction,
                    ("Vers moins de risque", "Vers plus de risque"),
                )

    def test_public_artifact_contains_no_source_identifier(self) -> None:
        raw_payload = PROFILES_PATH.read_text(encoding="utf-8")
        lowered = raw_payload.lower()

        self.assertNotIn("source_id", lowered)
        self.assertNotIn("sk_id_curr", lowered)
        self.assertNotIn("101041", raw_payload)
        self.assertNotIn("100170", raw_payload)
        self.assertNotIn("100038", raw_payload)
        profiles = json.loads(raw_payload)["profiles"]
        payload = json.loads(raw_payload)
        self.assertEqual(payload["model"]["raw_feature_count"], 120)
        self.assertEqual(payload["model"]["feature_count"], 158)
        self.assertEqual(len(profiles), 3)
        for profile in profiles:
            self.assertEqual(
                set(profile["profile_summary"]),
                {
                    "Situation personnelle",
                    "Activité et ressources",
                    "Crédit et patrimoine",
                },
            )

    def test_unknown_profile_is_rejected_cleanly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Profil inconnu"):
            predict_profile(
                "Profil absent",
                payload=self.payload,
                model=self.model,
            )


if __name__ == "__main__":
    unittest.main()
