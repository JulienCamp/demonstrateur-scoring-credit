"""Test de démarrage du squelette Streamlit."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "streamlit_app.py"


class AppStartupTest(unittest.TestCase):
    def test_app_starts_and_displays_its_main_sections(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

        self.assertEqual(app.exception, [])
        self.assertEqual(
            app.title[0].value,
            "Comprendre une estimation de risque de crédit",
        )
        self.assertEqual(len(app.selectbox), 1)
        self.assertEqual(len(app.selectbox[0].options), 3)
        self.assertGreaterEqual(len(app.metric), 2)
        self.assertEqual(app.metric[0].label, "Risque estimé de défaut")
        self.assertTrue(
            any("Informations du profil" in item.value for item in app.subheader)
        )
        self.assertTrue(
            any("variable sensible" in item.value for item in app.caption)
        )
        self.assertTrue(
            any("Lecture rapide" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any(
                "Comprendre les facteurs affichés" in item.label
                for item in app.expander
            )
        )
        self.assertIn("Démonstration éducative", app.warning[0].value)

    def test_each_profile_renders_without_error(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

        for profile_label in app.selectbox[0].options:
            app.selectbox[0].select(profile_label).run(timeout=10)
            self.assertEqual(app.exception, [])
            self.assertEqual(app.metric[0].label, "Risque estimé de défaut")


if __name__ == "__main__":
    unittest.main()
