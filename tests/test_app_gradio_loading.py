import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class AppGradioLoadingTests(unittest.TestCase):
    def test_model_discovery_handles_missing_file(self):
        import app.app_gradio as app_gradio

        self.assertIsNotNone(app_gradio)
        self.assertIsNone(app_gradio.discover_model_path())
        self.assertIn("modelo entrenado", app_gradio.model_status_message().lower())


if __name__ == "__main__":
    unittest.main()
