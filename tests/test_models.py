import tempfile
import unittest
from pathlib import Path

from finesteer_moe import resolve_model


class ModelResolutionTest(unittest.TestCase):
    def test_aliases(self):
        llama = resolve_model("llama3.1")
        self.assertEqual(llama.source, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(llama.model_key, "llama31")

        qwen = resolve_model("QWEN2.5")
        self.assertEqual(qwen.source, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(qwen.model_key, "qwen25")

    def test_hub_model_id(self):
        spec = resolve_model("organization/custom-model")
        self.assertEqual(spec.source, "organization/custom-model")
        self.assertEqual(spec.model_key, "custom-model")

    def test_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text("{}", encoding="utf-8")
            spec = resolve_model(str(path))
            self.assertEqual(Path(spec.source), path.resolve())

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            resolve_model("not-a-known-alias")


if __name__ == "__main__":
    unittest.main()
