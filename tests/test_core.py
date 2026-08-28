import unittest

import torch

from finesteer_moe import MoSE, MoSEConfig, build_mose_components, train_mose


class MoSECoreTest(unittest.TestCase):
    def test_supported_presets_build_and_train(self):
        generator = torch.Generator().manual_seed(0)
        queries = torch.randn(80, 32, generator=generator)
        deltas = torch.randn(80, 32, generator=generator)
        for name in ("MoSE", "orthogonal_residual"):
            cfg = MoSEConfig.preset(name, residual_dim=6)
            prototypes, basis, metadata = build_mose_components(deltas, cfg)
            self.assertEqual(prototypes.shape[1], 32)
            self.assertEqual(basis.shape, (32, 6))
            self.assertGreaterEqual(metadata["selection"]["chosen_k"], cfg.k_min)
            model = MoSE(prototypes, basis, value_projection=cfg.value_projection, attention_dim=16)
            result = train_mose(model, queries, deltas, epochs=2, patience=1)
            self.assertEqual(model(queries[:3]).shape, (3, 32))
            self.assertTrue(torch.isfinite(model(queries[:3])).all())
            self.assertGreaterEqual(result["epochs_ran"], 1)

    def test_mose_is_default_and_removed_presets_fail(self):
        self.assertEqual(MoSEConfig.preset().name, "MoSE")
        for removed in ("zip_base", "zip_delta_pca"):
            with self.assertRaises(ValueError):
                MoSEConfig.preset(removed)


if __name__ == "__main__":
    unittest.main()
