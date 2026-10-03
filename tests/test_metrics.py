"""Tests for the evaluation code.

    python -m unittest discover -s tests -p "test_*.py" -v

The metric tests need only NumPy. The model test runs only when PyTorch and
segmentation-models-pytorch are installed.
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.report import markdown_report, per_class_metrics, summary, top_confusions

ROOT = Path(__file__).resolve().parents[1]
SAVED_CM = ROOT / "results" / "evaluation" / "confusion_matrix.npy"


class MetricTests(unittest.TestCase):
    def test_perfect_prediction(self):
        cm = np.diag([10, 20, 30])
        s = summary(cm)
        self.assertEqual(s["pixel_accuracy"], 1.0)
        self.assertEqual(s["mean_iou_all"], 1.0)

    def test_iou_by_hand(self):
        # class 0: 8 correct, 2 predicted as 1; class 1: 5 correct, 5 predicted as 0
        cm = np.array([[8, 2], [5, 5]])
        iou = per_class_metrics(cm)["iou"]
        self.assertAlmostEqual(iou[0], 8 / (8 + 2 + 5))
        self.assertAlmostEqual(iou[1], 5 / (5 + 5 + 2))

    def test_dice_iou_relation(self):
        m = per_class_metrics(np.array([[50, 10, 0], [5, 30, 5], [0, 5, 20]]))
        np.testing.assert_allclose(m["dice"], 2 * m["iou"] / (1 + m["iou"]))

    def test_absent_class_does_not_crash(self):
        m = per_class_metrics(np.array([[5, 0], [0, 0]]))
        self.assertEqual(m["iou"][1], 0.0)

    def test_top_confusions(self):
        cm = np.array([[9, 1, 0], [7, 3, 0], [0, 2, 8]])
        self.assertEqual(top_confusions(cm, k=1, names=["a", "b", "c"]), [("b", "a", 7)])


@unittest.skipUnless(SAVED_CM.exists(), "saved confusion matrix not found")
class SavedResultTests(unittest.TestCase):
    """The committed confusion matrix must reproduce the committed test report."""

    def test_reproduces_report(self):
        cm = np.load(SAVED_CM)
        self.assertEqual(cm.sum(), 448 * 256 * 256)  # 448 test images at 256 x 256
        s = summary(cm)
        self.assertAlmostEqual(s["mean_iou_no_bg"], 0.6757, places=4)
        self.assertAlmostEqual(s["pixel_accuracy"], 0.8729, places=4)
        self.assertIn("Vehicle", markdown_report(cm))


HAS_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "segmentation_models_pytorch"))


@unittest.skipUnless(HAS_TORCH, "PyTorch / segmentation-models-pytorch not installed")
class ModelTests(unittest.TestCase):
    def test_output_shape(self):
        import torch

        from src.models.model_builder import build_unet
        model = build_unet(encoder_weights=None).eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 256, 256))
        self.assertEqual(tuple(out.shape), (1, 10, 256, 256))


if __name__ == "__main__":
    unittest.main()
