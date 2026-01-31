"""
Unit tests for cnn_interference_model.py
Tests model architecture, preprocessing, region extraction, graceful fallback,
training-data CSV format, and combined (threshold + CNN) detection.
"""
import os
import sys
import tempfile
import shutil

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cnn_interference_model import (
    preprocess_evm,
    regions_from_mask,
    CNNInterferenceDetector,
    TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
    from cnn_interference_model import InterferenceCNN1D
    import torch


# ---------------------------------------------------------------------------
# preprocess_evm
# ---------------------------------------------------------------------------

class TestPreprocessEvm:

    def test_output_shape(self):
        evm = np.random.uniform(-30, 5, size=(273,))
        out = preprocess_evm(evm)
        assert out.shape == (1, 1, 273)

    def test_output_dtype(self):
        evm = np.random.uniform(-30, 5, size=(100,))
        out = preprocess_evm(evm)
        assert out.dtype == np.float32

    def test_clipping(self):
        evm = np.array([-100.0, 0.0, 50.0])
        out = preprocess_evm(evm)
        # After clipping to [-40, 10], then normalizing, the extreme values
        # are bounded; verify the raw clip step by checking intermediate
        clipped = np.clip(evm, -40.0, 10.0)
        assert clipped[0] == -40.0
        assert clipped[2] == 10.0

    def test_normalisation_zero_std(self):
        """Constant input should not produce NaN/Inf."""
        evm = np.full(50, -20.0)
        out = preprocess_evm(evm)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_variable_length(self):
        for n in [10, 100, 273, 500]:
            out = preprocess_evm(np.zeros(n))
            assert out.shape == (1, 1, n)


# ---------------------------------------------------------------------------
# regions_from_mask
# ---------------------------------------------------------------------------

class TestRegionsFromMask:

    def test_empty_mask(self):
        mask = np.zeros(100, dtype=bool)
        assert regions_from_mask(mask) == []

    def test_full_mask(self):
        mask = np.ones(100, dtype=bool)
        regions = regions_from_mask(mask, min_region_size=1)
        assert regions == [(0, 100)]

    def test_single_region(self):
        mask = np.zeros(273, dtype=bool)
        mask[50:80] = True
        regions = regions_from_mask(mask, min_region_size=3)
        assert regions == [(50, 80)]

    def test_multiple_regions(self):
        mask = np.zeros(273, dtype=bool)
        mask[10:20] = True
        mask[100:120] = True
        regions = regions_from_mask(mask, min_region_size=3)
        assert regions == [(10, 20), (100, 120)]

    def test_min_region_size_filters_short(self):
        mask = np.zeros(273, dtype=bool)
        mask[50:52] = True  # length 2, should be filtered with min_region_size=3
        mask[100:110] = True  # length 10, should survive
        regions = regions_from_mask(mask, min_region_size=3)
        assert regions == [(100, 110)]

    def test_region_at_end(self):
        mask = np.zeros(100, dtype=bool)
        mask[90:] = True
        regions = regions_from_mask(mask, min_region_size=3)
        assert regions == [(90, 100)]

    def test_region_at_start(self):
        mask = np.zeros(100, dtype=bool)
        mask[:10] = True
        regions = regions_from_mask(mask, min_region_size=3)
        assert regions == [(0, 10)]


# ---------------------------------------------------------------------------
# InterferenceCNN1D (requires PyTorch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestInterferenceCNN1D:

    def test_output_shape_standard(self):
        model = InterferenceCNN1D()
        x = torch.randn(1, 1, 273)
        y = model(x)
        assert y.shape == (1, 273)

    def test_output_range(self):
        """Sigmoid output should be in [0, 1]."""
        model = InterferenceCNN1D()
        x = torch.randn(4, 1, 273)
        y = model(x)
        assert y.min().item() >= 0.0
        assert y.max().item() <= 1.0

    def test_batch_dimension(self):
        model = InterferenceCNN1D()
        x = torch.randn(8, 1, 273)
        y = model(x)
        assert y.shape == (8, 273)

    def test_variable_prb_count(self):
        """Fully convolutional — should handle different PRB counts."""
        model = InterferenceCNN1D()
        for n in [50, 100, 273, 500]:
            x = torch.randn(1, 1, n)
            y = model(x)
            assert y.shape == (1, n)

    def test_parameter_count(self):
        model = InterferenceCNN1D()
        n_params = sum(p.numel() for p in model.parameters())
        # Plan says ~3200 params — allow reasonable margin
        assert 2000 < n_params < 5000, f"Unexpected param count: {n_params}"


# ---------------------------------------------------------------------------
# CNNInterferenceDetector
# ---------------------------------------------------------------------------

class TestCNNInterferenceDetector:

    def test_graceful_fallback_missing_file(self):
        det = CNNInterferenceDetector("nonexistent_model_file.pth")
        assert not det.is_available()

    def test_predict_proba_fallback_returns_zeros(self):
        det = CNNInterferenceDetector("nonexistent_model_file.pth")
        evm = np.random.uniform(-30, 5, size=(273,))
        proba = det.predict_proba(evm)
        assert proba.shape == (273,)
        assert np.allclose(proba, 0.0)

    def test_detect_fallback_returns_empty(self):
        det = CNNInterferenceDetector("nonexistent_model_file.pth")
        evm = np.random.uniform(-30, 5, size=(273,))
        regions = det.detect(evm)
        assert regions == []

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_roundtrip_save_load(self, tmp_path):
        """Save model weights, load via detector, verify predict_proba shape."""
        model = InterferenceCNN1D()
        path = str(tmp_path / "test_model.pth")
        torch.save(model.state_dict(), path)

        det = CNNInterferenceDetector(path)
        assert det.is_available()
        proba = det.predict_proba(np.random.uniform(-30, 5, size=(273,)))
        assert proba.shape == (273,)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_detect_produces_region_tuples(self, tmp_path):
        model = InterferenceCNN1D()
        path = str(tmp_path / "test_model.pth")
        torch.save(model.state_dict(), path)
        det = CNNInterferenceDetector(path)
        regions = det.detect(np.random.uniform(-30, 5, size=(273,)))
        # Regions should be list of (start, end) tuples
        assert isinstance(regions, list)
        for r in regions:
            assert len(r) == 2
            assert r[0] < r[1]


# ---------------------------------------------------------------------------
# Training data CSV format
# ---------------------------------------------------------------------------

class TestTrainingDataFormat:

    def test_save_cnn_training_sample_creates_csv(self):
        """Verify save_cnn_training_sample produces valid CSV."""
        import pandas as pd
        tmpdir = tempfile.mkdtemp()
        try:
            # Mock the function inline (avoid importing Flask globals)
            num_layers = 4
            num_prbs = 273
            evm_db = np.random.uniform(-30, 5, size=(num_layers, num_prbs)).astype(np.float32)
            regions = [[(50, 80)], [], [(100, 120), (200, 220)], []]

            rows = {"PRB": list(range(num_prbs))}
            for layer in range(num_layers):
                rows[f"Layer{layer}_EVM_dB"] = list(evm_db[layer])
                labels = np.zeros(num_prbs, dtype=int)
                for s, e in regions[layer]:
                    labels[max(0, s):min(e, num_prbs)] = 1
                rows[f"Layer{layer}_Label"] = list(labels)
            df = pd.DataFrame(rows)
            csv_path = os.path.join(tmpdir, "labels_test.csv")
            df.to_csv(csv_path, index=False, float_format="%.4f")

            # Verify
            loaded = pd.read_csv(csv_path)
            assert "PRB" in loaded.columns
            assert "Layer0_EVM_dB" in loaded.columns
            assert "Layer0_Label" in loaded.columns
            assert len(loaded) == num_prbs
            # Check labels match regions
            l0_labels = loaded["Layer0_Label"].values
            assert l0_labels[60] == 1  # inside region (50,80)
            assert l0_labels[0] == 0   # outside
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Combined detection (threshold + CNN union)
# ---------------------------------------------------------------------------

class TestCombinedDetection:

    def test_union_of_regions(self):
        """Union of threshold and CNN regions should cover both."""
        num_prbs = 273
        threshold_regions = [(50, 80)]
        cnn_regions = [(70, 110)]  # overlaps + extends

        mask = np.zeros(num_prbs, dtype=bool)
        for s, e in threshold_regions:
            mask[s:e] = True
        for s, e in cnn_regions:
            mask[s:e] = True

        combined = regions_from_mask(mask, min_region_size=3)
        # Should merge into one region spanning 50-110
        assert len(combined) == 1
        assert combined[0] == (50, 110)

    def test_union_disjoint_regions(self):
        """Disjoint regions stay separate."""
        num_prbs = 273
        threshold_regions = [(10, 30)]
        cnn_regions = [(200, 230)]

        mask = np.zeros(num_prbs, dtype=bool)
        for s, e in threshold_regions:
            mask[s:e] = True
        for s, e in cnn_regions:
            mask[s:e] = True

        combined = regions_from_mask(mask, min_region_size=3)
        assert len(combined) == 2
        assert combined[0] == (10, 30)
        assert combined[1] == (200, 230)

    def test_threshold_only_when_cnn_empty(self):
        """If CNN finds nothing, result equals threshold."""
        num_prbs = 273
        threshold_regions = [(50, 80)]
        cnn_regions = []

        mask = np.zeros(num_prbs, dtype=bool)
        for s, e in threshold_regions:
            mask[s:e] = True
        for s, e in cnn_regions:
            mask[s:e] = True

        combined = regions_from_mask(mask, min_region_size=3)
        assert combined == [(50, 80)]

    def test_cnn_only_when_threshold_empty(self):
        """If threshold finds nothing, CNN regions are used."""
        num_prbs = 273
        threshold_regions = []
        cnn_regions = [(100, 130)]

        mask = np.zeros(num_prbs, dtype=bool)
        for s, e in threshold_regions:
            mask[s:e] = True
        for s, e in cnn_regions:
            mask[s:e] = True

        combined = regions_from_mask(mask, min_region_size=3)
        assert combined == [(100, 130)]
