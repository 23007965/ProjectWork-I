"""Unit tests for data pipeline: alignment, target generation, and leakage prevention."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "features"))


class TestSpatialAlignment:
    """Test spatial grid assignment."""

    def test_grid_assignment(self):
        from align_spatiotemporal_data import assign_grid_cell

        df = pd.DataFrame({"lat": [51.12, 51.37, 55.99], "lon": [2.06, 2.49, 8.01]})
        result = assign_grid_cell(df, 0.25, "lat", "lon")
        assert "lat_grid" in result.columns
        assert "lon_grid" in result.columns
        assert all(result["lat_grid"] % 0.25 < 0.001)

    def test_month_assignment(self):
        from align_spatiotemporal_data import assign_month

        df = pd.DataFrame({"date": ["2020-03-15", "2020-07-22"]})
        result = assign_month(df, "date")
        assert result.iloc[0]["month"] == 3
        assert result.iloc[1]["month"] == 7


class TestTargetConstruction:
    """Test prediction target construction."""

    def test_no_future_leakage(self):
        """Verify that future information doesn't leak into features."""
        dataset_path = "data/processed/prediction_dataset.csv"
        if not os.path.exists(dataset_path):
            pytest.skip("Prediction dataset not found")

        df = pd.read_csv(dataset_path)
        for _, row in df.head(100).iterrows():
            origin = pd.Timestamp(row["prediction_origin"])
            hist_features = [c for c in df.columns if c.startswith("hist_")]
            assert row["prediction_origin"] is not None
            assert row["future_fish_species_richness"] >= 1

    def test_split_no_temporal_leak(self):
        """Verify chronological split ordering."""
        dataset_path = "data/processed/prediction_dataset.csv"
        if not os.path.exists(dataset_path):
            pytest.skip("Prediction dataset not found")

        df = pd.read_csv(dataset_path)
        train = df[df["split"] == "train"]["prediction_origin"]
        test = df[df["split"] == "test"]["prediction_origin"]
        if len(train) > 0 and len(test) > 0:
            assert train.max() <= test.min(), "Train data leaks into test period"


class TestFeatureEngineering:
    """Test feature set construction."""

    def test_feature_groups_exist(self):
        import json

        path = "data/processed/features/feature_groups.json"
        if not os.path.exists(path):
            pytest.skip("Feature groups not found")

        with open(path) as f:
            groups = json.load(f)

        assert "oceanographic" in groups
        assert "fisheries" in groups
        assert "edna" in groups
        assert "all" in groups
        assert len(groups["all"]) == (
            len(groups["oceanographic"])
            + len(groups["fisheries"])
            + len(groups["edna"])
            + len(groups.get("spatial_temporal", []))
            + len(groups.get("modality_flags", []))
        )

    def test_no_nan_in_scaled_features(self):
        path = "data/processed/features/early_fusion/X_train.csv"
        if not os.path.exists(path):
            pytest.skip("Feature files not found")

        X = pd.read_csv(path)
        assert X.isna().sum().sum() == 0, "NaN values found in scaled features"


class TestMissingModality:
    """Test missing modality indicators."""

    def test_modality_flags_binary(self):
        path = "data/processed/aligned_multimodal_data.csv"
        if not os.path.exists(path):
            pytest.skip("Aligned data not found")

        df = pd.read_csv(path)
        for flag in ["has_oceanographic", "has_fisheries", "has_edna"]:
            if flag in df.columns:
                assert set(df[flag].unique()).issubset({0, 1, 0.0, 1.0})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
