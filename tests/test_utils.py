"""Unit tests for get_affine coordinate-fallback logic."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from rasterix.utils import get_affine


def _make_ds(lats, lons):
    """Minimal xarray Dataset with explicit lat/lon coordinate arrays."""
    return xr.Dataset(coords={"lat": ("lat", lats), "lon": ("lon", lons)})


class TestGetAffineCoordinateFallback:
    """Tests for the coordinate-array fallback path in get_affine."""

    def test_north_to_south_pixel_centers(self):
        """Standard north-to-south array: pixel centers match coord values."""
        lats = np.array([90.0, 89.0, 88.0])  # dy = -1
        lons = np.array([0.0, 1.0, 2.0])  # dx = +1
        ds = _make_ds(lats, lons)
        aff = get_affine(ds, x_dim="lon", y_dim="lat")

        assert aff.a == pytest.approx(1.0)
        assert aff.e == pytest.approx(-1.0)
        # pixel (row=0, col=0) center → should be at (lon[0], lat[0])
        col0_x, row0_y = aff * (0.5, 0.5)
        assert col0_x == pytest.approx(lons[0])
        assert row0_y == pytest.approx(lats[0])

    def test_south_to_north_pixel_centers(self):
        """South-to-north array (dy > 0): pixel centers must match coord values.

        Regression test: the old code used y[-1] as the y-origin when dy > 0,
        causing chunk bboxes to extend north of the data extent (e.g. into
        Canada/Arctic for a CONUS dataset), so every query_bbox returned 0 rows.
        """
        lats = np.array([22.43, 22.43 + 0.000308, 22.43 + 0.000616])  # dy > 0
        lons = np.array([-128.4, -128.4 + 0.000307, -128.4 + 0.000614])
        ds = _make_ds(lats, lons)
        aff = get_affine(ds, x_dim="lon", y_dim="lat")

        assert aff.e == pytest.approx(0.000308)  # positive: south-to-north
        # pixel (row=0, col=0) center → must be at (lons[0], lats[0]), NOT lats[-1]
        col0_x, row0_y = aff * (0.5, 0.5)
        assert col0_x == pytest.approx(lons[0])
        assert row0_y == pytest.approx(lats[0])

    def test_south_to_north_last_pixel(self):
        """Last row of a south-to-north array maps to the northernmost coord."""
        lats = np.linspace(22.0, 52.0, 100)  # dy > 0
        lons = np.linspace(-128.0, -64.0, 200)
        ds = _make_ds(lats, lons)
        aff = get_affine(ds, x_dim="lon", y_dim="lat")

        # pixel center of last row → should be ≈ lats[-1]
        _, last_y = aff * (0.5, len(lats) - 0.5)
        assert last_y == pytest.approx(lats[-1], abs=1e-9)

    def test_north_to_south_bbox_within_data_extent(self):
        """Chunk bbox for a north-to-south array stays within the coordinate range."""
        lats = np.linspace(52.0, 22.0, 100)  # dy < 0
        lons = np.linspace(-128.0, -64.0, 200)
        ds = _make_ds(lats, lons)
        aff = get_affine(ds, x_dim="lon", y_dim="lat")

        # Row 30 offset: y of pixel-center should be within [22, 52]
        _, y30 = aff * (0.5, 30.5)
        assert 22.0 <= y30 <= 52.0

    def test_south_to_north_bbox_within_data_extent(self):
        """Chunk bbox for a south-to-north array stays within the coordinate range.

        Regression test: the old bug shifted bboxes far north of the data.
        """
        lats = np.linspace(22.0, 52.0, 100)  # dy > 0
        lons = np.linspace(-128.0, -64.0, 200)
        ds = _make_ds(lats, lons)
        aff = get_affine(ds, x_dim="lon", y_dim="lat")

        # Row 30 offset: y of pixel-center must be within [22, 52], not above 52
        _, y30 = aff * (0.5, 30.5)
        assert 22.0 <= y30 <= 52.0
