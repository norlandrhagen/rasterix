"""Regression tests for south-up affine handling in the rusterize backend.

rusterize takes an extent and resolution — not a signed affine — so its output
is always north-up (row 0 = ymax). When the caller's grid is south-up
(``affine.e > 0``), the output must be flipped vertically to match the grid's
row→latitude mapping. These tests pin that behavior.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

pytest.importorskip("rusterize")


def _north_up_affine() -> Affine:
    # pixel (0, 0) top-left at (0, 2); y decreases downward.
    return Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0)


def _south_up_affine() -> Affine:
    # pixel (0, 0) bottom-left at (0, 0); y increases downward.
    return Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def test_rasterize_geometries_south_up_matches_rasterio():
    """South-up affine: rusterize output must match rasterio pixel-for-pixel."""
    rasterio = pytest.importorskip("rasterio.features")

    from rasterix.rasterize.rusterize import rasterize_geometries

    # Geometry in the lower-left cell of a 2×2 grid.
    geoms = [box(0, 0, 1, 1)]
    shape = (2, 2)
    affine = _south_up_affine()

    rust_out = rasterize_geometries(
        geoms,
        dtype=np.dtype("uint8"),
        shape=shape,
        affine=affine,
        offset=7,
        fill=0,
        merge_alg="last",
    )

    rio_out = rasterio.rasterize(
        [(geoms[0], 7)],
        out_shape=shape,
        transform=affine,
        fill=0,
        dtype="uint8",
    )

    np.testing.assert_array_equal(rust_out, rio_out)
    # In a south-up grid, row 0 corresponds to y ∈ [0, 1) — the cell the
    # geometry actually occupies.
    assert rust_out[0, 0] == 7
    assert rust_out[1, 0] == 0


def test_rasterize_geometries_north_up_unchanged():
    """North-up path must remain identical to rasterio (no-op for flip)."""
    rasterio = pytest.importorskip("rasterio.features")

    from rasterix.rasterize.rusterize import rasterize_geometries

    geoms = [box(0, 0, 1, 1)]
    shape = (2, 2)
    affine = _north_up_affine()

    rust_out = rasterize_geometries(
        geoms,
        dtype=np.dtype("uint8"),
        shape=shape,
        affine=affine,
        offset=7,
        fill=0,
        merge_alg="last",
    )

    rio_out = rasterio.rasterize(
        [(geoms[0], 7)],
        out_shape=shape,
        transform=affine,
        fill=0,
        dtype="uint8",
    )

    np.testing.assert_array_equal(rust_out, rio_out)
    # North-up: row 1 is y ∈ [0, 1) — the geometry's cell.
    assert rust_out[1, 0] == 7
    assert rust_out[0, 0] == 0


def test_np_geometry_mask_south_up_matches_rasterio():
    rasterio = pytest.importorskip("rasterio.features")

    from rasterix.rasterize.rusterize import np_geometry_mask

    geoms = [box(0, 0, 1, 1)]
    shape = (2, 2)
    affine = _south_up_affine()

    rust_mask = np_geometry_mask(geoms, shape=shape, affine=affine, invert=True)
    rio_mask = rasterio.geometry_mask(geoms, out_shape=shape, transform=affine, invert=True)

    np.testing.assert_array_equal(rust_mask, rio_mask)
    # South-up: inside-mask must be True at row 0, not row 1.
    assert bool(rust_mask[0, 0]) is True
    assert bool(rust_mask[1, 0]) is False


def test_rusterize_chunk_south_up_places_correct_row():
    """_rusterize_chunk: south-up affine must burn at the geometry's actual row."""
    pytest.importorskip("duckdb")
    import pyarrow as pa
    import shapely

    from rasterix.rasterize.duckdb import _rusterize_chunk

    wkb = shapely.to_wkb(shapely.box(0, 0, 1, 1))
    arrow_tbl = pa.table(
        {"wkb": pa.array([wkb], type=pa.large_binary()), "id": pa.array([42], type=pa.int64())}
    )

    result = _rusterize_chunk(arrow_tbl, _south_up_affine(), (2, 2), "replace", False)

    assert result.shape == (2, 2)
    assert result[0, 0] == 42, "south-up: row 0 is y∈[0,1), which is where the geom lives"
    assert result[1, 0] == np.iinfo(np.int32).min
