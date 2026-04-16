"""Tests for GeoParquetSource and its integration with rasterize()."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import xproj  # noqa: F401 — registers xproj accessor
from shapely.geometry import box

pytest.importorskip("duckdb", reason="duckdb is required for these tests")

import geopandas as gpd

from rasterix.rasterize import rasterize
from rasterix.rasterize.sources import GeoParquetSource

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_gdf():
    """Three non-overlapping boxes."""
    return gpd.GeoDataFrame(
        {"region_id": [10, 20, 30]},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3), box(5, 5, 6, 6)],
        crs="EPSG:4326",
    )


@pytest.fixture
def parquet_path(tmp_path, small_gdf):
    path = str(tmp_path / "test.parquet")
    small_gdf.to_parquet(path)
    return path


@pytest.fixture
def parquet_path_with_bbox(tmp_path, small_gdf):
    """GeoParquet written with a covering bbox column (GeoParquet 1.1)."""
    path = str(tmp_path / "test_bbox.parquet")
    small_gdf.to_parquet(path, write_covering_bbox=True)
    return path


@pytest.fixture
def dataset():
    with xr.tutorial.open_dataset("eraint_uvz") as ds:
        ds = ds.load()
        ds = ds.proj.assign_crs(spatial_ref="epsg:4326")
        ds["spatial_ref"].attrs = ds.proj.crs.to_cf()
        return ds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ids(tbl) -> set:
    """Extract id values from a query_bbox Arrow table as a Python set."""
    return set(tbl.column("id").to_pylist())


# ---------------------------------------------------------------------------
# Unit test: _rusterize_chunk burn value correctness
# ---------------------------------------------------------------------------


def test_rusterize_chunk_burns_correct_id():
    """_rusterize_chunk must burn the actual integer ID from the Arrow table.

    Regression test: an earlier version produced values near int32.max instead
    of the actual IDs when the polars DataFrame was not constructed correctly.
    """
    pytest.importorskip("rusterize")
    import pyarrow as pa
    import shapely
    from affine import Affine

    from rasterix.rasterize.duckdb import _rusterize_chunk

    # Single 1×1 box covering exactly one pixel, burned with ID 42.
    wkb = shapely.to_wkb(shapely.box(0, 0, 1, 1))
    arrow_tbl = pa.table(
        {"wkb": pa.array([wkb], type=pa.large_binary()), "id": pa.array([42], type=pa.int64())}
    )
    # Affine: 1×1 pixel whose extent is (0, 0) → (1, 1)
    affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1.0)
    result = _rusterize_chunk(arrow_tbl, affine, (1, 1), "replace", False)

    assert result.shape == (1, 1)
    assert result.dtype == np.int32
    assert result[0, 0] == 42, f"expected burn value 42, got {result[0, 0]}"


# ---------------------------------------------------------------------------
# Unit tests: GeoParquetSource.query_bbox
# ---------------------------------------------------------------------------


def test_query_bbox_returns_arrow_table(parquet_path):
    import pyarrow as pa

    source = GeoParquetSource(path=parquet_path)
    tbl = source.query_bbox(0.5, 0.5, 2.5, 2.5)
    assert isinstance(tbl, pa.Table)
    assert tbl.schema.names == ["wkb", "id"]
    assert pa.types.is_large_binary(tbl.schema.field("wkb").type) or pa.types.is_binary(
        tbl.schema.field("wkb").type
    )
    assert pa.types.is_integer(tbl.schema.field("id").type)


def test_query_bbox_auto_id(parquet_path):
    """row_number()-based auto IDs: two of three boxes intersect the bbox."""
    source = GeoParquetSource(path=parquet_path)
    tbl = source.query_bbox(0.5, 0.5, 2.5, 2.5)
    assert tbl.num_rows == 2
    # auto IDs are 0-based (row_number() - 1), matching the GeoDataFrame path
    for id_ in tbl.column("id").to_pylist():
        assert isinstance(id_, int)
        assert id_ >= 0


def test_query_bbox_explicit_id(parquet_path):
    source = GeoParquetSource(path=parquet_path, id_column="region_id")
    tbl = source.query_bbox(0.5, 0.5, 2.5, 2.5)
    assert tbl.num_rows == 2
    assert _ids(tbl) == {10, 20}


def test_query_bbox_no_hits(parquet_path):
    source = GeoParquetSource(path=parquet_path)
    tbl = source.query_bbox(10, 10, 11, 11)
    assert tbl.num_rows == 0


def test_query_bbox_all_hits(parquet_path):
    source = GeoParquetSource(path=parquet_path)
    tbl = source.query_bbox(-1, -1, 10, 10)
    assert tbl.num_rows == 3


def test_auto_id_stable_across_calls(parquet_path):
    """rowid must be the same for a given row regardless of the bbox queried."""
    source = GeoParquetSource(path=parquet_path)
    tbl_all = source.query_bbox(-1, -1, 10, 10)
    tbl_first = source.query_bbox(0.5, 0.5, 1.5, 1.5)  # only first box

    ids_all = _ids(tbl_all)
    ids_first = _ids(tbl_first)

    assert ids_first.issubset(ids_all)
    assert len(ids_all) == 3


def test_bbox_column_filter_matches_no_bbox(parquet_path, parquet_path_with_bbox):
    """Explicit bbox pre-filter must return the same rows as no pre-filter."""
    source_plain = GeoParquetSource(path=parquet_path, id_column="region_id")
    source_bbox = GeoParquetSource(path=parquet_path_with_bbox, id_column="region_id", bbox_column="bbox")
    for q in [(0.5, 0.5, 2.5, 2.5), (-1, -1, 10, 10), (10, 10, 11, 11)]:
        assert _ids(source_plain.query_bbox(*q)) == _ids(source_bbox.query_bbox(*q))


# ---------------------------------------------------------------------------
# Integration tests: rasterize() with GeoParquetSource
# ---------------------------------------------------------------------------


def test_rasterize_geoparquet_exactextract_coverage_matches_gdf(tmp_path, dataset):
    """exactextract engine: covered pixels must match the GeoDataFrame path."""
    pytest.importorskip("exactextract")

    world = pytest.importorskip("geodatasets")
    world_gdf = gpd.read_file(world.get_path("naturalearth land"))
    path = str(tmp_path / "world.parquet")
    world_gdf[["geometry"]].to_parquet(path)

    chunked = dataset.chunk(latitude=119, longitude=-1)

    gdf_result = rasterize(
        chunked, world_gdf[["geometry"]], xdim="longitude", ydim="latitude", engine="exactextract"
    )
    covered_gdf = (gdf_result != len(world_gdf)).compute()

    source = GeoParquetSource(path=path)
    parquet_result = rasterize(chunked, source, xdim="longitude", ydim="latitude", engine="exactextract")
    covered_parquet = (parquet_result != np.iinfo(np.int32).min).compute()

    xr.testing.assert_equal(covered_gdf, covered_parquet)


def test_rasterize_geoparquet_exactextract_all_touched_raises(tmp_path, dataset):
    """exactextract does not support all_touched=True."""
    pytest.importorskip("exactextract")

    world = pytest.importorskip("geodatasets")
    world_gdf = gpd.read_file(world.get_path("naturalearth land"))
    path = str(tmp_path / "world.parquet")
    world_gdf[["geometry"]].to_parquet(path)

    chunked = dataset.chunk(latitude=119, longitude=-1)
    source = GeoParquetSource(path=path)

    with pytest.raises(NotImplementedError, match="all_touched"):
        rasterize(
            chunked,
            source,
            xdim="longitude",
            ydim="latitude",
            engine="exactextract",
            all_touched=True,
        ).compute()


@pytest.mark.parametrize("backend", ["rasterio", "rusterize"])
def test_rasterize_geoparquet_returns_dask(tmp_path, dataset, backend):
    """Output is always a dask array for GeoParquetSource."""
    pytest.importorskip(backend)

    world = pytest.importorskip("geodatasets")
    world_gdf = gpd.read_file(world.get_path("naturalearth land"))
    path = str(tmp_path / "world.parquet")
    world_gdf[["geometry"]].to_parquet(path)

    chunked = dataset.chunk(latitude=119, longitude=-1)
    source = GeoParquetSource(path=path)
    result = rasterize(chunked, source, xdim="longitude", ydim="latitude", engine=backend)
    assert result.chunks is not None, "Output must be a dask array"
    assert result.dtype == np.int32


@pytest.mark.parametrize("backend", ["rasterio", "rusterize"])
def test_rasterize_geoparquet_coverage_matches_gdf(tmp_path, dataset, backend):
    """Pixels covered by GeoParquetSource must match the GeoDataFrame path."""
    pytest.importorskip(backend)

    world = pytest.importorskip("geodatasets")
    world_gdf = gpd.read_file(world.get_path("naturalearth land"))
    path = str(tmp_path / "world.parquet")
    world_gdf[["geometry"]].to_parquet(path)

    chunked = dataset.chunk(latitude=119, longitude=-1)

    # Reference result via the GeoDataFrame path
    gdf_result = rasterize(
        chunked, world_gdf[["geometry"]], xdim="longitude", ydim="latitude", engine=backend
    )
    covered_gdf = (gdf_result != len(world_gdf)).compute()

    # DuckDB path
    source = GeoParquetSource(path=path)
    parquet_result = rasterize(chunked, source, xdim="longitude", ydim="latitude", engine=backend)
    covered_parquet = (parquet_result != np.iinfo(np.int32).min).compute()

    xr.testing.assert_equal(covered_gdf, covered_parquet)


def test_rasterize_geoparquet_lazy(tmp_path, dataset):
    """Calling rasterize() with GeoParquetSource must not trigger computation."""
    from xarray.tests import raise_if_dask_computes

    pytest.importorskip("rasterio")
    world = pytest.importorskip("geodatasets")
    world_gdf = gpd.read_file(world.get_path("naturalearth land"))
    path = str(tmp_path / "world.parquet")
    world_gdf[["geometry"]].to_parquet(path)

    chunked = dataset.chunk(latitude=119, longitude=-1)
    source = GeoParquetSource(path=path)

    with raise_if_dask_computes():
        _ = rasterize(chunked, source, xdim="longitude", ydim="latitude", engine="rasterio")
