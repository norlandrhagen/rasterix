"""Tests for GeoParquetSource and its integration with rasterize()."""

from __future__ import annotations

import numpy as np
import pyproj
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


# ---------------------------------------------------------------------------
# Integration tests: coverage() with GeoParquetSource
# ---------------------------------------------------------------------------


def test_coverage_geoparquet_matches_gdf(tmp_path, dataset):
    """DuckDB coverage output must match GeoDataFrame-based coverage pixel-for-pixel."""
    pytest.importorskip("exactextract")
    geodatasets = pytest.importorskip("geodatasets")
    from rasterix.rasterize.exact import coverage

    world_gdf = gpd.read_file(geodatasets.get_path("naturalearth land"))
    path = str(tmp_path / "world.parquet")
    world_gdf[["geometry"]].to_parquet(path)

    chunked = dataset.chunk(latitude=119, longitude=-1)

    expected = (
        coverage(chunked, world_gdf[["geometry"]], xdim="longitude", ydim="latitude")
        .compute()
        .data.todense()
    )

    source = GeoParquetSource(path=path, crs="EPSG:4326")
    actual = (
        coverage(chunked, source, xdim="longitude", ydim="latitude")
        .compute()
        .data.todense()
    )

    np.testing.assert_array_equal(expected, actual)


def test_coverage_geoparquet_lazy(tmp_path, dataset):
    """coverage() with GeoParquetSource must not trigger dask computation."""
    pytest.importorskip("exactextract")
    from xarray.tests import raise_if_dask_computes

    from rasterix.rasterize.exact import coverage

    geodatasets = pytest.importorskip("geodatasets")
    world_gdf = gpd.read_file(geodatasets.get_path("naturalearth land"))
    path = str(tmp_path / "world.parquet")
    world_gdf[["geometry"]].to_parquet(path)

    chunked = dataset.chunk(latitude=119, longitude=-1)
    source = GeoParquetSource(path=path, crs="EPSG:4326")

    with raise_if_dask_computes():
        result = coverage(chunked, source, xdim="longitude", ydim="latitude")

    assert hasattr(result.data, "dask"), "output must be a dask-backed array"


def test_coverage_geoparquet_global_id_placement(tmp_path):
    """Each geometry must appear at its global file_row_number index in the output."""
    pytest.importorskip("exactextract")
    from rasterix.rasterize.exact import coverage

    # Three non-overlapping 1×1 boxes; each will cover a distinct spatial region.
    gdf = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3), box(5, 5, 6, 6)],
        crs="EPSG:4326",
    )
    path = str(tmp_path / "boxes.parquet")
    gdf.to_parquet(path)

    # Build a small raster covering [0,7] × [0,7] at 0.5° resolution.
    import xproj  # noqa: F401

    ds = xr.Dataset(
        coords={
            "spatial_ref": ((), 0, pyproj.CRS.from_epsg(4326).to_cf()),
            "x": (["x"], np.arange(0.25, 7, 0.5)),
            "y": (["y"], np.arange(6.75, 0, -0.5)),
        }
    )
    ds = ds.proj.assign_crs(spatial_ref="epsg:4326")
    import rasterix

    ds = rasterix.assign_index(ds)

    source = GeoParquetSource(path=path, crs="EPSG:4326")
    result = coverage(ds.chunk(x=7, y=7), source).compute()

    for geom_idx, (xlo, ylo, xhi, yhi) in enumerate([(0, 0, 1, 1), (2, 2, 3, 3), (5, 5, 6, 6)]):
        # The geometry at global index geom_idx must have non-zero coverage
        # only within its own bounding box.
        geom_slice = result.isel(geometry=geom_idx).data.todense()
        x_vals = result.x.values
        y_vals = result.y.values
        inside_x = (x_vals >= xlo) & (x_vals <= xhi)
        inside_y = (y_vals >= ylo) & (y_vals <= yhi)
        outside_x = ~inside_x
        outside_y = ~inside_y
        # Non-zero coverage exists inside the box
        assert geom_slice[np.ix_(inside_y, inside_x)].sum() > 0
        # No coverage outside the box
        assert geom_slice[np.ix_(outside_y, outside_x)].sum() == 0


def test_coverage_geoparquet_empty_result(tmp_path, dataset):
    """Coverage of a raster with no overlapping geometries must be all-zero."""
    pytest.importorskip("exactextract")
    from rasterix.rasterize.exact import coverage

    gdf = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs="EPSG:4326",
    )
    path = str(tmp_path / "boxes.parquet")
    gdf.to_parquet(path)

    # eraint_uvz covers the globe, so clip to a region far from both boxes.
    # Boxes are at 0-3° lon/lat; select only 90-180° lon.
    clipped = dataset.sel(longitude=slice(90, 180)).chunk(latitude=-1, longitude=90)
    source = GeoParquetSource(path=path, crs="EPSG:4326")
    result = coverage(clipped, source, xdim="longitude", ydim="latitude").compute()

    assert result.data.nnz == 0, "expected no coverage for non-overlapping extent"


@pytest.mark.parametrize("weight,expected_dtype", [("fraction", np.float64), ("none", np.uint8)])
def test_coverage_geoparquet_dtype(tmp_path, dataset, weight, expected_dtype):
    """coverage_weight controls output dtype: 'none' → uint8, others → float64."""
    pytest.importorskip("exactextract")
    from rasterix.rasterize.exact import coverage

    geodatasets = pytest.importorskip("geodatasets")
    world_gdf = gpd.read_file(geodatasets.get_path("naturalearth land"))
    path = str(tmp_path / "world.parquet")
    world_gdf[["geometry"]].to_parquet(path)

    source = GeoParquetSource(path=path, crs="EPSG:4326")
    result = coverage(
        dataset.chunk(latitude=119, longitude=-1),
        source,
        xdim="longitude",
        ydim="latitude",
        coverage_weight=weight,
    ).compute()

    assert result.data.dtype == expected_dtype
