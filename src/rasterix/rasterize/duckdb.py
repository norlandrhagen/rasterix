"""Per-chunk rasterization helpers for GeoParquetSource."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from affine import Affine

if TYPE_CHECKING:
    import pyarrow

__all__: list[str] = []

_FILL = np.iinfo(np.int32).min  # sentinel: pixel not covered by any geometry


def _chunk_bbox(affine: Affine, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) for a chunk defined by *affine* and *shape*."""
    nrows, ncols = shape
    xmin = affine.c
    ymax = affine.f
    xmax = xmin + ncols * affine.a
    ymin = ymax + nrows * affine.e  # e is typically negative
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    return xmin, ymin, xmax, ymax


def duckdb_chunk(
    chunk: np.ndarray,
    block_info: dict | None = None,
    *,
    source,
    raster_affine: Affine,
    engine: str,
    all_touched: bool = False,
    merge_alg: str = "replace",
) -> np.ndarray:
    """``dask.array.map_blocks`` worker: query + rasterize for one spatial chunk.

    Parameters
    ----------
    chunk : np.ndarray
        Template array for the current chunk (values ignored; shape used).
    block_info : dict
        Injected by dask — contains ``"array-location"`` for this block.
    source : GeoParquetSource
        The geometry source to query.
    raster_affine : Affine
        Affine transform of the full raster (top-left corner, global coords).
    engine : {"rasterio", "rusterize"}
        Which rasterization backend to use.
    all_touched : bool
        Passed through to the engine.
    merge_alg : str
        ``"replace"`` or ``"add"`` (rusterize also accepts ``"first"``, etc.).

    Returns
    -------
    np.ndarray
        ``int32`` array of shape ``(nrows, ncols)``.  Pixels not covered by any
        geometry are set to ``np.iinfo(np.int32).min``.
    """
    (row_start, row_end), (col_start, col_end) = block_info[0]["array-location"]
    nrows = row_end - row_start
    ncols = col_end - col_start
    shape = (nrows, ncols)

    # Shift global affine so that pixel (0, 0) maps to this chunk's top-left.
    chunk_affine = raster_affine * Affine.translation(col_start, row_start)
    xmin, ymin, xmax, ymax = _chunk_bbox(chunk_affine, shape)

    # Arrow table with columns 'wkb' (Binary) and 'id' (Int64)
    arrow_tbl = source.query_bbox(xmin, ymin, xmax, ymax)

    if arrow_tbl.num_rows == 0:
        return np.full(shape, _FILL, dtype=np.int32)

    if engine == "rusterize":
        return _rusterize_chunk(arrow_tbl, chunk_affine, shape, merge_alg, all_touched)
    elif engine == "exactextract":
        return _exactextract_chunk(arrow_tbl, chunk_affine, shape, merge_alg, all_touched)
    else:
        return _rasterio_chunk(arrow_tbl, chunk_affine, shape, merge_alg, all_touched)


# ---------------------------------------------------------------------------
# Engine-specific helpers
# ---------------------------------------------------------------------------


def _rusterize_chunk(
    arrow_tbl: pyarrow.Table,
    chunk_affine: Affine,
    shape: tuple[int, int],
    merge_alg: str,
    all_touched: bool,
) -> np.ndarray:
    import polars as pl
    import polars_st  # noqa: F401 — registers .st accessor; rusterize calls .st.srid() internally
    import rusterize as rust

    from .rusterize import _affine_to_extent_and_res

    alg_map = {"replace": "last", "add": "sum"}
    fun = alg_map.get(merge_alg, merge_alg)

    # Zero-copy: Arrow → polars shares the same memory layout.
    # polars_st.geom() re-registers the Binary WKB column as a geometry series
    # so rusterize can call .st methods on it (e.g. .st.srid() internally).
    # polars_st stores geometry as WKB, so this is a dtype annotation, not a
    # conversion.  Keep "value" as Int64; output dtype is controlled via
    # dtype="int32".
    df = (
        pl.from_arrow(arrow_tbl)
        .rename({"wkb": "geometry", "id": "value"})
        .with_columns(polars_st.geom("geometry"))
    )

    extent, (xres, yres) = _affine_to_extent_and_res(chunk_affine, shape)
    result = rust.rusterize(
        df,
        res=(xres, yres),
        extent=extent,
        out_shape=shape,
        field="value",
        fun=fun,
        background=_FILL,
        all_touched=all_touched,
        encoding="numpy",
        dtype="int32",
    )
    if result.ndim == 3 and result.shape[0] == 1:
        result = result.squeeze(axis=0)
    return result.astype(np.int32)


def _exactextract_chunk(
    arrow_tbl: pyarrow.Table,
    chunk_affine: Affine,
    shape: tuple[int, int],
    merge_alg: str,
    all_touched: bool,
) -> np.ndarray:
    import shapely

    from .exact import _rasterize_with_exact

    if all_touched:
        raise NotImplementedError("all_touched=True is not supported by the exactextract engine.")

    geoms = shapely.from_wkb(arrow_tbl.column("wkb").to_pylist())
    ids = np.array(arrow_tbl.column("id").to_pylist(), dtype=np.int32)
    return _rasterize_with_exact(
        geoms, ids, shape=shape, affine=chunk_affine, merge_alg=merge_alg, fill=_FILL
    )


def duckdb_coverage_chunk(
    chunk: np.ndarray,
    block_info: dict | None = None,
    *,
    source,
    raster_affine: Affine,
    n_geoms: int,
    coverage_weight: str,
    strategy: str,
) -> "sparse.COO":
    """``dask.array.map_blocks`` worker: query + coverage for one spatial chunk.

    The template passed to map_blocks has shape ``(n_geoms, y, x)`` with the
    geometry dimension as a single chunk, so ``block_info`` gives:
    ``[(0, n_geoms), (row_start, row_end), (col_start, col_end)]``.

    Returns a sparse COO of shape ``(n_geoms, chunk_h, chunk_w)`` with
    geometry coordinates set to global ``file_row_number`` positions.
    """
    import geopandas as gpd
    import shapely
    import sparse

    from .exact import np_coverage

    # Geometry dim is dim 0 (single chunk of n_geoms); spatial dims are 1 and 2.
    _, (row_start, row_end), (col_start, col_end) = block_info[0]["array-location"]
    shape = (row_end - row_start, col_end - col_start)
    dtype = np.uint8 if coverage_weight == "none" else np.float64

    chunk_affine = raster_affine * Affine.translation(col_start, row_start)
    xmin, ymin, xmax, ymax = _chunk_bbox(chunk_affine, shape)

    arrow_tbl = source.query_bbox(xmin, ymin, xmax, ymax)

    empty = sparse.COO([], data=np.array([], dtype=dtype), shape=(n_geoms, *shape), fill_value=0)

    if arrow_tbl.num_rows == 0:
        return empty

    geoms = shapely.from_wkb(arrow_tbl.column("wkb").to_pylist())
    global_ids = np.array(arrow_tbl.column("id").to_pylist(), dtype=np.intp)
    local_gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=source.crs)

    local_coo = np_coverage(
        chunk_affine,
        shape=shape,
        geometries=local_gdf,
        strategy=strategy,
        coverage_weight=coverage_weight,
    )

    if local_coo.nnz == 0:
        return empty

    # Remap local geom indices (0..k-1) → global file_row_number positions via
    # NumPy fancy indexing: global_ids[local_idx] gives the parquet row number.
    global_geom_coords = global_ids[local_coo.coords[0]]
    new_coords = np.stack([global_geom_coords, local_coo.coords[1], local_coo.coords[2]])
    return sparse.COO(new_coords, local_coo.data, shape=(n_geoms, *shape), fill_value=0)


def _rasterio_chunk(
    arrow_tbl: pyarrow.Table,
    chunk_affine: Affine,
    shape: tuple[int, int],
    merge_alg: str,
    all_touched: bool,
) -> np.ndarray:
    import rasterio.features
    import shapely
    from rasterio.features import MergeAlg

    # shapely.from_wkb accepts a list of bytes; to_pylist() on a pyarrow Binary
    # column is unavoidable here since shapely must parse each WKB individually.
    geoms = shapely.from_wkb(arrow_tbl.column("wkb").to_pylist())
    ids = arrow_tbl.column("id").to_pylist()  # Python ints for rasterio's zip

    alg = MergeAlg.replace if merge_alg == "replace" else MergeAlg.add
    return rasterio.features.rasterize(
        zip(geoms, ids),
        out_shape=shape,
        transform=chunk_affine,
        fill=_FILL,
        dtype=np.int32,
        all_touched=all_touched,
        merge_alg=alg,
    )
