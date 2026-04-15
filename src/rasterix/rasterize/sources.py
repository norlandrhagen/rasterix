from __future__ import annotations

import threading
from dataclasses import dataclass, field

__all__ = ["GeoParquetSource"]

# Per-thread DuckDB connection cache.  Keyed by sorted duckdb_config items
# (a single connection can query any parquet path via read_parquet).  Reusing
# connections avoids paying connect() + load_extension("spatial") overhead
# (~100 ms) on every chunk across the 800+ map_blocks calls in a typical job.
_local = threading.local()


def _get_connection(duckdb_config: dict) -> duckdb.DuckDBPyConnection:
    """Return a cached DuckDB connection, creating one per thread if needed.

    The connection is stored in thread-local storage so that each dask worker
    thread has its own independent connection (DuckDB connections are not
    thread-safe to share).  ``install_extension`` / ``load_extension`` are
    called only the first time a connection is created on a given thread.
    """
    import duckdb

    key = tuple(sorted(duckdb_config.items()))
    cache: dict = vars(_local).setdefault("connections", {})
    if key not in cache:
        con = duckdb.connect()
        con.install_extension("spatial")
        con.load_extension("spatial")
        con.execute("SET enable_progress_bar = false")
        for k, v in duckdb_config.items():
            con.execute(f"SET {k} = '{v}'")
        cache[key] = con
    return cache[key]


@dataclass
class GeoParquetSource:
    """A lazy GeoParquet geometry source backed by DuckDB spatial queries.

    Each call to :meth:`query_bbox` performs a spatial intersection filter
    against the parquet file and returns only the geometries relevant to the
    requested bounding box as an Arrow table.  DuckDB connections are cached
    per-thread (via :data:`threading.local`) so ``connect()`` and
    ``load_extension("spatial")`` are paid only once per worker thread rather
    than once per chunk.  The dataclass is serialisable and picklable, making
    it safe to pass across dask workers.

    Parameters
    ----------
    path : str
        Path to the GeoParquet file.  May be a local path or an S3 URI
        (``s3://...``).
    geometry_column : str
        Name of the geometry column in the parquet file.
    id_column : str or None
        Column to use as the integer feature ID burned into the output raster.
        If ``None``, ``row_number() OVER ()`` is computed over all rows in
        physical file order before the spatial filter is applied, producing
        stable 1-based IDs that are consistent across all chunk queries on the
        same file.
    bbox_column : str or None
        Name of a GeoParquet 1.1 covering-bbox struct column (typically
        ``"bbox"``), whose fields are ``xmin``, ``ymin``, ``xmax``, ``ymax``.
        When set, a fast numeric pre-filter on this column is prepended before
        the ``ST_Intersects`` check.  Because the bbox fields are plain numeric
        leaf columns, Parquet row-group statistics allow DuckDB to skip entire
        row groups without reading any geometry — a major speedup for large
        files on S3.  Write the column with
        ``gdf.to_parquet(..., write_covering_bbox=True)`` (geopandas ≥ 1.0).
    duckdb_config : dict
        Extra DuckDB ``SET`` key/value pairs applied before the query, e.g.
        S3 credentials::

            {"s3_access_key_id": "...", "s3_secret_access_key": "..."}
    """

    path: str
    geometry_column: str = "geometry"
    id_column: str | None = None
    bbox_column: str | None = None
    duckdb_config: dict = field(default_factory=dict)

    def query_bbox(self, xmin: float, ymin: float, xmax: float, ymax: float) -> pyarrow.Table:
        """Return geometries intersecting the given envelope as an Arrow table.

        Parameters
        ----------
        xmin, ymin, xmax, ymax : float
            Spatial bounding box in the same CRS as the geometries.

        Returns
        -------
        pyarrow.Table
            Two columns: ``"wkb"`` (Binary — WKB-encoded geometry) and
            ``"id"`` (Int64 — integer ID to burn into the raster).  Zero rows
            if nothing intersects the bbox.
        """
        con = _get_connection(self.duckdb_config)

        g = self.geometry_column

        # Build WHERE clause in two stages:
        #   1. bbox pre-filter (only when bbox_column is set) — Parquet row-group
        #      stats skip entire groups before touching the geometry column.
        #   2. ST_Intersects — precise filter on the surviving rows.
        #
        # Geometry with bbox intersects query bbox when:
        #   g.xmin <= q.xmax  AND  g.xmax >= q.xmin
        #   g.ymin <= q.ymax  AND  g.ymax >= q.ymin
        if self.bbox_column is not None:
            b = self.bbox_column
            bbox_filter = f"{b}.xmin <= ? AND {b}.xmax >= ? AND {b}.ymin <= ? AND {b}.ymax >= ?"
            where = f"{bbox_filter} AND ST_Intersects({g}, ST_MakeEnvelope(?, ?, ?, ?))"
            params: list = [xmax, xmin, ymax, ymin, xmin, ymin, xmax, ymax]
        else:
            where = f"ST_Intersects({g}, ST_MakeEnvelope(?, ?, ?, ?))"
            params = [xmin, ymin, xmax, ymax]

        if self.id_column is None:
            # row_number() OVER () follows physical file order and is stable for
            # a given parquet file.  The CTE assigns numbers over ALL rows first
            # so the same row always gets the same ID regardless of which chunk's
            # bbox is being queried.
            #
            # Include bbox in the CTE select when needed so the outer WHERE can
            # reference it (the struct column isn't implicitly forwarded).
            cte_extra_col = f", {self.bbox_column}" if self.bbox_column is not None else ""
            sql = f"""
                WITH numbered AS (
                    SELECT row_number() OVER () AS _auto_id,
                           ST_AsWKB({g}) AS wkb,
                           {g}{cte_extra_col}
                    FROM read_parquet('{self.path}')
                )
                SELECT wkb, CAST(_auto_id AS BIGINT) AS id
                FROM numbered
                WHERE {where}
            """
        else:
            id_col = self.id_column
            sql = f"""
                SELECT ST_AsWKB({g}) AS wkb, CAST({id_col} AS BIGINT) AS id
                FROM read_parquet('{self.path}')
                WHERE {where}
            """

        return con.execute(sql, params).fetch_arrow_table()
