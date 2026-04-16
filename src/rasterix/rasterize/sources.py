from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Literal

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


def _ensure_rtree_table(
    con: duckdb.DuckDBPyConnection,
    path: str,
    geometry_column: str,
) -> str:
    """Materialize *path* into an in-memory DuckDB table with an R-Tree index.

    This is called at most once per (connection, path) pair.  The first call
    pays a one-time full-scan cost to:

    1. Create a table ``_rasterix_<hash>`` holding ``_auto_id`` (stable
       ``row_number()``-based integer) and the geometry column.
    2. Create an R-Tree index on the geometry column.

    All subsequent :meth:`GeoParquetSource.query_bbox` calls on the same
    thread use ``ST_Intersects`` against the indexed table, which DuckDB
    resolves via an O(log n + k) R-Tree lookup rather than a full scan.

    Returns
    -------
    str
        The name of the materialized table.
    """
    table_name = "_rasterix_" + hashlib.md5(path.encode()).hexdigest()[:16]
    index_name = table_name + "_rtree"

    # Check whether the table already exists on this connection.
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()[0]

    if not exists:
        g = geometry_column
        con.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT row_number() OVER () AS _auto_id, {g} AS geometry
            FROM read_parquet('{path}')
        """)
        con.execute(f"CREATE INDEX {index_name} ON {table_name} USING RTREE (geometry)")

    return table_name


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
        If ``None``, stable 1-based IDs are derived from physical file order
        via ``row_number() OVER ()``.  The exact execution strategy (in-memory
        R-Tree vs. CTE on ``read_parquet``) is controlled by ``use_rtree``.
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
    use_rtree : bool or None
        Whether to materialise the parquet file into an in-memory DuckDB table
        with an R-Tree spatial index on the first chunk query, then reuse that
        table for all subsequent queries on the same worker thread.

        * ``True``  — always materialise + index (best per-chunk speed; only
          sensible when the whole file fits comfortably in worker memory).
        * ``False`` — always query ``read_parquet()`` directly, using
          ``bbox_column`` row-group pruning when available.
        * ``None`` (default) — choose automatically based on available hints:

          - ``id_column`` set → ``False`` (stable IDs provided, no scan needed)
          - ``bbox_column`` set (but no ``id_column``) → ``False`` (row-group
            pruning is already efficient; file may be large)
          - neither set → check row count via parquet footer metadata;
            use R-Tree if ``n_rows < 5_000_000``, else fall back to the CTE
            path.
    """

    path: str
    geometry_column: str = "geometry"
    id_column: str | None = None
    bbox_column: str | None = None
    duckdb_config: dict = field(default_factory=dict)
    use_rtree: bool | None = None

    def _resolve_strategy(self, con: duckdb.DuckDBPyConnection) -> Literal["rtree", "parquet"]:
        """Return the query strategy to use, applying heuristics when needed.

        Called once per worker thread (result is not cached here; the R-Tree
        table creation in :func:`_ensure_rtree_table` is idempotent).
        """
        if self.use_rtree is True:
            return "rtree"
        if self.use_rtree is False:
            return "parquet"

        # Auto-detect from available hints.
        if self.id_column is not None:
            # Stable ID provided — no row_number() scan needed at all.
            return "parquet"
        if self.bbox_column is not None:
            # bbox column implies the file was written for performance and may
            # be large; row-group pruning is already efficient.
            return "parquet"

        # No hints: check total row count from the parquet footer (cheap —
        # reads only file metadata, not data pages).  Cache the result in
        # thread-local storage so we only pay this cost once per worker thread.
        strategy_cache: dict = vars(_local).setdefault("strategies", {})
        if self.path not in strategy_cache:
            n_rows = con.execute(f"SELECT sum(num_rows) FROM parquet_metadata('{self.path}')").fetchone()[0]
            strategy_cache[self.path] = "rtree" if n_rows < 5_000_000 else "parquet"
        return strategy_cache[self.path]

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
        strategy = self._resolve_strategy(con)
        g = self.geometry_column

        if strategy == "rtree":
            # Materialise parquet once per thread into an in-memory table with
            # stable row_number()-based IDs and an R-Tree index.  Subsequent
            # queries on the same thread are O(log n + k) lookups.
            tbl = _ensure_rtree_table(con, self.path, g)
            sql = f"""
                SELECT ST_AsWKB(geometry) AS wkb, CAST(_auto_id AS BIGINT) AS id
                FROM {tbl}
                WHERE ST_Intersects(geometry, ST_MakeEnvelope(?, ?, ?, ?))
            """
            return con.execute(sql, [xmin, ymin, xmax, ymax]).fetch_arrow_table()

        # --- parquet path ---------------------------------------------------
        # Build WHERE in two stages:
        #   1. bbox struct pre-filter (row-group stats skip entire groups).
        #   2. ST_Intersects (precise per-geometry filter).
        if self.bbox_column is not None:
            b = self.bbox_column
            bbox_filter = f"{b}.xmin <= ? AND {b}.xmax >= ? AND {b}.ymin <= ? AND {b}.ymax >= ?"
            where = f"{bbox_filter} AND ST_Intersects({g}, ST_MakeEnvelope(?, ?, ?, ?))"
            params: list = [xmax, xmin, ymax, ymin, xmin, ymin, xmax, ymax]
        else:
            where = f"ST_Intersects({g}, ST_MakeEnvelope(?, ?, ?, ?))"
            params = [xmin, ymin, xmax, ymax]

        if self.id_column is None:
            # No natural ID: assign stable row numbers via CTE before filtering.
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
            sql = f"""
                SELECT ST_AsWKB({g}) AS wkb, CAST({self.id_column} AS BIGINT) AS id
                FROM read_parquet('{self.path}')
                WHERE {where}
            """

        return con.execute(sql, params).fetch_arrow_table()
