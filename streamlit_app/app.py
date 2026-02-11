"""
DuckDB Snowflake Explorer
==========================
Query any Snowflake database through DuckDB's Snowflake extension.
All compute runs in DuckDB (local) via Arrow ADBC – Snowflake is used only
as a data source.  Supports Password, Key Pair, and OAuth authentication.

Key DuckDB extension patterns used:
  CREATE SECRET … (TYPE snowflake)
  snowflake_query(sql, secret_name)   – passthrough query
  ATTACH '' AS db (TYPE snowflake, SECRET …, READ_ONLY)
"""

import streamlit as st
import duckdb
import pandas as pd
import time
import io
import os
import traceback

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DuckDB ❄️ Snowflake Explorer",
    page_icon="🦆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    .block-container { padding-top: 1.2rem; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #404040;
        padding: 12px 16px;
        border-radius: 8px;
    }
    .table-header {
        background: linear-gradient(90deg, #29B5E8 0%, #1a7a9e 100%);
        padding: 8px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
        color: white;
        font-weight: 600;
    }
    .engine-badge {
        background: linear-gradient(90deg, #FFC107 0%, #FF9800 100%);
        color: #333;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .status-connected {
        background-color: #28a745; color: white;
        padding: 2px 10px; border-radius: 12px; font-size: 0.85rem;
    }
    .status-disconnected {
        background-color: #dc3545; color: white;
        padding: 2px 10px; border-radius: 12px; font-size: 0.85rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state defaults ───────────────────────────────────────────────────

_DEFAULTS: dict = {
    "connected": False,
    "secret_name": "",
    "attach_alias": "",
    "conn_string_preview": "",
    "query_history": [],
    "current_db": None,
    "current_schema": None,
    "current_table": None,
    "last_result": None,
    "last_query": "",
    "last_elapsed": 0.0,
    "last_row_count": 0,
    "duckdb_conn": None,
    "extension_loaded": False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── DuckDB connection management ────────────────────────────────────────────

def _get_duckdb() -> duckdb.DuckDBPyConnection:
    """Return (and cache) a persistent in-process DuckDB connection."""
    if st.session_state.duckdb_conn is None:
        conn = duckdb.connect(database=":memory:")
        st.session_state.duckdb_conn = conn
    return st.session_state.duckdb_conn


def _ensure_extension(conn: duckdb.DuckDBPyConnection) -> tuple[bool, str]:
    """Install + load the snowflake extension.  Returns (ok, detail)."""
    if st.session_state.extension_loaded:
        return True, "already loaded"
    try:
        conn.execute("INSTALL snowflake FROM community;")
    except Exception:
        pass  # may already be installed
    try:
        conn.execute("LOAD snowflake;")
    except Exception as exc:
        return False, str(exc)
    try:
        ver = conn.execute("SELECT snowflake_version();").fetchone()[0]
    except Exception:
        ver = "unknown"
    st.session_state.extension_loaded = True
    return True, ver


def _create_secret(
    conn: duckdb.DuckDBPyConnection,
    secret_name: str,
    account: str,
    user: str,
    auth_method: str,
    password: str = "",
    warehouse: str = "",
    database: str = "",
    role: str = "",
    oauth_token: str = "",
    private_key_pem: str = "",
    private_key_passphrase: str = "",
) -> tuple[bool, str]:
    """DROP + CREATE SECRET in DuckDB.  Returns (ok, msg)."""
    try:
        conn.execute(f"DROP SECRET IF EXISTS {secret_name};")
    except Exception:
        pass

    parts = [
        f"TYPE snowflake",
        f"ACCOUNT '{account}'",
        f"USER '{user}'",
    ]

    if auth_method == "Password":
        parts.append(f"PASSWORD '{password}'")
    elif auth_method == "Key Pair":
        parts.append("AUTH_TYPE 'key_pair'")
        # Escape single quotes in PEM
        parts.append(f"PRIVATE_KEY '{private_key_pem}'")
        if private_key_passphrase:
            parts.append(f"PRIVATE_KEY_PASSPHRASE '{private_key_passphrase}'")
    elif auth_method == "OAuth Token":
        parts.append("AUTH_TYPE 'oauth'")
        parts.append(f"TOKEN '{oauth_token}'")

    if database:
        parts.append(f"DATABASE '{database}'")
    if warehouse:
        parts.append(f"WAREHOUSE '{warehouse}'")
    if role:
        parts.append(f"ROLE '{role}'")

    sql = f"CREATE SECRET {secret_name} ({', '.join(parts)});"
    try:
        conn.execute(sql)
        return True, "Secret created"
    except Exception as exc:
        return False, str(exc)


def _test_connection_via_duckdb(conn: duckdb.DuckDBPyConnection, secret_name: str) -> tuple[bool, str]:
    """Run a trivial query through snowflake_query() to verify connectivity."""
    try:
        df = conn.execute(
            f"SELECT * FROM snowflake_query('SELECT CURRENT_VERSION() AS V, CURRENT_ACCOUNT() AS A, "
            f"CURRENT_USER() AS U, CURRENT_ROLE() AS R', '{secret_name}');"
        ).fetchdf()
        row = df.iloc[0]
        info = f"Snowflake v{row['V']}  •  Account: {row['A']}  •  User: {row['U']}  •  Role: {row['R']}"
        return True, info
    except Exception as exc:
        return False, str(exc)


def _run_query_duckdb(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[pd.DataFrame | None, float, str]:
    """Execute *sql* in DuckDB and return (df, elapsed, error)."""
    try:
        t0 = time.perf_counter()
        result = conn.execute(sql)
        df = result.fetchdf()
        elapsed = time.perf_counter() - t0
        return df, elapsed, ""
    except Exception as exc:
        return None, 0.0, str(exc)


def _snowflake_query(conn: duckdb.DuckDBPyConnection, sf_sql: str, secret: str) -> tuple[pd.DataFrame | None, float, str]:
    """Execute a passthrough Snowflake SQL via snowflake_query()."""
    # Escape single quotes in the user SQL for embedding
    escaped = sf_sql.replace("'", "''")
    wrapper = f"SELECT * FROM snowflake_query('{escaped}', '{secret}');"
    return _run_query_duckdb(conn, wrapper)


def _sf_fetch_list(conn: duckdb.DuckDBPyConnection, sf_sql: str, secret: str, col: str | int = 0) -> list[str]:
    """Fetch a list from one column via snowflake_query().

    *col* can be a column name (str) or positional index (int).
    For SHOW commands the interesting column is usually called 'name'.
    """
    df, _, err = _snowflake_query(conn, sf_sql, secret)
    if df is not None and not df.empty:
        if isinstance(col, str):
            # case-insensitive column lookup
            col_lower = col.lower()
            for c in df.columns:
                if c.lower() == col_lower:
                    return df[c].dropna().astype(str).tolist()
            # fallback: first column
            return df.iloc[:, 0].astype(str).tolist()
        return df.iloc[:, col].astype(str).tolist()
    return []


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🦆 DuckDB + ❄️ Snowflake")
    st.caption("All compute runs locally in DuckDB via Arrow ADBC")

    # Connection status
    if st.session_state.connected:
        st.markdown('<span class="status-connected">● Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-disconnected">● Disconnected</span>', unsafe_allow_html=True)

    st.divider()

    auth_method = st.selectbox(
        "Authentication Method",
        ["Password", "Key Pair", "OAuth Token"],
        help=(
            "**Password** – standard user / password\n\n"
            "**Key Pair** – RSA private key (paste PEM text)\n\n"
            "**OAuth Token** – provide an access token"
        ),
    )

    account = st.text_input("Account Identifier", placeholder="xy12345.us-east-1",
                            help="Snowflake account locator, e.g. xy12345 or xy12345.us-east-1")
    user = st.text_input("Username", placeholder="your_username")

    # Auth-specific fields
    password = ""
    oauth_token = ""
    private_key_pem = ""
    private_key_passphrase = ""

    if auth_method == "Password":
        password = st.text_input("Password", type="password")
    elif auth_method == "OAuth Token":
        oauth_token = st.text_area("OAuth Access Token", height=68, placeholder="eyJhbGci…")
    elif auth_method == "Key Pair":
        key_file = st.file_uploader("Private Key File (.p8 / .pem)", type=["p8", "pem", "key"])
        if key_file is not None:
            private_key_pem = key_file.read().decode("utf-8", errors="replace")
        private_key_passphrase = st.text_input("Key Passphrase (optional)", type="password")

    st.divider()
    st.markdown("##### Optional Defaults")
    warehouse = st.text_input("Warehouse", placeholder="COMPUTE_WH")
    database = st.text_input("Database", placeholder="MY_DATABASE")
    role = st.text_input("Role", placeholder="PUBLIC")

    st.divider()

    col_c, col_d = st.columns(2)
    with col_c:
        connect_clicked = st.button("🔌 Connect", use_container_width=True, type="primary")
    with col_d:
        disconnect_clicked = st.button("Disconnect", use_container_width=True)

    if connect_clicked:
        if not account or not user:
            st.error("Account and Username are required.")
        else:
            conn = _get_duckdb()
            with st.spinner("Loading Snowflake extension…"):
                ext_ok, ext_detail = _ensure_extension(conn)
            if not ext_ok:
                st.error(f"Failed to load extension: {ext_detail}")
            else:
                secret_name = "sf_explorer_secret"
                with st.spinner("Creating DuckDB secret…"):
                    sec_ok, sec_msg = _create_secret(
                        conn, secret_name,
                        account=account, user=user, auth_method=auth_method,
                        password=password, warehouse=warehouse, database=database,
                        role=role, oauth_token=oauth_token,
                        private_key_pem=private_key_pem,
                        private_key_passphrase=private_key_passphrase,
                    )
                if not sec_ok:
                    st.error(f"Secret creation failed: {sec_msg}")
                else:
                    with st.spinner("Testing Snowflake connection via DuckDB…"):
                        ok, info = _test_connection_via_duckdb(conn, secret_name)
                    if ok:
                        st.session_state.connected = True
                        st.session_state.secret_name = secret_name
                        # Build preview string
                        preview_parts = [f"account={account}", f"user={user}"]
                        if auth_method == "Password":
                            preview_parts.append("password=****")
                        elif auth_method == "Key Pair":
                            preview_parts.append("auth_type=key_pair;private_key=****")
                        elif auth_method == "OAuth Token":
                            preview_parts.append("auth_type=oauth;token=****")
                        if warehouse:
                            preview_parts.append(f"warehouse={warehouse}")
                        if database:
                            preview_parts.append(f"database={database}")
                        if role:
                            preview_parts.append(f"role={role}")
                        st.session_state.conn_string_preview = ";".join(preview_parts)
                        st.success(f"✅ {info}")
                        st.caption(f"Extension: {ext_detail}")
                    else:
                        st.error(f"Connection test failed:\n\n{info}")

    if disconnect_clicked:
        # Clean up DuckDB state
        if st.session_state.duckdb_conn is not None:
            try:
                st.session_state.duckdb_conn.execute(f"DROP SECRET IF EXISTS {st.session_state.secret_name};")
            except Exception:
                pass
            if st.session_state.attach_alias:
                try:
                    st.session_state.duckdb_conn.execute(f"DETACH {st.session_state.attach_alias};")
                except Exception:
                    pass
        st.session_state.connected = False
        st.session_state.secret_name = ""
        st.session_state.attach_alias = ""
        st.session_state.current_db = None
        st.session_state.current_schema = None
        st.session_state.current_table = None
        st.session_state.extension_loaded = False
        st.session_state.duckdb_conn = None
        st.info("Disconnected. DuckDB state reset.")

    # Connection-string preview
    if st.session_state.conn_string_preview:
        with st.expander("🔗 DuckDB Secret / Connection Preview"):
            st.code(st.session_state.conn_string_preview, language="text")
            st.caption("CREATE SECRET … (TYPE snowflake) – stored in DuckDB memory only")


# ── Main area ────────────────────────────────────────────────────────────────

st.markdown("# 🦆 DuckDB Snowflake Explorer")
st.markdown(
    '<span class="engine-badge">Compute: DuckDB (local) via Arrow ADBC</span>',
    unsafe_allow_html=True,
)

if not st.session_state.connected:
    st.info("👈 Configure your Snowflake connection in the sidebar and click **Connect**.")
    with st.expander("ℹ️ How it works"):
        st.markdown(
            """
This app uses the **DuckDB Snowflake extension** to query Snowflake data.

- All query execution happens **locally in DuckDB** — Snowflake is only the data source.
- Data is transferred via **Apache Arrow ADBC** for efficient columnar transport.
- Two query modes are available:
  1. **`snowflake_query(sql, secret)`** — passthrough: your SQL runs on Snowflake, results stream to DuckDB via Arrow.
  2. **`ATTACH … (TYPE snowflake)`** — DuckDB SQL syntax over Snowflake tables with optional filter pushdown.
- You can also run **pure DuckDB SQL** (local tables, analytics, COPY TO, etc.) alongside Snowflake queries.
            """
        )
    st.stop()

conn = _get_duckdb()
secret = st.session_state.secret_name

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_browser, tab_query, tab_local, tab_history = st.tabs(
    ["📂 Database Browser", "📝 Snowflake Query", "🦆 DuckDB Local SQL", "📜 History"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Database / Schema / Table browser  (via snowflake_query)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_browser:
    st.markdown("#### Browse Snowflake Databases  *(fetched via DuckDB → Arrow ADBC)*")

    # ── Databases ────────────────────────────────────────────────────────────
    databases = _sf_fetch_list(conn, "SHOW DATABASES", secret, col="name")
    if not databases:
        st.warning("No databases found (check privileges or connection).")
        st.stop()

    col_db, col_schema, col_table = st.columns(3)

    with col_db:
        selected_db = st.selectbox("Database", databases, key="sel_db")

    # ── Schemas ──────────────────────────────────────────────────────────────
    schemas: list[str] = []
    if selected_db:
        schemas = _sf_fetch_list(
            conn,
            f"SELECT SCHEMA_NAME FROM \"{selected_db}\".INFORMATION_SCHEMA.SCHEMATA ORDER BY 1",
            secret,
        )

    with col_schema:
        selected_schema = st.selectbox("Schema", schemas, key="sel_schema") if schemas else None

    # ── Tables ───────────────────────────────────────────────────────────────
    tables: list[str] = []
    if selected_db and selected_schema:
        tables = _sf_fetch_list(
            conn,
            f"SELECT TABLE_NAME FROM \"{selected_db}\".INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = '{selected_schema}' ORDER BY 1",
            secret,
        )

    with col_table:
        selected_table = st.selectbox("Table", tables, key="sel_table") if tables else None

    st.divider()

    if selected_db and selected_schema and selected_table:
        st.session_state.current_db = selected_db
        st.session_state.current_schema = selected_schema
        st.session_state.current_table = selected_table

        meta_col, preview_col = st.columns([2, 3])

        with meta_col:
            st.markdown(
                f'<div class="table-header">📋 {selected_db}.{selected_schema}.{selected_table}</div>',
                unsafe_allow_html=True,
            )
            col_sql = (
                f"SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH, "
                f"NUMERIC_PRECISION, NUMERIC_SCALE, COLUMN_DEFAULT "
                f"FROM \"{selected_db}\".INFORMATION_SCHEMA.COLUMNS "
                f"WHERE TABLE_SCHEMA = '{selected_schema}' AND TABLE_NAME = '{selected_table}' "
                f"ORDER BY ORDINAL_POSITION"
            )
            col_df, _, col_err = _snowflake_query(conn, col_sql, secret)
            if col_df is not None and not col_df.empty:
                st.dataframe(col_df, use_container_width=True, hide_index=True)
            else:
                st.caption(f"Could not retrieve metadata. {col_err}")

            cnt_sql = f"SELECT COUNT(*) AS ROW_COUNT FROM \"{selected_db}\".\"{selected_schema}\".\"{selected_table}\""
            cnt_df, _, _ = _snowflake_query(conn, cnt_sql, secret)
            if cnt_df is not None and not cnt_df.empty:
                st.metric("Row Count", f"{int(cnt_df.iloc[0, 0]):,}")

        with preview_col:
            st.markdown(
                '<div class="table-header">🔍 Data Preview  (via DuckDB snowflake_query)</div>',
                unsafe_allow_html=True,
            )
            preview_limit = st.slider("Preview rows", 10, 1000, 100, step=10, key="preview_limit")
            preview_sf_sql = (
                f"SELECT * FROM \"{selected_db}\".\"{selected_schema}\".\"{selected_table}\" "
                f"LIMIT {preview_limit}"
            )

            if st.button("🔄 Load Preview", key="load_preview"):
                with st.spinner("Fetching via DuckDB → Arrow ADBC…"):
                    df, elapsed, err = _snowflake_query(conn, preview_sf_sql, secret)
                if err:
                    st.error(err)
                elif df is not None:
                    st.caption(f"{len(df)} rows  •  {elapsed:.2f}s  •  DuckDB compute")
                    st.dataframe(df, use_container_width=True, hide_index=True)

            with st.expander("📋 DuckDB SQL for this table"):
                passthrough = (
                    f"SELECT * FROM snowflake_query(\n"
                    f"    'SELECT * FROM \"{selected_db}\".\"{selected_schema}\".\"{selected_table}\" LIMIT {preview_limit}',\n"
                    f"    '{secret}'\n);"
                )
                st.code(passthrough, language="sql")
    elif selected_db:
        st.caption("Select a schema and table to preview data and column metadata.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Snowflake passthrough query  (snowflake_query)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_query:
    st.markdown("### Snowflake SQL  *(passthrough via `snowflake_query()`)*")
    st.caption(
        "Your SQL is sent to Snowflake; results stream back through Arrow ADBC "
        "and are processed locally in DuckDB."
    )

    # Quick helpers
    hc = st.columns(4)
    table_ref = (
        f'"{st.session_state.current_db}"."{st.session_state.current_schema}"."{st.session_state.current_table}"'
        if st.session_state.current_table
        else '"DB"."SCHEMA"."TABLE"'
    )
    helpers = {
        "SHOW WAREHOUSES": "SHOW WAREHOUSES",
        "SHOW DATABASES": "SHOW DATABASES",
        "Session Info": "SELECT CURRENT_USER() AS U, CURRENT_ROLE() AS R, "
                        "CURRENT_WAREHOUSE() AS WH, CURRENT_DATABASE() AS DB, "
                        "CURRENT_SCHEMA() AS SCHEMA",
        "Table Sample": f"SELECT * FROM {table_ref} LIMIT 100",
    }
    for idx, (lbl, snip) in enumerate(helpers.items()):
        with hc[idx]:
            if st.button(lbl, key=f"sf_helper_{idx}", use_container_width=True):
                st.session_state["sf_sql_input"] = snip

    sf_sql = st.text_area(
        "Snowflake SQL",
        value=st.session_state.get("sf_sql_input", ""),
        height=160,
        placeholder="SELECT * FROM MY_DB.MY_SCHEMA.MY_TABLE LIMIT 100",
        key="sf_sql_editor",
    )

    rc = st.columns([1, 1, 4])
    with rc[0]:
        sf_run = st.button("▶️ Run", type="primary", use_container_width=True, key="sf_run_btn")
    with rc[1]:
        sf_explain = st.button("📊 Explain", use_container_width=True, key="sf_explain_btn")

    sf_query = None
    if sf_run and sf_sql.strip():
        sf_query = sf_sql.strip()
    elif sf_explain and sf_sql.strip():
        sf_query = f"EXPLAIN {sf_sql.strip().rstrip(';')}"

    if sf_query:
        with st.spinner("Executing via DuckDB → snowflake_query()…"):
            df, elapsed, err = _snowflake_query(conn, sf_query, secret)
        if err:
            st.error(f"**Query Error**\n\n```\n{err}\n```")
        elif df is not None:
            st.session_state.last_result = df
            st.session_state.last_query = sf_query
            st.session_state.last_elapsed = elapsed
            st.session_state.last_row_count = len(df)
            st.session_state.query_history.insert(0, {
                "sql": sf_query, "mode": "snowflake_query",
                "rows": len(df), "time": f"{elapsed:.2f}s",
                "ts": time.strftime("%H:%M:%S"),
            })
            st.session_state.query_history = st.session_state.query_history[:50]

    # Results
    if st.session_state.last_result is not None and not st.session_state.last_result.empty:
        df = st.session_state.last_result
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{st.session_state.last_row_count:,}")
        m2.metric("Columns", str(len(df.columns)))
        m3.metric("Time", f"{st.session_state.last_elapsed:.2f}s")
        st.dataframe(df, use_container_width=True, hide_index=True, height=420)

        # Export
        ec = st.columns(3)
        with ec[0]:
            st.download_button("⬇️ CSV", df.to_csv(index=False), "result.csv", "text/csv", use_container_width=True)
        with ec[1]:
            st.download_button("⬇️ JSON", df.to_json(orient="records", indent=2), "result.json", "application/json", use_container_width=True)
        with ec[2]:
            buf = io.BytesIO()
            try:
                df.to_parquet(buf, index=False)
                st.download_button("⬇️ Parquet", buf.getvalue(), "result.parquet", "application/octet-stream", use_container_width=True)
            except Exception:
                st.button("Parquet N/A", disabled=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – DuckDB local SQL (cross-database analytics, COPY TO, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_local:
    st.markdown("### DuckDB Local SQL")
    st.caption(
        "Run any DuckDB SQL here – analytics on local tables, COPY TO exports, "
        "cross-database joins between Snowflake and local data, etc."
    )

    lc = st.columns(5)
    local_helpers = {
        "DuckDB Version": "SELECT version() AS duckdb_version",
        "List Secrets": "SELECT name, type, provider FROM duckdb_secrets() WHERE type = 'snowflake'",
        "Local Tables": "SHOW ALL TABLES",
        "Snowflake Ext": "SELECT snowflake_version()",
        "SF → Local": (
            f"CREATE OR REPLACE TABLE sample_data AS\n"
            f"SELECT * FROM snowflake_query(\n"
            f"    'SELECT * FROM {table_ref} LIMIT 500',\n"
            f"    '{secret}'\n)"
            if st.session_state.current_table
            else "-- Select a table in Browser tab first"
        ),
    }
    for idx, (lbl, snip) in enumerate(local_helpers.items()):
        with lc[idx]:
            if st.button(lbl, key=f"local_helper_{idx}", use_container_width=True):
                st.session_state["local_sql_input"] = snip

    # Show example of cross-database query
    with st.expander("💡 Example: cross-database join & export"):
        st.code(
            f"""-- Bring Snowflake data into a local DuckDB table
CREATE TABLE local_customers AS
SELECT * FROM snowflake_query(
    'SELECT * FROM "MY_DB"."PUBLIC"."CUSTOMERS" LIMIT 1000',
    '{secret}'
);

-- Analyze locally with DuckDB's analytical engine
SELECT state, COUNT(*) AS cnt, AVG(balance) AS avg_bal
FROM local_customers
GROUP BY state
ORDER BY cnt DESC;

-- Export to Parquet
COPY local_customers TO '/tmp/customers.parquet' (FORMAT PARQUET);""",
            language="sql",
        )

    local_sql = st.text_area(
        "DuckDB SQL",
        value=st.session_state.get("local_sql_input", ""),
        height=160,
        placeholder="SELECT version();",
        key="local_sql_editor",
    )

    lrc = st.columns([1, 5])
    with lrc[0]:
        local_run = st.button("▶️ Run", type="primary", use_container_width=True, key="local_run_btn")

    if local_run and local_sql.strip():
        # Support multiple statements separated by ;  (naive split, good enough for simple cases)
        statements = [s.strip() for s in local_sql.strip().rstrip(";").split(";") if s.strip()]
        for stmt in statements:
            with st.spinner(f"Executing: {stmt[:60]}…"):
                df, elapsed, err = _run_query_duckdb(conn, stmt)
            if err:
                st.error(f"```\n{err}\n```")
            elif df is not None:
                st.session_state.query_history.insert(0, {
                    "sql": stmt, "mode": "duckdb_local",
                    "rows": len(df), "time": f"{elapsed:.2f}s",
                    "ts": time.strftime("%H:%M:%S"),
                })
                st.session_state.query_history = st.session_state.query_history[:50]

                if not df.empty:
                    st.caption(f"{len(df)} rows  •  {elapsed:.2f}s  •  DuckDB local")
                    st.dataframe(df, use_container_width=True, hide_index=True, height=400)
                    # Export
                    ec2 = st.columns(3)
                    with ec2[0]:
                        st.download_button("⬇️ CSV", df.to_csv(index=False), "local_result.csv", "text/csv",
                                           use_container_width=True, key=f"lcsv_{stmt[:20]}")
                    with ec2[1]:
                        st.download_button("⬇️ JSON", df.to_json(orient="records", indent=2), "local_result.json",
                                           "application/json", use_container_width=True, key=f"ljson_{stmt[:20]}")
                    with ec2[2]:
                        buf2 = io.BytesIO()
                        try:
                            df.to_parquet(buf2, index=False)
                            st.download_button("⬇️ Parquet", buf2.getvalue(), "local_result.parquet",
                                               "application/octet-stream", use_container_width=True,
                                               key=f"lpq_{stmt[:20]}")
                        except Exception:
                            st.button("Parquet N/A", disabled=True, use_container_width=True, key=f"lpqna_{stmt[:20]}")
                else:
                    st.success(f"✅ Statement executed ({elapsed:.2f}s, {len(df)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Query History
# ═══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.markdown("### Query History")
    if not st.session_state.query_history:
        st.caption("No queries executed yet.")
    else:
        for i, entry in enumerate(st.session_state.query_history):
            mode_icon = "❄️" if entry.get("mode") == "snowflake_query" else "🦆"
            with st.expander(
                f"{mode_icon} **{entry['ts']}** — {entry['rows']} rows in {entry['time']}",
                expanded=(i == 0),
            ):
                st.code(entry["sql"], language="sql")
                if st.button("♻️ Re-run", key=f"rerun_{i}"):
                    target = "sf_sql_input" if entry.get("mode") == "snowflake_query" else "local_sql_input"
                    st.session_state[target] = entry["sql"]
                    st.rerun()

        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()
