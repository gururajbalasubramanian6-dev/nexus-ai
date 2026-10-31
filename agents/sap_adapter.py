"""
Nexus-AI: SAP HANA Cloud Database Adapter
==========================================

This module provides secure, read-only integration between SAP HANA Cloud and
the Nexus-AI multi-agent platform.

Features:
1. Secure environment-variable-based configuration (with Streamlit secrets fallback).
2. Host and port parsing (supports host:port endpoints and explicit port configuration).
3. Read-only ingestion of NEXUS_NODES and NEXUS_DEPENDENCIES from schema HACKFEST0316.
4. Robust validation matching mapping agent specifications.
5. Node enrichment using NEXUS_NODES without breaking downstream agent attributes.
6. Zero hardcoding of credentials; error messages sanitize any secret strings.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import pandas as pd

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    # Search for .env in project root or current working dir
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.is_file():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()
except ImportError:
    pass

# Import Member 1's graph builder and validation utilities
from agents.mapping_agent import (
    DataValidationError,
    REQUIRED_COLUMNS,
    build_supply_chain_graph,
    is_blank_or_na,
    np_is_inf,
)


class SAPConfigurationError(ValueError):
    """Raised when SAP HANA configuration or environment variables are missing or invalid."""
    pass


class SAPConnectionError(RuntimeError):
    """Raised when connection to SAP HANA Cloud fails."""
    pass


class SAPDataError(DataValidationError):
    """Raised when records retrieved from SAP HANA fail schema or integrity validation."""
    pass


@dataclass
class SAPConnectionConfig:
    """
    Secure connection configuration for SAP HANA Cloud.
    Passwords are never exposed in string representations or logs.
    """
    address: str
    port: int
    user: str
    password: str
    schema: str = "HACKFEST0316"
    encrypt: bool = True
    ssl_validate_certificate: bool = True
    connect_timeout: int = 10000        # 10 seconds
    communication_timeout: int = 15000  # 15 seconds

    def __repr__(self) -> str:
        # Never expose password in logs, repr, or diagnostics
        return (
            f"SAPConnectionConfig(address='{self.address}', port={self.port}, "
            f"user='{self.user}', password='***', schema='{self.schema}', "
            f"encrypt={self.encrypt}, ssl_validate_certificate={self.ssl_validate_certificate})"
        )

    @classmethod
    def from_env(cls) -> SAPConnectionConfig:
        """
        Loads configuration from environment variables or Streamlit secrets.

        Environment variables:
          - HANA_ADDRESS (or SAP_HANA_HOST): Hostname (e.g. host.hanacloud.ondemand.com or host:port)
          - HANA_PORT (or SAP_HANA_PORT): SQL port (e.g. 443, 30015)
          - HANA_USER (or SAP_HANA_USER): Database user
          - HANA_PASSWORD (or SAP_HANA_PASSWORD): Database password
          - HANA_SCHEMA (or SAP_HANA_SCHEMA): Schema name (default: HACKFEST0316)
          - HANA_ENCRYPT: 'true'/'false' (default: true)
          - HANA_SSL_VALIDATE_CERTIFICATE: 'true'/'false' (default: true)
        """
        # 1. Try Streamlit secrets first if running under Streamlit
        st_secrets: Dict[str, Any] = {}
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                st_secrets = dict(st.secrets.get("hana", st.secrets))
        except Exception:
            pass

        def get_val(keys: List[str], default: Optional[str] = None) -> Optional[str]:
            for key in keys:
                val = os.getenv(key)
                if val:
                    return val.strip()
                val_sec = st_secrets.get(key) or st_secrets.get(key.lower())
                if val_sec:
                    return str(val_sec).strip()
            return default

        raw_address = get_val(["HANA_ADDRESS", "SAP_HANA_HOST", "HANA_HOST"])
        if not raw_address:
            raise SAPConfigurationError(
                "Missing SAP HANA Cloud host. Please set the 'HANA_ADDRESS' environment variable "
                "or define it in your .env / secrets file."
            )

        # 2. Parse host and port: Handle host:port pattern in address
        parsed_port: Optional[int] = None
        if ":" in raw_address:
            host_part, port_str = raw_address.rsplit(":", 1)
            raw_address = host_part.strip()
            if port_str.isdigit():
                parsed_port = int(port_str)

        # Explicit port environment variable takes precedence if provided
        port_env = get_val(["HANA_PORT", "SAP_HANA_PORT"])
        if port_env:
            try:
                final_port = int(port_env)
            except ValueError:
                raise SAPConfigurationError(
                    f"Invalid HANA_PORT: '{port_env}'. Port must be an integer."
                )
        elif parsed_port is not None:
            final_port = parsed_port
        else:
            # Default to 443 if neither explicit port nor address:port was provided
            final_port = 443

        user = get_val(["HANA_USER", "SAP_HANA_USER"])
        if not user:
            raise SAPConfigurationError(
                "Missing SAP HANA Cloud user. Please set the 'HANA_USER' environment variable."
            )

        password = get_val(["HANA_PASSWORD", "SAP_HANA_PASSWORD"])
        if not password:
            raise SAPConfigurationError(
                "Missing SAP HANA Cloud password. Please set the 'HANA_PASSWORD' environment variable."
            )

        schema = get_val(["HANA_SCHEMA", "SAP_HANA_SCHEMA"], default="HACKFEST0316") or "HACKFEST0316"
        encrypt_str = get_val(["HANA_ENCRYPT"], default="true")
        encrypt = str(encrypt_str).lower() in ("true", "1", "yes")

        ssl_validate_str = get_val(["HANA_SSL_VALIDATE_CERTIFICATE"], default="true")
        ssl_validate = str(ssl_validate_str).lower() in ("true", "1", "yes")

        return cls(
            address=raw_address,
            port=final_port,
            user=user,
            password=password,
            schema=schema,
            encrypt=encrypt,
            ssl_validate_certificate=ssl_validate,
        )


def is_sap_configured() -> bool:
    """
    Checks if SAP HANA connection credentials are configured in the environment
    without performing a network round-trip.
    """
    try:
        config = SAPConnectionConfig.from_env()
        return bool(config.address and config.user and config.password)
    except Exception:
        return False


def _sanitize_error_message(msg: str, config: Optional[SAPConnectionConfig] = None) -> str:
    """Removes passwords or secret strings from exception messages."""
    sanitized = str(msg)
    if config and config.password:
        sanitized = sanitized.replace(config.password, "***")
    # Redact any generic password parameters in tracebacks
    sanitized = re.sub(r"password=['\"][^'\"]*['\"]", "password='***'", sanitized, flags=re.IGNORECASE)
    return sanitized


def _sanitize_identifier(name: str) -> str:
    """Ensures a schema or table identifier contains only valid alphanumeric/underscore characters."""
    clean = re.sub(r"[^\w]", "", name)
    if not clean:
        raise ValueError(f"Invalid SQL identifier: '{name}'")
    return clean


def get_sap_connection(config: Optional[SAPConnectionConfig] = None):
    """
    Creates and returns a live connection to SAP HANA Cloud using the official hdbcli driver.

    Parameters:
        config: Optional SAPConnectionConfig instance. If None, loaded from environment.

    Returns:
        hdbcli.dbapi.Connection instance.

    Raises:
        ImportError: If hdbcli is not installed.
        SAPConfigurationError: If configuration is invalid.
        SAPConnectionError: If connection cannot be established.
    """
    try:
        from hdbcli import dbapi
    except ImportError as err:
        raise ImportError(
            "The official SAP HANA client driver 'hdbcli' is not installed. "
            "Please run: pip install hdbcli>=2.20.0"
        ) from err

    if config is None:
        config = SAPConnectionConfig.from_env()

    try:
        conn = dbapi.connect(
            address=config.address,
            port=config.port,
            user=config.user,
            password=config.password,
            encrypt=config.encrypt,
            sslValidateCertificate=config.ssl_validate_certificate,
            connectTimeout=config.connect_timeout,
            communicationTimeout=config.communication_timeout,
        )
        return conn
    except Exception as exc:
        sanitized_msg = _sanitize_error_message(str(exc), config)
        raise SAPConnectionError(
            f"Failed to connect to SAP HANA Cloud ({config.address}:{config.port}): {sanitized_msg}"
        ) from None


def test_sap_connection(config: Optional[SAPConnectionConfig] = None) -> Tuple[bool, str]:
    """
    Performs a lightweight connectivity test against SAP HANA Cloud.

    Returns:
        Tuple[bool, str]: (Success boolean, diagnostic status message)
    """
    try:
        if config is None:
            config = SAPConnectionConfig.from_env()
    except Exception as exc:
        return False, f"Configuration Error: {_sanitize_error_message(str(exc))}"

    try:
        conn = get_sap_connection(config)
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM DUMMY")
                cursor.fetchone()
        finally:
            conn.close()
        return True, f"Successfully connected to SAP HANA Cloud ({config.address}:{config.port})"
    except Exception as exc:
        return False, f"Connection Failed: {_sanitize_error_message(str(exc), config)}"


def validate_sap_dependencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates dependency records retrieved from SAP HANA.
    Ensures columns match Member 1's required format:
      [upstream_node, downstream_node, material, capacity]

    Parameters:
        df: Raw DataFrame from SAP query.

    Returns:
        pd.DataFrame: Validated, lowercase-column DataFrame.
    """
    if df.empty:
        raise SAPDataError("NEXUS_DEPENDENCIES table returned 0 records.")

    # Normalize column names to lowercase and strip whitespace
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise SAPDataError(
            f"NEXUS_DEPENDENCIES is missing required column(s): {missing_cols}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )

    # Validate non-blank node names
    invalid_upstream = is_blank_or_na(df["upstream_node"])
    invalid_downstream = is_blank_or_na(df["downstream_node"])
    bad_nodes = df[invalid_upstream | invalid_downstream].index.tolist()
    if bad_nodes:
        raise SAPDataError(
            f"Found {len(bad_nodes)} dependency row(s) with missing or blank node names."
        )

    # Clean string columns
    for col in ["upstream_node", "downstream_node", "material"]:
        df[col] = df[col].astype(str).str.strip()

    # Validate positive numeric capacity
    numeric_capacities = pd.to_numeric(df["capacity"], errors="coerce")
    invalid_cap_mask = (
        numeric_capacities.isna()
        | (numeric_capacities <= 0)
        | np_is_inf(numeric_capacities)
    )
    if invalid_cap_mask.any():
        bad_values = df.loc[invalid_cap_mask, "capacity"].tolist()
        raise SAPDataError(
            f"Invalid capacity values {bad_values} in NEXUS_DEPENDENCIES. "
            "Capacity must be a positive number greater than zero."
        )

    df["capacity"] = numeric_capacities.astype(float)

    # Check for duplicate relationships
    duplicate_mask = df.duplicated(subset=["upstream_node", "downstream_node"], keep=False)
    if duplicate_mask.any():
        dup_rows = df[duplicate_mask]
        dup_pairs = list(zip(dup_rows["upstream_node"], dup_rows["downstream_node"]))
        raise SAPDataError(
            f"Duplicate supply relationship(s) detected in SAP HANA: {set(dup_pairs)}."
        )

    return df[REQUIRED_COLUMNS]


def validate_sap_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates node master records retrieved from SAP HANA.

    Parameters:
        df: Raw DataFrame from NEXUS_NODES query.

    Returns:
        pd.DataFrame: Validated DataFrame with columns ['node_name', 'node_type'].
    """
    if df.empty:
        raise SAPDataError("NEXUS_NODES table returned 0 records.")

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    expected = ["node_name", "node_type"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SAPDataError(f"NEXUS_NODES missing column(s): {missing}. Expected: {expected}")

    # Check for blank node names
    invalid_names = is_blank_or_na(df["node_name"])
    if invalid_names.any():
        raise SAPDataError("NEXUS_NODES contains rows with missing or blank 'NODE_NAME'.")

    df["node_name"] = df["node_name"].astype(str).str.strip()
    df["node_type"] = df["node_type"].fillna("Other").astype(str).str.strip()

    return df[["node_name", "node_type"]]


def fetch_nexus_dependencies(
    connection: Any = None,
    config: Optional[SAPConnectionConfig] = None,
    schema: Optional[str] = None,
) -> pd.DataFrame:
    """
    Executes a SELECT query to retrieve NEXUS_DEPENDENCIES from SAP HANA Cloud.

    Parameters:
        connection: Optional active hdbcli connection.
        config: Optional configuration if connection needs to be created.
        schema: Optional schema name override.

    Returns:
        pd.DataFrame: Validated dependencies DataFrame.
    """
    effective_schema = _sanitize_identifier(
        schema or (config.schema if config else os.getenv("HANA_SCHEMA", "HACKFEST0316"))
    )

    query = (
        f'SELECT UPSTREAM_NODE, DOWNSTREAM_NODE, MATERIAL, CAPACITY '
        f'FROM "{effective_schema}"."NEXUS_DEPENDENCIES"'
    )

    should_close = False
    conn = connection
    if conn is None:
        conn = get_sap_connection(config)
        should_close = True

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            cols = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

        raw_df = pd.DataFrame(rows, columns=cols)
        return validate_sap_dependencies(raw_df)
    finally:
        if should_close and conn is not None:
            conn.close()


def fetch_nexus_nodes(
    connection: Any = None,
    config: Optional[SAPConnectionConfig] = None,
    schema: Optional[str] = None,
) -> pd.DataFrame:
    """
    Executes a SELECT query to retrieve NEXUS_NODES from SAP HANA Cloud.

    Parameters:
        connection: Optional active hdbcli connection.
        config: Optional configuration if connection needs to be created.
        schema: Optional schema name override.

    Returns:
        pd.DataFrame: Validated nodes DataFrame.
    """
    effective_schema = _sanitize_identifier(
        schema or (config.schema if config else os.getenv("HANA_SCHEMA", "HACKFEST0316"))
    )

    query = f'SELECT NODE_NAME, NODE_TYPE FROM "{effective_schema}"."NEXUS_NODES"'

    should_close = False
    conn = connection
    if conn is None:
        conn = get_sap_connection(config)
        should_close = True

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            cols = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

        raw_df = pd.DataFrame(rows, columns=cols)
        return validate_sap_nodes(raw_df)
    finally:
        if should_close and conn is not None:
            conn.close()


def load_sap_supply_chain(
    config: Optional[SAPConnectionConfig] = None,
) -> Tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    """
    High-level integration function for Nexus-AI:
    1. Connects to SAP HANA Cloud.
    2. Ingests NEXUS_DEPENDENCIES and NEXUS_NODES.
    3. Validates records against supply chain rules.
    4. Constructs the directed graph using Member 1's mapping agent.
    5. Enriches graph nodes with verified database NODE_TYPE attributes.
    6. Preserves all existing graph attributes required by downstream agents.

    Returns:
        Tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
          - graph: Ready-to-use NetworkX directed supply chain graph.
          - dependencies_df: Validated dependencies DataFrame.
          - nodes_df: Validated node master DataFrame.
    """
    if config is None:
        config = SAPConnectionConfig.from_env()

    conn = get_sap_connection(config)
    try:
        dep_df = fetch_nexus_dependencies(connection=conn, config=config)
        nodes_df = fetch_nexus_nodes(connection=conn, config=config)
    finally:
        conn.close()

    # 1. Build baseline graph using Member 1's mapping agent
    graph = build_supply_chain_graph(dep_df)

    # 2. Enrich node attributes using database-backed NEXUS_NODES
    # Preserves in_degree and out_degree while applying verified node_type
    for _, row in nodes_df.iterrows():
        node_name = str(row["node_name"]).strip()
        db_node_type = str(row["node_type"]).strip()

        if node_name not in graph:
            # If a node was defined in NEXUS_NODES but has no edges, include it
            graph.add_node(
                node_name,
                node_type=db_node_type or "Other",
                in_degree=0,
                out_degree=0,
            )
        elif db_node_type:
            # Update with explicit database classification
            graph.nodes[node_name]["node_type"] = db_node_type

    return graph, dep_df, nodes_df

