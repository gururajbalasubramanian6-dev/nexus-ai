"""
Nexus-AI: Main Application Entry Point
Team Project: Supply Chain Resilience

Member Responsibilities:
- Member 1: Supply Chain Mapping Agent (Implemented below)
- Member 2: Disruption Simulation Agent (Hook ready)
- Member 3: Risk Assessment Agent (Hook ready)
- Member 4: Optimization Agent (Hook ready)
- Member 5: Dashboard / Reporting Agent (Hook ready)
"""

import argparse
from pathlib import Path
import sys
import webbrowser

# Add current workspace to path for robust imports
WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from agents.mapping_agent import (
    DataValidationError,
    build_supply_chain_graph,
    create_interactive_graph,
    load_and_validate_dependencies,
    print_graph_summary,
)


def serve_test_page(html_file: Path, port: int = 8080, open_browser: bool = True) -> None:
    """Starts a lightweight local HTTP server to display the interactive graph test page."""
    import http.server
    import socketserver

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(html_file.parent), **kwargs)

        def do_GET(self):
            if self.path in ("/", "/index.html", ""):
                self.path = f"/{html_file.name}"
            return super().do_GET()

        def log_message(self, format, *args):
            # Suppress noisy standard request logs for cleaner student output
            pass

    server_address = ("127.0.0.1", port)
    try:
        with socketserver.TCPServer(server_address, CustomHandler) as httpd:
            url = f"http://127.0.0.1:{port}/"
            print(f"\n[Live Web Page] Serving interactive test page at: {url}")
            print("                 Press Ctrl+C in terminal to stop.")
            if open_browser:
                webbrowser.open(url)
            httpd.serve_forever()
    except OSError as err:
        print(f"\n[Notice] Could not start server on port {port}: {err}")
        print(f"You can still view the graph by opening: {html_file.resolve()}")


def run_supply_chain_mapping(
    csv_file: Path,
    output_html: Path,
    open_browser: bool = False,
    serve: bool = False,
    port: int = 8080,
) -> None:
    """
    Executes Member 1's Supply Chain Mapping pipeline:
    1. Loads and validates supply chain dependencies from CSV.
    2. Constructs the NetworkX directed graph.
    3. Prints human-readable summary metrics.
    4. Generates an interactive Plotly HTML visualization.
    5. Optionally opens the visualization or serves it via a local test page.
    """
    print("=" * 65)
    print("             NEXUS-AI: SUPPLY CHAIN RESILIENCE PLATFORM")
    print("=" * 65)
    print(f"[Step 1/3] Reading & validating data from: {csv_file}")

    try:
        df = load_and_validate_dependencies(csv_file)
        print(f"           SUCCESS: Loaded {len(df)} valid supply relationships.")
    except (FileNotFoundError, DataValidationError) as err:
        print(f"\n[ERROR] Data validation failed:\n{err}")
        sys.exit(1)

    print("\n[Step 2/3] Constructing directed supply chain graph...")
    graph = build_supply_chain_graph(df)
    print_graph_summary(graph)

    print(f"[Step 3/3] Generating interactive visualization...")
    fig = create_interactive_graph(
        graph,
        title="Nexus-AI: Supply Chain Network Map",
        output_html=output_html,
    )
    print(f"           SUCCESS: Interactive graph generated at:")
    print(f"           {output_html.resolve()}")

    if serve:
        serve_test_page(output_html, port=port, open_browser=open_browser)
    elif open_browser:
        print("\nOpening interactive map in your web browser...")
        webbrowser.open(output_html.resolve().as_uri())

    print("\n" + "=" * 65)
    print(" Supply Chain Mapping Agent executed successfully!")
    print(f" To view the graph, open: {output_html.name} in any web browser.")
    print("=" * 65)


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Nexus-AI: Supply Chain Mapping Application"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=WORKSPACE_DIR / "data" / "dependencies.csv",
        help="Path to dependencies CSV file (default: data/dependencies.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_DIR / "supply_chain_graph.html",
        help="Path to save output HTML graph (default: supply_chain_graph.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open the interactive HTML graph in your default web browser",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a local web test server to display the graph in your browser",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number for the local test server (default: 8080)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_supply_chain_mapping(
        csv_file=args.csv,
        output_html=args.output,
        open_browser=args.open,
        serve=args.serve,
        port=args.port,
    )

