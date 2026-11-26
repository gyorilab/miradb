import logging
import threading
import time
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request, send_file, abort
import json
import io
from miradb import export
from miradb.db.client import get_client
from miradb.db import queries
from miradb.db.extraction_methods import EXTRACTION_METHODS_PRIORITY


logger = logging.getLogger(__name__)

explorer_blueprint = Blueprint("explorer", __name__, url_prefix="/explorer")
explorer_blueprint.template_folder = "templates"

_client = None


def get_db_client():
    """Return the database client, creating it on first use.

    Initialized lazily so importing this module does not require a reachable
    database.
    """
    global _client
    if _client is None:
        _client = get_client("primary")
    return _client


@explorer_blueprint.route("/")
def index():
    """Serve the explorer SPA shell."""
    return render_template(
        "explorer.html",
        extraction_method_priority=EXTRACTION_METHODS_PRIORITY,
    )


@explorer_blueprint.route("/api/search")
def search_pmids():
    """Search across text_references metadata and grounded_concepts JSON.

    The search is performed over:
      - text_references, including pmid, title, authors, and publication year.
      - mira_template_models.grounded_concepts JSON, which covers variable
        names, ontology IDs, context keys and values.

    Response format:
    [
        {
            "pmid": "...",
            "title": "...",
            "author_list": "...",
            "pub_year": ...,
            "model_count": ...,
            "scenario_count": ...
        },
        ...
    ]

    Only publications with at least one extracted model (model_count > 0) are
    included. Returns empty list if no query is provided.
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    return jsonify(queries.search_publication_summaries(get_db_client(), q))


@explorer_blueprint.route("/api/pmids")
def get_all_pmids():
    """Return text_references that have at least one extracted model.

    Response format:
    [
        {
            "pmid": "33451107",
            "title": "An SEIR model of influenza",
            "author_list": "Zhang et al.",
            "pub_year":   2021,
            "model_count": 3,
            "scenario_count": 2
        },
        ...
    ]

    Only publications with model_count > 0 are included.
    """
    rows = queries.list_publication_summaries(get_db_client())
    return jsonify(rows)


# GeoNames location ID -> country code, built by
# scripts/build_geonames_countries.py
GEONAMES_COUNTRIES_PATH = Path(__file__).parent / "geonames_countries.json"

OVERVIEW_TTL_SECONDS = 600
_overview_cache = {"data": None, "time": 0.0}
_overview_lock = threading.Lock()


def _load_geonames_countries() -> dict:
    try:
        return json.loads(GEONAMES_COUNTRIES_PATH.read_text())
    except FileNotFoundError:
        logger.warning(
            "%s not found; run scripts/build_geonames_countries.py. "
            "The overview map will be empty.", GEONAMES_COUNTRIES_PATH
        )
        return {"locations": {}, "countries": {}}


def _build_overview() -> dict:
    geo = _load_geonames_countries()
    to_country = geo["locations"]
    papers = queries.list_annotation_overview(get_db_client())
    for paper in papers:
        paper["countries"] = sorted({
            to_country[t["id"].partition(":")[2]]
            for t in paper["locations"]
            if t["id"].startswith("geonames:")
            and t["id"].partition(":")[2] in to_country
        })
    return {"papers": papers, "countries": geo["countries"]}


@explorer_blueprint.route("/api/overview")
def get_overview():
    """Return per-paper annotations for the explorer overview map and charts.

    The response is cached for OVERVIEW_TTL_SECONDS, since it covers every
    publication and the underlying data changes only on ingestion.

    Response format:
    {
        "papers": [
            {
                "pmid": "34671103",
                "title": "...",
                "author_list": "...",
                "pub_year": 2021,
                "model_count": 5,
                "scenario_count": 1,
                "diseases":  [{"id": "doid:0080600", "label": "COVID-19"}],
                "pathogens": [...],
                "hosts":     [...],
                "locations": [{"id": "geonames:1733045", "label": "Malaysia"}],
                "countries": ["MY", "CN"],
                "compartments": 3
            },
            ...
        ],
        "countries": {"MY": {"name": "Malaysia", "iso_numeric": "458"}, ...}
    }
    """
    with _overview_lock:
        now = time.monotonic()
        if (_overview_cache["data"] is None
                or now - _overview_cache["time"] > OVERVIEW_TTL_SECONDS):
            _overview_cache["data"] = _build_overview()
            _overview_cache["time"] = now
        return jsonify(_overview_cache["data"])


@explorer_blueprint.route("/api/pmids/<pmid>/models")
def get_models_for_pmid(pmid: str):
    """Return all ode_expressions for a PMID, with LaTeX-rendered equations.

    Response format:
    [
        {
            "id": 1,
            "extraction_method": "marker",
            "method_label": "Marker Extraction",
            "latex": ["\\frac{dS}{dt} = ...", ...]
            "grounded_concepts": {}
        },
      ...
    ]
    """
    results = queries.list_models_for_pmid(get_db_client(), pmid)
    return jsonify(results)


@explorer_blueprint.route("/api/pmids/<pmid>/scenarios")
def get_scenarios_for_pmid(pmid: str):
    """Return the scenarios and parameters for a PMID, grouped by extractor.

    Response format:
    [
        {
            "extraction_method": "marker",
            "method_label": "Marker Extraction",
            "scenarios": [
                {
                    "id": 1,
                    "name": "default",
                    "description": "...",
                    "parameters": [
                        {
                            "name": "beta",
                            "display_name": "β",
                            "description": "...",
                            "value": 0.5,
                            "value_expression": null,
                            "units": "1/day",
                            "distribution": null
                        },
                        ...
                    ]
                },
                ...
            ]
        },
        ...
    ]
    """
    results = queries.list_scenarios_for_pmid(get_db_client(), pmid)
    return jsonify(results)


@explorer_blueprint.route("/api/models/<int:ode_id>/download/json")
def download_json(ode_id: int):
    """Export the given model as a MIRA TemplateModel JSON"""
    tm = queries.get_template_model_by_ode_id(get_db_client(), ode_id)
    if tm is None:
        abort(404, description=f"No TemplateModel found for ode id {ode_id}")

    json_bytes = json.dumps(tm.to_json(), indent=2).encode("utf-8")

    return send_file(
        io.BytesIO(json_bytes),
        mimetype="application/json",
        as_attachment=True,
        download_name=f"model_{ode_id}.json",
    )


@explorer_blueprint.route("/api/models/<int:ode_id>/download/sbml")
def download_sbml(ode_id: int):
    """Export the given model as SBML xml"""
    tm = queries.get_template_model_by_ode_id(get_db_client(), ode_id)
    if tm is None:
        abort(404, description=f"No TemplateModel found for ode id {ode_id}")

    try:
        sbml_str = export.to_sbml(tm)
    except Exception as e:
        logger.error(f"SBML export failed for ode id {ode_id}")
        logger.exception(e)
        abort(500, description=f"SBML export failed for ode id {ode_id}")

    return send_file(
        io.BytesIO(sbml_str.encode()),
        mimetype="application/xml",
        as_attachment=True,
        download_name=f"model_{ode_id}.xml",
    )


@explorer_blueprint.route("/api/models/<int:ode_id>/download/sympy")
def download_sympy(ode_id: int):
    """Export ODEs as a SymPy .py file"""
    tm = queries.get_template_model_by_ode_id(get_db_client(), ode_id)

    if tm is None:
        abort(404, description=f"No TemplateModel found for ode id {ode_id}")

    try:
        py_str = export.to_sympy(tm)
    except Exception:
        logger.exception(f"SymPy export failed for ode_id={ode_id}")
        abort(500, description="SymPy ODE export failed - see server logs.")

    return send_file(
        io.BytesIO(py_str.encode()),
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"model_{ode_id}.py",
    )
