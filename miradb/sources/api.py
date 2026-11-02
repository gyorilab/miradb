"""REST API for searching and downloading MIRA-DB models."""

import io
import json
import logging
import zipfile
from typing import Any, Literal

from flask import Response, send_file
from flask_openapi3 import APIBlueprint, Schema, Tag
from mira.metamodel import SCHEMA_PATH
from pydantic import BaseModel, Field, RootModel

from miradb import export
from miradb.db import queries
from miradb.sources.explorer_ui import get_db_client

logger = logging.getLogger(__name__)

models_tag = Tag(name="models", description="Search and download models")
publications_tag = Tag(name="publications", description="Source publications")

api_blueprint = APIBlueprint("api", __name__, url_prefix="/api")

Format = Literal["json", "sbml", "sympy"]
EXTENSIONS = {"json": "json", "sbml": "xml", "sympy": "py"}
MIMETYPES = {
    "json": "application/json",
    "sbml": "application/xml",
    "sympy": "text/x-python",
}


class ModelFilter(BaseModel):
    q: str | None = Field(
        None,
        description="Text matched against publication metadata and "
        "grounded concepts",
    )
    pmid: list[str] | None = Field(None, description="PubMed IDs")
    method: str | None = Field(None, description="Extraction method")
    concept: str | None = Field(
        None,
        description="CURIE of a grounding or context value, "
        "e.g. ido:0000514",
    )
    best: bool = Field(
        False,
        description="Keep only the highest priority model per publication",
    )


class ModelSearch(ModelFilter):
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class DownloadFilter(ModelFilter):
    format: Format = Field(
        "json",
        description="json gives JSON Lines, sbml and sympy give a zip",
    )


class FormatQuery(BaseModel):
    format: Format = "json"


class PmidPath(BaseModel):
    pmid: str


class ModelPath(PmidPath):
    method: str


class PublicationQuery(BaseModel):
    q: str | None = None


class ModelSummary(BaseModel):
    pmid: str
    method: str
    title: str | None
    year: int | None


class ModelSearchResult(BaseModel):
    total: int
    results: list[ModelSummary]


class ModelRecord(ModelSummary):
    template_model: dict[str, Any]
    grounded_concepts: dict[str, Any] | None


class PublicationSummary(BaseModel):
    pmid: str
    title: str
    author_list: str
    pub_year: int | None
    model_count: int
    scenario_count: int


class PublicationList(RootModel):
    root: list[PublicationSummary]


class Publication(BaseModel):
    pmid: str
    doi: str | None
    pmcid: str | None
    title: str | None
    authors: list[str] | None
    journal: str | None
    year: int | None
    keywords: list[str] | None
    models: list[ModelSummary]


class ScenarioGroups(RootModel):
    root: list[dict[str, Any]]


class Error(BaseModel):
    error: str


# Document TemplateModel payloads with MIRA's own JSON schema, which
# Pydantic would otherwise try to resolve
REF = "#/components/schemas/"
_schemas = json.loads(SCHEMA_PATH.read_text().replace("#/$defs/", REF))
_schemas = _schemas["$defs"]
_schemas["ModelRecord"] = ModelRecord.model_json_schema()
_schemas["ModelRecord"]["properties"]["template_model"] = {
    "$ref": REF + "TemplateModel"
}
api_blueprint.components_schemas.update(
    {name: Schema(**value) for name, value in _schemas.items()}
)

MODEL_RESPONSE = {
    "content": {
        MIMETYPES["json"]: {"schema": {"$ref": REF + "ModelRecord"}},
        MIMETYPES["sbml"]: {"schema": {"type": "string"}},
        MIMETYPES["sympy"]: {"schema": {"type": "string"}},
    }
}
DOWNLOAD_RESPONSE = {
    "content": {
        "application/jsonl": {"schema": {"type": "string"}},
        "application/zip": {"schema": {"type": "string", "format": "binary"}},
    }
}


def _filter_kwargs(f: ModelFilter) -> dict:
    return dict(
        q=f.q, pmids=f.pmid, method=f.method, concept=f.concept, best=f.best
    )


def _model_file(record: dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(record)
    tm = export.load_template_model(record["template_model"])
    return export.to_sbml(tm) if fmt == "sbml" else export.to_sympy(tm)


def _model_response(record: dict | None, fmt: str):
    if record is None:
        return {"error": "Model not found"}, 404
    if fmt == "json":
        return record
    try:
        content = _model_file(record, fmt)
    except Exception:
        logger.exception(f"{fmt} export failed for {record['pmid']}")
        return {"error": f"{fmt} export failed"}, 500
    return send_file(
        io.BytesIO(content.encode()),
        mimetype=MIMETYPES[fmt],
        as_attachment=True,
        download_name=f"{record['pmid']}_{record['method']}."
        f"{EXTENSIONS[fmt]}",
    )


def _download(f: DownloadFilter):
    _, records = queries.search_models(
        get_db_client(), full=True, **_filter_kwargs(f)
    )
    if f.format == "json":
        lines = (json.dumps(r) + "\n" for r in records)
        return Response(
            lines,
            mimetype="application/jsonl",
            headers={
                "Content-Disposition": "attachment; filename=models.jsonl"
            },
        )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in records:
            try:
                content = _model_file(r, f.format)
            except Exception:
                logger.warning(f"Skipping {r['pmid']} {r['method']}")
                continue
            name = f"{r['pmid']}_{r['method']}.{EXTENSIONS[f.format]}"
            zf.writestr(name, content)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"models_{f.format}.zip",
    )


@api_blueprint.get(
    "/models", tags=[models_tag], responses={200: ModelSearchResult}
)
def search_models(query: ModelSearch):
    """Search models

    All given filters must match. Results are ordered by PMID, then by
    extraction method priority.
    """
    total, results = queries.search_models(
        get_db_client(),
        limit=query.limit,
        offset=query.offset,
        **_filter_kwargs(query),
    )
    return {"total": total, "results": results}


@api_blueprint.get(
    "/models/<pmid>",
    tags=[models_tag],
    responses={200: MODEL_RESPONSE, 404: Error},
)
def get_best_model(path: PmidPath, query: FormatQuery):
    """Get the highest priority model of a publication

    The json format returns the model with its metadata, sbml and sympy
    return a file.
    """
    _, records = queries.search_models(
        get_db_client(), pmids=[path.pmid], best=True, full=True
    )
    return _model_response(records[0] if records else None, query.format)


@api_blueprint.get(
    "/models/<pmid>/<method>",
    tags=[models_tag],
    responses={200: MODEL_RESPONSE, 404: Error},
)
def get_model(path: ModelPath, query: FormatQuery):
    """Get the model extracted from a publication by a given method

    The json format returns the model with its metadata, sbml and sympy
    return a file.
    """
    _, records = queries.search_models(
        get_db_client(), pmids=[path.pmid], method=path.method, full=True
    )
    return _model_response(records[0] if records else None, query.format)


@api_blueprint.get(
    "/download", tags=[models_tag], responses={200: DOWNLOAD_RESPONSE}
)
def download_models(query: DownloadFilter):
    """Download all models matching the filters"""
    return _download(query)


@api_blueprint.post(
    "/download", tags=[models_tag], responses={200: DOWNLOAD_RESPONSE}
)
def download_models_post(body: DownloadFilter):
    """Download all models matching the filters, given as a JSON body

    Use this instead of GET for lists of PMIDs too long for a URL.
    """
    return _download(body)


@api_blueprint.get(
    "/publications", tags=[publications_tag], responses={200: PublicationList}
)
def list_publications(query: PublicationQuery):
    """List or search publications that have at least one model"""
    client = get_db_client()
    if query.q:
        return queries.search_publication_summaries(client, query.q)
    return queries.list_publication_summaries(client)


@api_blueprint.get(
    "/publications/<pmid>",
    tags=[publications_tag],
    responses={200: Publication, 404: Error},
)
def get_publication(path: PmidPath):
    """Get a publication's metadata and its models"""
    client = get_db_client()
    ref = queries.get_text_ref(client, path.pmid)
    if ref is None:
        return {"error": "Publication not found"}, 404
    _, models = queries.search_models(client, pmids=[path.pmid])
    fields = Publication.model_fields.keys() - {"models"}
    return {**{k: ref[k] for k in fields}, "models": models}


@api_blueprint.get(
    "/publications/<pmid>/scenarios",
    tags=[publications_tag],
    responses={200: ScenarioGroups},
)
def get_publication_scenarios(path: PmidPath):
    """Get a publication's scenarios and parameters, grouped by method"""
    return queries.list_scenarios_for_pmid(get_db_client(), path.pmid)
