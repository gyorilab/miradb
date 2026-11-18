import logging

from sqlalchemy.orm import declarative_base
from sqlalchemy import Boolean, Column, Integer, Float, String, CheckConstraint, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON, JSONB, ARRAY

from datetime import datetime, timezone


logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


Base = declarative_base()

class EpiTable:
    """Base class for all tables in the MIRA Database"""


class TextRef(Base, EpiTable):
    __tablename__ = 'text_references'

    id = Column(Integer, primary_key=True)
    pmid = Column(String, unique=True, nullable=False)
    doi = Column(String, unique=True, nullable=True)
    pmcid = Column(String, unique=True, nullable=True)
    authors = Column(ARRAY(String), nullable=True)
    title = Column(String, nullable=True)
    journal = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    keywords = Column(ARRAY(String), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    text_contents = relationship(
        "TextContent",
        back_populates="text_ref_obj",
        cascade="all, delete-orphan",
    )
    annotation = relationship(
        "TextAnnotation",
        back_populates="text_ref_obj",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self):
        return f"<TextRef(id='{self.id}', pmid='{self.pmid}')>"

    def to_dict(self):
        return {
            'id': self.id,
            'pmid': self.pmid,
            'doi': self.doi,
            'pmcid': self.pmcid,
            'authors': self.authors,
            'title': self.title,
            'journal': self.journal,
            'year': self.year,
            'keywords': self.keywords,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class TextAnnotation(Base, EpiTable):
    """Paper-level MIRA ``Annotations`` that are not already in TextRef.

    One row per paper, keyed by PMID. Title, authors, DOI, PMCID, journal,
    year and keywords stay in ``text_references``; this table holds the rest
    of the MIRA ``Annotations`` fields. Each term list holds CURIEs
    (diseases: doid/mondo/mesh, pathogens and hosts: ncbitaxon/mesh,
    locations: geonames, model_types: mamo), and ``term_provenance`` records
    each term's label and the sources that found it, e.g.
    ``{"diseases": {"doid:0080600": {"label": "COVID-19",
    "sources": ["mesh", "pubtator"]}}}``.
    """
    __tablename__ = 'text_annotations'

    id = Column(Integer, primary_key=True)
    pmid = Column(
        String,
        ForeignKey("text_references.pmid", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    abstract = Column(String, nullable=True)
    license = Column(String, nullable=True)
    time_scale = Column(String, nullable=True)
    time_start = Column(DateTime(timezone=True), nullable=True)
    time_end = Column(DateTime(timezone=True), nullable=True)
    diseases = Column(ARRAY(String), nullable=False, default=list)
    pathogens = Column(ARRAY(String), nullable=False, default=list)
    hosts = Column(ARRAY(String), nullable=False, default=list)
    locations = Column(ARRAY(String), nullable=False, default=list)
    model_types = Column(ARRAY(String), nullable=False, default=list)
    term_provenance = Column(JSONB, nullable=True)
    # True if NLM indexed the paper with MeSH (MEDLINE), so MeSH terms were
    # available as a source in addition to PubTator and text matching
    medline_indexed = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    text_ref_obj = relationship("TextRef", back_populates="annotation")

    def __repr__(self):
        return f"<TextAnnotation(id='{self.id}', pmid='{self.pmid}')>"

    def to_dict(self):
        return {
            'id': self.id,
            'pmid': self.pmid,
            'abstract': self.abstract,
            'license': self.license,
            'time_scale': self.time_scale,
            'time_start': self.time_start.isoformat() if self.time_start else None,
            'time_end': self.time_end.isoformat() if self.time_end else None,
            'diseases': self.diseases,
            'pathogens': self.pathogens,
            'hosts': self.hosts,
            'locations': self.locations,
            'model_types': self.model_types,
            'term_provenance': self.term_provenance,
            'medline_indexed': self.medline_indexed,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class ExtractionMethod(Base, EpiTable):
    __tablename__ = 'extraction_method'

    id = Column(Integer, primary_key=True)
    extraction_method = Column(String, unique=True, nullable=False)
    extraction_method_desc = Column(String, nullable=True)
    text_ref = relationship(
        "TextContent",
        back_populates="extraction_method_obj",
        cascade="all, delete-orphan",
    )
    ode_ref = relationship(
        "ODEs",
        back_populates="extraction_method_obj",
        cascade="all, delete-orphan",
    )
    scenario_ref = relationship(
        "Scenario",
        back_populates="extraction_method_obj",
        cascade="all, delete-orphan",
    )
    scenario_parameter_ref = relationship(
        "ScenarioParameter",
        back_populates="extraction_method_obj",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<ExtractionMethod(id='{self.id}', extraction_method='{self.extraction_method}')>"

    def to_dict(self):
        return {
            'id': self.id,
            'extraction_method': self.extraction_method,
            'extraction_method_desc': self.extraction_method_desc,
        }


class TextContent(Base, EpiTable):
    __tablename__ = 'text_contents'

    id = Column(Integer, primary_key=True)
    text_ref = Column(
        Integer,
        ForeignKey("text_references.id", ondelete="CASCADE"),
        nullable=False,
    )
    folder_path = Column(String, nullable=False)
    extraction_method_id = Column(
        Integer, ForeignKey("extraction_method.id"), nullable=False
    )
    extracted_info_path = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    text_ref_obj = relationship("TextRef", back_populates="text_contents")
    extraction_method_obj = relationship(
        "ExtractionMethod", back_populates="text_ref"
    )
    odes = relationship(
        "ODEs", back_populates="text_con_obj", cascade="all, delete-orphan"
    )
    scenarios = relationship(
        "Scenario", back_populates="text_con_obj", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint(
            'text_ref',
            'extraction_method_id',
            name='uq_text_ref_extraction_method'
        ),
    )

    def __repr__(self):
        return f"<TextContent(id='{self.id}', text_ref='{self.text_ref}')>"

    def to_dict(self):
        return {
            'id': self.id,
            'text_ref': self.text_ref,
            'folder_path': self.folder_path,
            'extraction_method_id': self.extraction_method_id,
            'extracted_info_path': self.extracted_info_path,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class ODEs(Base, EpiTable):
    __tablename__ = 'ode_expressions'

    id = Column(Integer, primary_key=True)
    txt_content_ref = Column(
        Integer,
        ForeignKey("text_contents.id", ondelete="CASCADE"),
        nullable=False,
    )
    ode = Column(String, nullable=False)
    corrected_ode = Column(String, nullable=True)
    extraction_method_id = Column(
        Integer, ForeignKey("extraction_method.id"), nullable=False
    )
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    mira_model = relationship(
        "MiraModel", back_populates="ode_obj", cascade="all, delete-orphan"
    )
    extraction_method_obj = relationship(
        "ExtractionMethod", back_populates="ode_ref"
    )

    __table_args__ = (
        CheckConstraint(
            "ode IS NOT NULL OR corrected_ode IS NOT NULL",
            name="ck_at_least_one_not_null_ode"
        ),
        UniqueConstraint('txt_content_ref', name='uq_txt_content_ref'),
    )

    text_con_obj = relationship("TextContent", back_populates="odes")

    def __repr__(self):
        return f"<ODEs(id='{self.id}', ode='{self.ode}')>"

    def to_dict(self):
        return {
            'id': self.id,
            'txt_content_ref': self.txt_content_ref,
            'ode': self.ode,
            'corrected_ode': self.corrected_ode,
            'extraction_method_id': self.extraction_method_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class MiraModel(Base, EpiTable):
    __tablename__ = 'mira_template_models'

    id = Column(Integer, primary_key=True)
    ode_ref = Column(
        Integer,
        ForeignKey("ode_expressions.id", ondelete="CASCADE"),
        nullable=False,
    )
    grounded_concepts = Column(JSON, nullable=False)
    mira_template_model = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    ode_obj = relationship("ODEs", back_populates="mira_model")
    scenarios = relationship("Scenario", back_populates="mira_model_obj")

    def __repr__(self):
        return f"<MiraModel(id='{self.id}', template_model='{self.mira_template_model}')>"

    __table_args__ = (
    UniqueConstraint('ode_ref', name='uq_ode_ref'),
)

    def to_dict(self):
        return {
            'id': self.id,
            'ode_ref': self.ode_ref,
            'grounded_concepts': self.grounded_concepts,
            'mira_template_model': self.mira_template_model,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class Scenario(Base, EpiTable):
    __tablename__ = 'scenarios'

    id = Column(Integer, primary_key=True)
    txt_content_ref = Column(
        Integer,
        ForeignKey("text_contents.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Template model whose parameter names were used during extraction, if any
    mira_model_ref = Column(
        Integer,
        ForeignKey("mira_template_models.id", ondelete="SET NULL"),
        nullable=True,
    )
    extraction_method_id = Column(
        Integer, ForeignKey("extraction_method.id"), nullable=False
    )
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    scenarios_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    text_con_obj = relationship("TextContent", back_populates="scenarios")
    mira_model_obj = relationship("MiraModel", back_populates="scenarios")
    extraction_method_obj = relationship(
        "ExtractionMethod", back_populates="scenario_ref"
    )
    parameters = relationship(
        "ScenarioParameter",
        back_populates="scenario_obj",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            'txt_content_ref',
            'extraction_method_id',
            'name',
            name='uq_scenario_txt_content_method_name'
        ),
    )

    def __repr__(self):
        return f"<Scenario(id='{self.id}', name='{self.name}')>"

    def to_dict(self):
        return {
            'id': self.id,
            'txt_content_ref': self.txt_content_ref,
            'mira_model_ref': self.mira_model_ref,
            'extraction_method_id': self.extraction_method_id,
            'name': self.name,
            'description': self.description,
            'scenarios_path': self.scenarios_path,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class ScenarioParameter(Base, EpiTable):
    __tablename__ = 'scenario_parameters'

    id = Column(Integer, primary_key=True)
    scenario_ref = Column(
        Integer,
        ForeignKey("scenarios.id", ondelete="CASCADE"),
        nullable=False,
    )
    extraction_method_id = Column(
        Integer, ForeignKey("extraction_method.id"), nullable=False
    )
    name = Column(String, nullable=False)
    display_name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    value = Column(Float, nullable=True)
    # Unevaluated value expression, e.g. '1/5.2'
    value_expression = Column(String, nullable=True)
    units = Column(String, nullable=True)
    # e.g. {"type": "Uniform1", "parameters": {"minimum": 0.1, "maximum": 0.5}}
    distribution = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    scenario_obj = relationship("Scenario", back_populates="parameters")
    extraction_method_obj = relationship(
        "ExtractionMethod", back_populates="scenario_parameter_ref"
    )

    __table_args__ = (
        UniqueConstraint(
            'scenario_ref',
            'extraction_method_id',
            'name',
            name='uq_scenario_parameter_method_name'
        ),
    )

    def __repr__(self):
        return f"<ScenarioParameter(id='{self.id}', name='{self.name}', value='{self.value}')>"

    def to_dict(self):
        return {
            'id': self.id,
            'scenario_ref': self.scenario_ref,
            'extraction_method_id': self.extraction_method_id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'value': self.value,
            'value_expression': self.value_expression,
            'units': self.units,
            'distribution': self.distribution,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
