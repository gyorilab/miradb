"""Convert stored TemplateModel JSON into downloadable formats."""

import math

from mira.metamodel import TemplateModel
from mira.metamodel.template_model import Time
from mira.modeling import Model
from mira.modeling.ode import OdeModel
from mira.modeling.sbml import template_model_to_sbml_string


def load_template_model(data: dict) -> TemplateModel:
    """Return a TemplateModel from its JSON with a default time variable."""
    tm = TemplateModel.from_json(data)
    tm.time = Time(name="t", units=None)
    return tm


def to_sbml(tm: TemplateModel) -> str:
    """Return the model as an SBML XML string."""
    # SBML does not allow empty/nan parameter values
    for param in tm.parameters.values():
        if param.value is None or (
            isinstance(param.value, float) and math.isnan(param.value)
        ):
            param.value = 0.0
        elif not isinstance(param.value, (int, float)):
            try:
                param.value = float(param.value)
            except (TypeError, ValueError):
                param.value = 0.0
    return template_model_to_sbml_string(tm)


def to_sympy(tm: TemplateModel) -> str:
    """Return the model's ODEs as the source of a SymPy Python file."""
    return OdeModel(Model(tm), initialized=False).get_sympy_code()
