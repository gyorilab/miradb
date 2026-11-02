"""Convert stored TemplateModel JSON into downloadable formats."""

import math
import re

from mira.metamodel import TemplateModel
from mira.metamodel.template_model import Time
from mira.modeling import Model
from mira.modeling.ode import OdeModel
from mira.modeling.sbml import template_model_to_sbml_string

symbol_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


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
    om = OdeModel(model=Model(template_model=tm), initialized=False)
    kinetics = om.get_interpretable_kinetics()

    # Collect (lhs_str, rhs_str) pairs
    ode_pairs = []
    if hasattr(kinetics, "tolist"):
        for row in kinetics.tolist():
            if len(row) == 3:
                lhs, _, rhs = row
                ode_pairs.append((str(lhs), str(rhs)))
            elif len(row) == 2:
                lhs, rhs = row
                ode_pairs.append((str(lhs), str(rhs)))
    elif isinstance(kinetics, (list, tuple)):
        for expr in kinetics:
            ode_pairs.append((str(expr), ""))
    else:
        ode_pairs.append((str(kinetics), ""))

    lines = ["from sympy import *", ""]

    # Declare all symbols that appear across lhs + rhs
    all_symbols: set[str] = set()
    skip = {"Derivative", "Function", "Symbol", "symbols", "t"}
    for lhs, rhs in ode_pairs:
        for token in symbol_re.findall(lhs + " " + rhs):
            if token not in skip:
                all_symbols.add(token)

    if all_symbols:
        names = sorted(all_symbols)
        lines.append("# Declare symbols")
        lines.append(f"{', '.join(names)} = symbols('{' '.join(names)}')")
        lines.append("")

    for lhs, rhs in ode_pairs:
        lines.append(f"# {lhs} = {rhs}" if rhs else f"# {lhs}")

    lines += ["", "odes = {"]
    for lhs, rhs in ode_pairs:
        if rhs:
            lines.append(f"    {lhs!r}: {rhs},")
    lines.append("}")
    return "\n".join(lines) + "\n"
