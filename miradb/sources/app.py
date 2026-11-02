import click
from flask import redirect
from flask_openapi3 import Info, OpenAPI
from .api import api_blueprint
from .explorer_ui import explorer_blueprint

app = OpenAPI(
    __name__,
    info=Info(title="MIRA-DB API", version="0.1.0"),
    doc_prefix="/api/docs",
)


app.register_blueprint(explorer_blueprint)
app.register_api(api_blueprint)


@app.after_request
def add_no_cache(response):
    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/health")
def health():
    return "OK", 200


@app.route("/")
def index():
    # Redirect to /explorer
    return redirect("/explorer", code=301)


@click.command()
@click.option(
    "--host",
    default="0.0.0.0",
    show_default=True,
    help="Address to bind the server to.",
)
@click.option(
    "--port", default=5001, show_default=True, help="Port to run the server on."
)
@click.option(
    "--debug/--no-debug",
    is_flag=True,
    default=False,
    show_default=True,
    help="Enable or disable Flask debug mode.",
)
def main(host, port, debug):
    """Run the Flask application."""
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    main()
