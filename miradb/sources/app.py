from flask import Flask
from .explorer_ui import explorer_blueprint

app = Flask(__name__)


app.register_blueprint(explorer_blueprint)

@app.after_request
def add_no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == "__main__":
    import sys
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == "debug"
    app.run(debug=debug, use_reloader=False, port=5000)