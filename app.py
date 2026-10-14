from flask import Flask, jsonify, render_template, request

from common import get_connection, load_model
from search_engine import search

app = Flask(__name__)

print("Loading embedding model...")
model = load_model()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Please enter a search query."}), 400

    with get_connection() as conn:
        results = search(conn, model, query)

    return jsonify({"query": query, "results": results})


if __name__ == "__main__":
    app.run(debug=True)