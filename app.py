import json  # NEW

import anthropic  # NEW
from anthropic import Anthropic  # NEW
from flask import Flask, Response, jsonify, render_template, request

from common import get_connection, load_model
from rag_core import (  # NEW
    CLAUDE_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
    build_user_message,
    describe_location,
)
from search_engine import search

app = Flask(__name__)

print("Loading embedding model...")
model = load_model()
client = Anthropic()  # NEW


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


# NEW: everything below
def event(data):
    return json.dumps(data) + "\n"


@app.route("/ask")
def ask_page():
    return render_template("ask.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    with get_connection() as conn:
        results = search(conn, model, question, top_k=TOP_K)

    sources = [
        {
            "number": number,
            "location": describe_location(result),
            "content": result["content"],
            "similarity": result["similarity"],
        }
        for number, result in enumerate(results, start=1)
    ]

    def generate():
        yield event({"type": "sources", "sources": sources})

        if not results:
            yield event({"type": "text", "text": "No relevant passages were found in your documents."})
            yield event({"type": "done", "usage": None})
            return

        try:
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_message(question, results)}],
            ) as stream:
                for text in stream.text_stream:
                    yield event({"type": "text", "text": text})
                final = stream.get_final_message()

            yield event({
                "type": "done",
                "usage": {
                    "input": final.usage.input_tokens,
                    "output": final.usage.output_tokens,
                },
            })
        except anthropic.APIError as error:
            yield event({"type": "error", "message": f"Claude API request failed ({type(error).__name__})."})

    return Response(generate(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    app.run(debug=True)