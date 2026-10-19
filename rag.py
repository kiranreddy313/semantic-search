import anthropic
from anthropic import Anthropic

from common import get_connection, load_model
from search_engine import search

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
TOP_K = 5
SHOW_PROMPT = False

SYSTEM_PROMPT = """You answer questions using ONLY the documents provided inside <documents> tags.

Rules:
- If the documents do not contain the answer, reply exactly: "I couldn't find that in the documents." Do not use outside knowledge.
- After each fact, cite the document number it came from in square brackets, like [1] or [2][3].
- Keep answers concise: a short paragraph or a few bullet points.
- Answer in plain text without markdown headings."""


def describe_location(result):
    if result["page"]:
        return f"{result['source']}, page {result['page']}"
    return result["source"]


def build_context(results):
    blocks = []
    for number, result in enumerate(results, start=1):
        blocks.append(
            f'<document index="{number}" source="{describe_location(result)}">\n'
            f"{result['content']}\n"
            f"</document>"
        )
    return "<documents>\n" + "\n".join(blocks) + "\n</documents>"


def build_user_message(question, results):
    return f"{build_context(results)}\n\nQuestion: {question}"


print("Loading embedding model...")
embedding_model = load_model()
client = Anthropic()

with get_connection() as conn:
    print("Ready! Ask a question about your documents, or type 'quit'.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        results = search(conn, embedding_model, question, top_k=TOP_K)

        if not results:
            print("\nNo relevant passages found in your documents, so Claude was not asked.")
            continue

        user_message = build_user_message(question, results)
        if SHOW_PROMPT:
            print("\n----- PROMPT SENT TO CLAUDE -----")
            print(user_message)
            print("----- END OF PROMPT -----")

        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIError as error:
            print(f"\n[Error] Claude API request failed: {error}")
            continue

        print("\nAnswer:")
        print(response.content[0].text)

        print("\nSources:")
        for number, result in enumerate(results, start=1):
            print(f"  [{number}] {describe_location(result)} (similarity {result['similarity']})")

        print(f"\n[tokens in: {response.usage.input_tokens}, out: {response.usage.output_tokens}]")