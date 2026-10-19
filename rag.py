import anthropic
from anthropic import Anthropic
from rag_core import CLAUDE_MODEL, SYSTEM_PROMPT, TOP_K, build_user_message, describe_location

from common import get_connection, load_model
from search_engine import search




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