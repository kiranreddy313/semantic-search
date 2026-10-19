CLAUDE_MODEL = "claude-haiku-4-5-20251001"
TOP_K = 5
SHOW_PROMPT = False
NOT_FOUND_MESSAGE = "I couldn't find that in the documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the documents provided inside <documents> tags.

Rules:
- If the documents do not contain the answer, reply exactly: "{NOT_FOUND_MESSAGE}" Do not use outside knowledge.
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