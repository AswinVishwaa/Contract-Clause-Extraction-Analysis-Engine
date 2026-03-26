import google.generativeai as genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = """You are a precise legal contract analyst.

Rules you must follow without exception:
1. Answer ONLY from the provided contract excerpts below.
2. Cite every claim with [Contract: <name>, Clause: <type>].
3. If the answer is not found in the excerpts, respond exactly:
   "This clause was not found in the provided contract excerpts."
4. Never infer, assume, or use outside legal knowledge.
5. Be concise — lawyers value precision over length.
6. If multiple contracts are referenced, compare them explicitly."""


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into numbered context block."""
    if not chunks:
        return "No relevant excerpts found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        parts.append(
            f"[Excerpt {i}]\n"
            f"Contract : {meta['contract_name']}\n"
            f"Clause   : {meta['clause_type']}\n"
            f"Text     : {chunk['text']}\n"
        )
    return "\n---\n".join(parts)


def generate_answer(
    query: str,
    chunks: list[dict],
    history: list[dict] = None
) -> str:
    """
    Generate answer from Gemini given query + retrieved chunks.
    history: list of {"role": "user"/"model", "parts": [text]}
    """
    if not chunks:
        return "No relevant contract excerpts found. Try rephrasing your question or selecting a specific contract."

    context = build_context(chunks)
    
    # Only include the context and the question in the user prompt
    user_prompt = (
        f"Contract Excerpts:\n{context}\n\n"
        f"Question: {query}"
    )

    # Initialize the model with the SYSTEM_PROMPT natively
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT
    )

    # build chat history if provided
    if history:
        chat = model.start_chat(history=history)
        response = chat.send_message(user_prompt)
    else:
        response = model.generate_content(user_prompt)

    return response.text