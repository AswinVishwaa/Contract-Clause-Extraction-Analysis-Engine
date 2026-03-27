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
6. If multiple contracts are referenced, compare them explicitly.
7. You may recognize standard legal synonyms (e.g., 'construed under' means 'governing law').
8. Do not rely solely on the 'Clause Type' label; read the actual text provided in the excerpts to find the answer. Even if a chunk is labeled 'document_name' or 'unknown', use its content if it contains the answer."""


def build_context(chunks: list[dict]) -> str:
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


def generate_answer(query: str,
                    chunks: list[dict],
                    history: list[dict] = None) -> str:
    """Blocking call — used internally if needed."""
    if not chunks:
        return "No relevant contract excerpts found. Try rephrasing or selecting a specific contract."

    context = build_context(chunks)
    user_prompt = (
        f"Contract Excerpts:\n{context}\n\n"
        f"Question: {query}"
    )
    
    # Use native system instructions
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT
    )
    
    # Fix: Actually use the history if provided
    if history:
        chat = model.start_chat(history=history)
        response = chat.send_message(user_prompt)
    else:
        response = model.generate_content(user_prompt)
        
    return response.text


def generate_answer_stream(query: str,
                           chunks: list[dict],
                           history: list[dict] = None):
    """
    Streaming generator — yields accumulated text for Gradio typewriter effect.
    """
    if not chunks:
        yield "No relevant contract excerpts found. Try rephrasing or selecting a specific contract."
        return

    context = build_context(chunks)
    user_prompt = (
        f"Contract Excerpts:\n{context}\n\n"
        f"Question: {query}"
    )

    # Use native system instructions
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT
    )

    if history:
        chat = model.start_chat(history=history)
        response = chat.send_message(user_prompt, stream=True)
    else:
        response = model.generate_content(user_prompt, stream=True)

    # Accumulate the text for Gradio's UI updates
    accumulated_text = ""
    for chunk in response:
        if chunk.text:
            accumulated_text += chunk.text
            yield accumulated_text

def generate_answer_fulltext_stream(query: str,
                                    full_text: str,
                                    history: list[dict] = None):
    CHAR_LIMIT_PER_BATCH = 2_500_000

    batches = [full_text[i:i + CHAR_LIMIT_PER_BATCH]
               for i in range(0, len(full_text), CHAR_LIMIT_PER_BATCH)]

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT
    )

    for i, batch_text in enumerate(batches):
        batch_prompt = (
            f"DOCUMENT BATCH {i + 1} of {len(batches)}\n"
            f"CONTEXT:\n{batch_text}\n\n"
            f"QUESTION: {query}\n"
            "INSTRUCTION: If the answer is in this batch, provide it in detail. "
            "If the answer is NOT in this batch, respond EXACTLY with 'NOT_FOUND_IN_BATCH' and nothing else."
        )

        # ── Collect full batch response first, then decide ──────────
        try:
            response = model.generate_content(batch_prompt)
            full_response = response.text.strip()
        except Exception:
            continue

        if full_response == "NOT_FOUND_IN_BATCH":
            continue

        # Answer found — now stream it token by token for the UI
        accumulated = ""
        try:
            stream_response = model.generate_content(batch_prompt, stream=True)
            for chunk in stream_response:
                if chunk.text:
                    accumulated += chunk.text
                    yield accumulated
        except Exception:
            # fallback: just yield the non-streamed response we already have
            yield full_response

        return  # stop after first batch that has the answer

    yield "This information was not found in any section of the document."