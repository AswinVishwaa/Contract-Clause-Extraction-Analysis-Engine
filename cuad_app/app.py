import gradio as gr
import os
import pandas as pd

from app.retriever import HybridRetriever
from app.reranker  import Reranker
from app.ingestor  import ingest_pdf
from app.generator import generate_answer_stream
from app.generator import generate_answer_stream, generate_answer_fulltext_stream
import google.generativeai as genai

# ── Boot ─────────────────────────────────────────────────────────────
print("Initializing AI components...")
retriever = HybridRetriever()
reranker  = Reranker()
print("✓ App backend fully initialized.")


# ── Helpers ───────────────────────────────────────────────────────────
def format_chat_history(gradio_history: list) -> list[dict]:
    gemini_history = []
    for msg in gradio_history:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
    return gemini_history


def build_source_footer(chunks: list[dict]) -> str:
    """Build a citation block shown below the streamed answer."""
    if not chunks:
        return ""
    lines = ["\n\n---\n**Sources retrieved:**"]
    seen  = set()
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        key  = f"{meta['contract_name']}|{meta['clause_type']}"
        if key not in seen:
            seen.add(key)
            score = c.get("rerank_score", c.get("score", 0))
            lines.append(
                f"- **[{i}]** `{meta['contract_name']}`  "
                f"· clause: `{meta['clause_type']}`  "
                f"· confidence: `{score:.2f}`"
            )
    return "\n".join(lines)


# ── Callbacks ─────────────────────────────────────────────────────────

def rewrite_query_for_search(user_query: str) -> str:
    """
    Translates a conversational user query into a keyword-dense search string
    optimized for ChromaDB and BM25.
    """
    prompt = f"""You are a legal search optimization AI. 
Extract the core legal concepts, entities, and jurisdictions from the user's question.
Remove conversational filler like "Which contracts", "Tell me about", "Are there any".
Output ONLY the optimized search terms on a single line. Do not wrap in quotes.

User Question: {user_query}
Optimized Search Query:"""

    # Use the fast, cheap lite model so we don't hit your rate limits
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    
    try:
        response = model.generate_content(prompt)
        optimized_query = response.text.strip()
        return optimized_query
    except Exception as e:
        # If the API fails, fall back to the user's original query so the app doesn't crash
        return user_query
    
def answer_query(message: str, history: list, selected_contract: str):
    if not message.strip():
        yield history
        return

    target = selected_contract if selected_contract != "All Contracts" else None

    # 1. Immediate UI update
    history = history + [{"role": "user", "content": message}]
    yield history

    history = history + [{"role": "assistant", "content": "▍"}]
    partial = ""
    gemini_history = format_chat_history(history[:-2])

    # ── STRATEGY A: Full Document Mode ──────────────────────────────
    if target:
        full_text, within_limit = retriever.get_fulltext(target)
        if within_limit and full_text:
            for token in generate_answer_fulltext_stream(message, full_text, history=gemini_history):
                partial = token
                history[-1] = {"role": "assistant", "content": partial + " ▍"}
                yield history
            
            footer = f"\n\n---\n**Source:** `{target}` (Full Document Mode)"
            history[-1] = {"role": "assistant", "content": partial + footer}
            yield history
            return

    # ── STRATEGY B: Hybrid RAG Mode ─────────────────────────────────
    # We use the original message directly - no "Translator" AI needed.
    retrieved  = retriever.retrieve(message, contract_name=target)
    top_chunks = reranker.rerank(message, retrieved)

    # DYNAMIC PROMOTION FALLBACK
    if top_chunks and top_chunks[0]["rerank_score"] > 2.0:
        winner_name = top_chunks[0]["metadata"]["contract_name"]
        full_text, _ = retriever.get_fulltext(winner_name)
        
        if full_text:
            # We use the batching stream we just wrote above
            for token in generate_answer_fulltext_stream(message, full_text, history=gemini_history):
                partial = token
                history[-1] = {"role": "assistant", "content": partial + " ▍"}
                yield history
            
            footer = f"\n\n---\n**Source:** `{winner_name}` (Dynamic Promotion Mode)"
            history[-1] = {"role": "assistant", "content": partial + footer}
            yield history
            return


def process_upload(files, progress=gr.Progress()):
    """Handle PDF uploads — ingest + add to retriever and full-text memory."""
    if not files:
        choices = ["All Contracts"] + retriever.contract_names
        return gr.Dropdown(choices=choices), "⚠️ No files uploaded."

    results = []
    for i, file in enumerate(files):
        filename = os.path.basename(file.name)
        progress(i / len(files), desc=f"Processing {filename}...")
        
        # 1. Chunk and ingest the PDF
        contract_name, chunks = ingest_pdf(file.name)
        
        # 2. Add chunks to ChromaDB (Strategy B)
        retriever.add_chunks(chunks)
        
        # 3. NEW: Stitch chunks back together for Full Document Mode (Strategy A)
        full_text = "\n\n".join(c["text"] for c in chunks)
        retriever.fulltext_by_name[contract_name] = full_text
        
        results.append(f"✓ `{contract_name}` — {len(chunks)} chunks indexed")

    progress(1.0, desc="Done!")
    updated = ["All Contracts"] + retriever.contract_names
    status  = "**Upload complete:**\n" + "\n".join(results)
    return gr.Dropdown(choices=updated, value=updated[-1]), status


def explore_contract(selected_contract: str):
    """Extract and display clause summary table for a contract."""
    if not selected_contract or selected_contract == "All Contracts":
        return pd.DataFrame(columns=["Clause Type", "Excerpt"])

    cid = retriever.contract_map.get(selected_contract)
    if not cid:
        return pd.DataFrame(columns=["Clause Type", "Excerpt"])

    seen  = {}
    for chunk in retriever.all_chunks:
        meta   = chunk["metadata"]
        if meta["contract_id"] != cid:
            continue
        ctype = meta["clause_type"]
        if ctype in ("unknown", "document_name"):
            continue
        # keep only first chunk per clause type for summary
        if ctype not in seen:
            seen[ctype] = chunk["text"][:400] + "..." if len(chunk["text"]) > 400 else chunk["text"]

    if not seen:
        return pd.DataFrame([{
            "Clause Type": "None detected",
            "Excerpt":     "No specific CUAD clauses found in this contract."
        }])

    rows = [
        {
            "Clause Type": k.replace("_", " ").title(),
            "Excerpt":     v
        }
        for k, v in sorted(seen.items())
    ]
    return pd.DataFrame(rows)


def clear_chat():
    return [], ""


# ── UI ────────────────────────────────────────────────────────────────
css = """
.header-bar {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 20px 28px;
    border-radius: 12px;
    margin-bottom: 8px;
}
.header-bar h1 {
    color: #e2e8f0 !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
}
.header-bar p {
    color: #94a3b8 !important;
    margin: 4px 0 0 !important;
    font-size: 0.9rem !important;
}
.stat-box {
    background: var(--color-background-secondary);
    border: 1px solid var(--color-border-tertiary);
    border-radius: 10px;
    padding: 14px 20px;
    text-align: center;
}
.stat-box .stat-num {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--color-text-primary);
}
.stat-box .stat-label {
    font-size: 0.78rem;
    color: var(--color-text-secondary);
    margin-top: 2px;
}
.send-btn { min-width: 90px !important; }
.tab-content { padding-top: 16px; }
footer { display: none !important; }
"""

# stats for header
n_contracts = len(retriever.contract_names)
n_chunks    = len(retriever.all_chunks)

with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Legal RAG") as demo:

    # ── Header ──────────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="header-bar">
        <h1>⚖️ Legal Contract Analysis System</h1>
        <p>Hybrid RAG · Gemini 2.5 Flash · CUAD Dataset · 41 Clause Categories</p>
    </div>
    """)

    # ── Stats row ───────────────────────────────────────────────────
    with gr.Row():
        gr.HTML(f'<div class="stat-box"><div class="stat-num">{n_contracts}</div><div class="stat-label">Contracts indexed</div></div>')
        gr.HTML(f'<div class="stat-box"><div class="stat-num">{n_chunks:,}</div><div class="stat-label">Chunks in vector DB</div></div>')
        gr.HTML(f'<div class="stat-box"><div class="stat-num">41</div><div class="stat-label">Clause categories</div></div>')
        gr.HTML(f'<div class="stat-box"><div class="stat-num">Hybrid</div><div class="stat-label">Dense + BM25 + Reranker</div></div>')

    # ── Global contract selector ─────────────────────────────────────
    with gr.Row():
        contract_selector = gr.Dropdown(
            choices  = ["All Contracts"] + retriever.contract_names,
            value    = "All Contracts",
            label    = "Active Contract",
            info     = "Type to search across 510 contracts. Applies to Chat and Explorer tabs.",
            scale    = 4,
            filterable = True,
        )

    # ── Tabs ─────────────────────────────────────────────────────────
    with gr.Tabs():

        # Tab 1 — Chat
        with gr.Tab("💬 Chat & Query"):
            with gr.Column(elem_classes="tab-content"):
                gr.Markdown(
                    "Ask questions in plain English. "
                    "The system retrieves relevant clauses and cites its sources."
                )
                chatbot = gr.Chatbot(
                    height          = 480,
                    show_copy_button= True,
                    type            = "messages",
                    placeholder     = (
                        "**Select a contract above, then ask a question.**\n\n"
                        "Try:\n"
                        "- *What is the governing law?*\n"
                        "- *Does this contract have a non-compete clause?*\n"
                        "- *What happens if either party wants to terminate early?*"
                    ),
                    avatar_images   = (None, "https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg"),
                )
                with gr.Row():
                    msg_box    = gr.Textbox(
                        placeholder = "Ask about any legal clause...",
                        show_label  = False,
                        scale       = 5,
                        autofocus   = True,
                    )
                    submit_btn = gr.Button("Send ➤", variant="primary",
                                           scale=1, elem_classes="send-btn")
                    clear_btn  = gr.Button("Clear", scale=1)

                # example queries
                gr.Examples(
                    examples=[
                        ["What is the governing law of this agreement?"],
                        ["Does this contract have a cap on liability?"],
                        ["Is there a non-compete clause? What are the restrictions?"],
                        ["What are the termination conditions?"],
                        ["Who owns the intellectual property created under this agreement?"],
                        ["Does either party have audit rights?"],
                    ],
                    inputs=msg_box,
                    label="Example questions",
                )

            # wire streaming
            submit_btn.click(
                fn      = answer_query,
                inputs  = [msg_box, chatbot, contract_selector],
                outputs = [chatbot],
            ).then(lambda: "", outputs=msg_box)

            msg_box.submit(
                fn      = answer_query,
                inputs  = [msg_box, chatbot, contract_selector],
                outputs = [chatbot],
            ).then(lambda: "", outputs=msg_box)

            clear_btn.click(fn=clear_chat, outputs=[chatbot, msg_box])

        # Tab 2 — Upload
        with gr.Tab("📂 Upload New Contract"):
            with gr.Column(elem_classes="tab-content"):
                gr.Markdown(
                    "Upload any PDF contract. It will be chunked, embedded, "
                    "and added to the database — immediately queryable in the Chat tab."
                )
                upload_box = gr.File(
                    label       = "Drop PDF contracts here",
                    file_types  = [".pdf"],
                    file_count  = "multiple",
                )
                upload_status = gr.Markdown("*Waiting for upload...*")

                upload_box.upload(
                    fn      = process_upload,
                    inputs  = [upload_box],
                    outputs = [contract_selector, upload_status],
                )

        # Tab 3 — Explorer
        with gr.Tab("📊 Contract Explorer"):
            with gr.Column(elem_classes="tab-content"):
                gr.Markdown(
                    "Select a contract above and click **Extract Clauses** "
                    "to see all detected clause types and their excerpts."
                )
                explore_btn  = gr.Button("Extract Clauses", variant="secondary")
                clause_table = gr.Dataframe(
                    headers     = ["Clause Type", "Excerpt"],
                    wrap        = True,
                    interactive = False,
                    row_count   = 10,
                )

                explore_btn.click(
                    fn      = explore_contract,
                    inputs  = [contract_selector],
                    outputs = [clause_table],
                )

# ── Launch ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        debug       = True,
        show_error  = True,
    )