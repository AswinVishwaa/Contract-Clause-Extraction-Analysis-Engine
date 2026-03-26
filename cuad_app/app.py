import gradio as gr
import os
import pandas as pd

from app.retriever import HybridRetriever
from app.reranker import Reranker
from app.ingestor import ingest_pdf
from app.generator import generate_answer

# 1. Initialize Backend Services
print("Initializing AI components...")
retriever = HybridRetriever()
reranker  = Reranker()
print("✓ App backend fully initialized.")

def format_chat_history(gradio_history):
    gemini_history = []
    for user_msg, bot_msg in gradio_history:
        gemini_history.append({"role": "user", "parts": [user_msg]})
        gemini_history.append({"role": "model", "parts": [bot_msg]})
    return gemini_history

# 2. Define UI Callbacks
def process_upload(files, progress=gr.Progress()):
    """Tab 2: Handle PDF uploads with progress tracking."""
    if not files:
        choices = ["All Contracts"] + retriever.contract_names
        return gr.Dropdown(choices=choices), "No files uploaded."
    
    progress(0, desc="Starting ingestion...")
    
    for i, file in enumerate(files):
        filename = os.path.basename(file.name)
        progress((i) / len(files), desc=f"Processing {filename}...")
        
        contract_name, chunks = ingest_pdf(file.name)
        retriever.add_chunks(chunks)
        
    progress(1.0, desc="Ingestion complete!")
    
    updated_choices = ["All Contracts"] + retriever.contract_names
    # Return updated dropdown choices and a success message
    return gr.Dropdown(choices=updated_choices, value=updated_choices[-1]), f"✓ Successfully processed {len(files)} document(s)."

def answer_query(message, history, selected_contract):
    """Tab 1: The main RAG pipeline."""
    if not message.strip():
        return "", history

    target_contract = selected_contract if selected_contract != "All Contracts" else None

    # RAG Pipeline
    retrieved_chunks = retriever.retrieve(message, contract_name=target_contract)
    top_chunks = reranker.rerank(message, retrieved_chunks)
    
    gemini_history = format_chat_history(history)
    bot_response = generate_answer(message, top_chunks, history=gemini_history)

    history.append((message, bot_response))
    return "", history 

def explore_contract(selected_contract):
    """Tab 3: Extract and display identified clauses by category."""
    if selected_contract == "All Contracts" or not selected_contract:
        return pd.DataFrame(columns=["Clause Type", "Extracted Text"])
    
    cid = retriever.contract_map.get(selected_contract)
    if not cid:
        return pd.DataFrame(columns=["Clause Type", "Extracted Text"])
    
    # Filter chunks for this specific contract
    data = []
    for chunk in retriever.all_chunks:
        if chunk["metadata"]["contract_id"] == cid:
            ctype = chunk["metadata"]["clause_type"]
            # Skip unknown or generic document names for the summary table
            if ctype not in ["unknown", "document_name"]:
                data.append({
                    "Clause Type": ctype.replace("_", " ").title(),
                    "Extracted Text": chunk["text"]
                })
                
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame([{"Clause Type": "None", "Extracted Text": "No specific CUAD clauses detected."}])
    return df

# 3. Build the UI Layout (3 Tabs)
with gr.Blocks(theme=gr.themes.Soft(), title="Legal RAG System") as demo:
    gr.Markdown("# ⚖️ Legal Contract Analysis System")
    
    # Global Contract Selector (Used across tabs)
    with gr.Row():
        global_contract_selector = gr.Dropdown(
            choices=["All Contracts"] + retriever.contract_names,
            value="All Contracts",
            label="Active Contract (Searchable)",
            info="Type to search. Select a specific contract for Chat or Explorer.",
        )

    with gr.Tabs():
        
        # TAB 1: Chat Interface
        with gr.Tab("💬 1. Chat & Query"):
            gr.Markdown("Ask specific questions about the selected contract. The AI will cite its sources.")
            chatbot = gr.Chatbot(height=500, show_copy_button=True)
            with gr.Row():
                msg_input = gr.Textbox(show_label=False, placeholder="e.g., What is the cap on liability?", scale=4)
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                
            msg_input.submit(answer_query, inputs=[msg_input, chatbot, global_contract_selector], outputs=[msg_input, chatbot])
            submit_btn.click(answer_query, inputs=[msg_input, chatbot, global_contract_selector], outputs=[msg_input, chatbot])
            gr.ClearButton([msg_input, chatbot], value="Clear Chat")

        # TAB 2: Upload & Ingest
        with gr.Tab("📂 2. Upload New PDF"):
            gr.Markdown("Drag and drop new PDF contracts here. They will be chunked, embedded, and added to the database immediately.")
            with gr.Row():
                upload_box = gr.File(label="Upload Contract (PDF)", file_types=[".pdf"], file_count="multiple")
            upload_status = gr.Markdown("Status: Waiting for upload...")
            
            upload_box.upload(
                fn=process_upload,
                inputs=[upload_box],
                outputs=[global_contract_selector, upload_status]
            )

        # TAB 3: Contract Explorer
        with gr.Tab("📊 3. Contract Explorer"):
            gr.Markdown("View all automatically detected legal clauses (e.g., Governing Law, Indemnification) extracted from the selected document.")
            explore_btn = gr.Button("Extract Clauses for Selected Contract", variant="secondary")
            clause_table = gr.Dataframe(
                headers=["Clause Type", "Extracted Text"],
                wrap=True,
                interactive=False
            )
            
            explore_btn.click(
                fn=explore_contract,
                inputs=[global_contract_selector],
                outputs=[clause_table]
            )

# 4. Run the App
if __name__ == "__main__":
    os.makedirs("artifacts/chroma_db", exist_ok=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)