import pandas as pd

from app.ingestor  import CLAUSE_PATTERNS
from app.generator import generate_answer 


ALL_CLAUSE_TYPES = list(CLAUSE_PATTERNS.keys())

# ── Clauses whose ABSENCE is a red flag ───────────────────────────────────────
HIGH_RISK_MISSING = {
    "cap_on_liability":            "No cap on liability — party may face uncapped damages.",
    "termination_for_convenience": "No termination for convenience — neither party can exit without cause.",
    "governing_law":               "No governing law clause — jurisdiction is ambiguous.",
    "ip_ownership_assignment":     "No IP ownership clause — ownership of created IP is unclear.",
    "audit_rights":                "No audit rights — financial compliance cannot be verified.",
}

# ── Clauses whose PRESENCE is a red flag ─────────────────────────────────────
HIGH_RISK_PRESENT = {
    "non_compete":        "Non-compete detected — review scope and duration carefully.",
    "liquidated_damages": "Liquidated damages detected — fixed penalty on breach.",
}


# ── 1. Cross-Contract Comparison ─────────────────────────────────────────────

def cross_contract_compare(query: str, contract_names: list[str],
                           retriever, reranker) -> str:
    if len(contract_names) < 2:
        return "⚠️ Please select at least two contracts to compare."

    all_chunks = []
    for name in contract_names:
        retrieved  = retriever.retrieve(query, contract_name=name, top_k=10)
        top_chunks = reranker.rerank(query, retrieved)
        all_chunks.extend(top_chunks[:3])   # top 3 per contract

    if not all_chunks:
        return "No relevant clauses found in the selected contracts."

    comparison_query = (
        f"Compare the following across ALL provided contracts: {query}\n\n"
        "For each contract:\n"
        "1. Summarise what the clause says\n"
        "2. Note any differences or risks compared to the others\n"
        "Cite every claim with [Contract: <n>, Clause: <type>]."
    )
    return generate_answer(comparison_query, all_chunks)


# ── 2. Risk Flagging ──────────────────────────────────────────────────────────

def flag_risks(contract_name: str, retriever) -> pd.DataFrame:
    cid = retriever.contract_map.get(contract_name)
    if not cid:
        return pd.DataFrame(columns=["Clause", "Status", "Risk Note", "Excerpt"])

    present: dict[str, str] = {}
    for chunk in retriever.all_chunks:
        meta = chunk["metadata"]
        if meta["contract_id"] != cid:
            continue
        ct = meta["clause_type"]
        if ct not in present:
            present[ct] = chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else "")

    rows = []
    for clause, note in HIGH_RISK_MISSING.items():
        if clause not in present:
            rows.append({"Clause": clause.replace("_", " ").title(),
                         "Status": "⚠️ MISSING", "Risk Note": note, "Excerpt": "—"})

    for clause, note in HIGH_RISK_PRESENT.items():
        if clause in present:
            rows.append({"Clause": clause.replace("_", " ").title(),
                         "Status": "🔴 DETECTED", "Risk Note": note,
                         "Excerpt": present[clause]})

    if not rows:
        rows.append({"Clause": "—", "Status": "✅ No major risks flagged",
                     "Risk Note": "All protective clauses present; no restrictive red flags detected.",
                     "Excerpt": "—"})
    return pd.DataFrame(rows)


# ── 3. Batch Clause Matrix ────────────────────────────────────────────────────

def build_clause_matrix(retriever, contract_names: list[str] = None) -> pd.DataFrame:
    names = contract_names or retriever.contract_names

    presence: dict[str, set] = {
        retriever.contract_map[n]: set()
        for n in names if n in retriever.contract_map
    }

    for chunk in retriever.all_chunks:
        meta = chunk["metadata"]
        cid  = meta["contract_id"]
        ct   = meta["clause_type"]
        if cid in presence and ct != "unknown":
            presence[cid].add(ct)

    rows = []
    for name in names:
        cid = retriever.contract_map.get(name)
        if not cid:
            continue
        found = presence.get(cid, set())
        row   = {"Contract": name}
        for ct in ALL_CLAUSE_TYPES:
            row[ct.replace("_", " ").title()] = "✓" if ct in found else ""
        rows.append(row)

    return pd.DataFrame(rows)