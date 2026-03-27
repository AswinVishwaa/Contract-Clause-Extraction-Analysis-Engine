import pickle
from pathlib import Path
from collections import defaultdict

ARTIFACTS = Path("artifacts")

print("Loading chunks...")
with open(ARTIFACTS / "all_chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

# group chunks by contract_id, preserve order via chunk_index
grouped = defaultdict(list)
for chunk in all_chunks:
    meta = chunk["metadata"]
    grouped[meta["contract_id"]].append(chunk)

# sort each contract's chunks by chunk_index — order matters
for cid in grouped:
    grouped[cid].sort(key=lambda c: c["metadata"]["chunk_index"])

# build fulltext dict — two keys for flexible lookup
# { contract_name: full_text } and { contract_id: full_text }
fulltext_by_name = {}
fulltext_by_id   = {}

for cid, chunks in grouped.items():
    full_text     = " ".join(c["text"] for c in chunks)
    contract_name = chunks[0]["metadata"]["contract_name"]
    fulltext_by_name[contract_name] = full_text
    fulltext_by_id[cid]             = full_text

print(f"✓ contracts processed : {len(fulltext_by_name)}")
print(f"  avg chars/contract  : {sum(len(t) for t in fulltext_by_name.values()) // len(fulltext_by_name):,}")
print(f"  max chars/contract  : {max(len(t) for t in fulltext_by_name.values()):,}")
print(f"  min chars/contract  : {min(len(t) for t in fulltext_by_name.values()):,}")

# save
out = {
    "by_name": fulltext_by_name,
    "by_id":   fulltext_by_id
}
with open(ARTIFACTS / "contracts_fulltext.pkl", "wb") as f:
    pickle.dump(out, f)

size_mb = (ARTIFACTS / "contracts_fulltext.pkl").stat().st_size / 1e6
print(f"\n✓ saved contracts_fulltext.pkl — {size_mb:.1f} MB")