import pickle
from collections import Counter
from app.config import CHUNKS_PATH

print(f"Loading database from: {CHUNKS_PATH}...\n")

try:
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
except FileNotFoundError:
    print("❌ Error: all_chunks.pkl not found! Check your artifacts folder.")
    exit()

total_chunks = len(chunks)
print(f"📊 Total chunks in database: {total_chunks:,}")

# Extract and count every clause type
clause_counts = Counter(c["metadata"]["clause_type"] for c in chunks)

print(f"🔍 Distinct Clause Categories Found: {len(clause_counts)}")
print("-" * 50)

# Print sorted from most common to least common
for clause, count in clause_counts.most_common():
    print(f"{clause.ljust(40)} : {count:,}")

print("-" * 50)
if clause_counts.get("document_name", 0) > 20000:
    print("⚠️ WARNING: Almost everything is tagged as 'document_name'. You are using the OLD database files!")
else:
    print("✅ SUCCESS: Your clauses are properly distributed. You are using the NEW database files!")