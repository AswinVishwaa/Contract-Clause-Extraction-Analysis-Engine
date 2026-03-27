import re
import hashlib
from pathlib import Path

import fitz  # PyMuPDF

from app.config import CHAR_LIMIT, OVERLAP_CHARS

# Restored official 41 CUAD categories
CLAUSE_PATTERNS = {
    "document_name": r'\b(agreement|contract|amendment|addendum|statement\s+of\s+work)\b',
    "parties": r'\b(entered\s+into|by\s+and\s+between|by\s+and\s+among)\b',
    "agreement_date": r'\b(dated\s+as\s+of|agreement\s+date)\b',
    "effective_date": r'\b(effective\s+date|shall\s+become\s+effective)\b',
    "expiration_date": r'\b(expiration\s+date|expire(s)?\s+on)\b',
    "renewal_term": r'\brenewal\s+term\b|\bauto(matic)?\s*renew',
    "notice_period_to_terminate_renewal": r'\bnotice\s+(to\s+terminate|of\s+non[-\s]renewal)\b',
    "governing_law": r'\bgoverning\s+law\b|\bchoice\s+of\s+law\b',
    "most_favored_nation": r'\b(most\s+favored\s+nation|mfn)\b',
    "revenue_profit_sharing": r'\b(revenue|profit)\s+shar(e|ing)\b',
    "price_restrictions": r'\b(price\s+(restriction|protection|decrease)|lock[-\s]in\s+price)\b',
    "minimum_commitment": r'\bminimum\s+(purchase|commit|order|volume)\b',
    "volume_restriction": r'\b(volume\s+restriction|maximum\s+quantity)\b',
    "non_compete": r'\bnon[-\s]compete\b|\bnot\s+to\s+compete\b',
    "exclusivity": r'\b(exclusive|exclusivity)\b',
    "no_solicit_of_customers": r'\b(non[-\s]solicit|not\s+to\s+solicit)\s+(customers|clients)\b',
    "competitive_restriction_exception": r'\bexception(s)?\s+to\s+(non[-\s]compete|exclusivity)\b',
    "no_solicit_of_employees": r'\b(non[-\s]solicit|not\s+to\s+solicit)\s+(employees|personnel)\b',
    "non_disparagement": r'\b(non[-\s]disparage|disparagement)\b',
    "termination_for_convenience": r'\btermination\s+for\s+convenience\b',
    "rofr_rofo_rofn": r'\bright\s+of\s+first\s+(refusal|offer|negotiation)|rofr|rofo|rofn\b',
    "change_of_control": r'\bchange\s+of\s+control\b',
    "anti_assignment": r'\b(anti[-\s]assignment|assignment\s+and\s+delegation)\b|\bassign\s+this\s+agreement\b',
    "post_termination_services": r'\b(post[-\s]termination\s+services|transition\s+assistance)\b',
    "ip_ownership_assignment": r'\bintellectual\s+property\s+ownership\b|\bip\s+ownership\b|\bassignment\s+of\s+(ip|intellectual)\b',
    "joint_ip_ownership": r'\b(joint|shared)\s+(intellectual\s+property|ip|ownership)\b',
    "license_grant": r'\b(grant\s+of\s+license|license\s+grant)\b',
    "non_transferable_license": r'\bnon[-\s]transferable\s+license\b',
    "affiliate_license_licensor": r'\blicensor\s+affiliate(s)?\b',
    "affiliate_license_licensee": r'\blicensee\s+affiliate(s)?\b',
    "unlimited_license": r'\b(unlimited|all[-\s]you[-\s]can[-\s]eat)\s+license\b',
    "perpetual_license": r'\b(irrevocable|perpetual)\s+license\b',
    "source_code_escrow": r'\b(source\s+code\s+escrow|escrow\s+agent)\b',
    "audit_rights": r'\baudit\s+right(s)?\b',
    "uncapped_liability": r'\b(uncapped|unlimited)\s+liability\b',
    "cap_on_liability": r'\blimitation\s+of\s+liability\b|\bcap\s+on\s+liability\b',
    "liquidated_damages": r'\bliquidated\s+damages\b',
    "warranty_duration": r'\bwarranty\s+(duration|period)\b',
    "insurance": r'\binsurance\s+(policy|coverage|requirements)\b',
    "covenant_not_to_sue": r'\bcovenant\s+not\s+to\s+sue\b',
    "third_party_beneficiary": r'\bthird[-\s]party\s+beneficiar(y|ies)\b'
}

# Restored Kaggle Regex
SECTION_HEADER_RE = re.compile(
    r'(?:^|\n)('
    r'(?:\d+\.)+\d*\s+[A-Z]'          
    r'|[A-Z][A-Z\s]{4,}(?:\.|:|\n)'   
    r'|Section\s+\d+'                 
    r'|Article\s+\d+'                 
    r'|ARTICLE\s+[IVXLC]+'            
    r')',
    re.MULTILINE
)


def clean_text(text: str) -> str:
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(page\s+\d+(?:\s+of\s+\d+)?)\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Restored footer removal
    text = re.sub(r'^\s*(confidential|draft|execution copy|exhibit\s+[a-z])\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'-\n([a-z])', r'\1', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


def detect_clause_type(text: str) -> str:
    text_lower = text.lower()
    for clause_type, pattern in CLAUSE_PATTERNS.items():
        if re.search(pattern, text_lower):
            return clause_type
    return "unknown"

def extract_pages_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """Returns list of (page_number, page_text) — 1-indexed."""
    doc = fitz.open(pdf_path)
    pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    doc.close()
    return pages


# Keep old function as fallback for any callers that pass plain text
def extract_text_from_pdf(pdf_path: str) -> str:
    doc  = fitz.open(pdf_path)
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def chunk_text_with_pages(pages: list[tuple[int, str]],
                           contract_id: str,
                           contract_name: str,
                           source: str = "uploaded") -> list[dict]:
    """
    Section-aware chunking that preserves page numbers in metadata.
    pages: list of (page_number, page_text)
    """
    result    = []
    chunk_idx = 0

    for page_num, raw_page_text in pages:
        cleaned = clean_text(raw_page_text)
        if len(cleaned) <= 80:
            continue

        # split this page on section headers
        positions = [m.start() for m in SECTION_HEADER_RE.finditer(cleaned)]
        positions.append(len(cleaned))

        sections = []
        if positions[0] > 0:
            sections.append(cleaned[:positions[0]])
        for i in range(len(positions) - 1):
            sections.append(cleaned[positions[i]:positions[i + 1]])

        for section in sections:
            section = section.strip()
            if len(section) <= 80:
                continue

            if len(section) <= CHAR_LIMIT:
                sub_chunks = [section]
            else:
                sub_chunks = []
                start = 0
                while start < len(section):
                    end = start + CHAR_LIMIT
                    piece = section[start:end]
                    if len(piece) > 80:
                        sub_chunks.append(piece)
                    start = end - OVERLAP_CHARS

            for piece in sub_chunks:
                result.append({
                    "text": piece,
                    "metadata": {
                        "contract_id":   contract_id,
                        "contract_name": contract_name,
                        "source":        source,
                        "chunk_index":   chunk_idx,
                        "page_number":   page_num,
                        "clause_type":   detect_clause_type(piece),
                        "auto_tagged":   False,
                        "char_length":   len(piece),
                    }
                })
                chunk_idx += 1

    return result


def ingest_pdf(pdf_path: str) -> tuple[str, list[dict]]:
    """
    Full pipeline for a new PDF upload.
    Returns (contract_name, chunks)
    """
    path          = Path(pdf_path)
    contract_name = path.stem
    contract_id   = hashlib.md5(contract_name.encode()).hexdigest()[:12]

    pages  = extract_pages_from_pdf(pdf_path)
    chunks = chunk_text_with_pages(pages, contract_id, contract_name, source="uploaded")

    return contract_name, chunks