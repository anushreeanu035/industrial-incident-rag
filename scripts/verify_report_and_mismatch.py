import sys
import os
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from incident_rag.dependencies import get_rag_pipeline
from incident_rag.generation import validate_equipment_relevance
from incident_rag.schemas import InvestigationRequest
from frontend.components.evidence_cards import generate_document_analysis_markdown

def main():
    print("=" * 70)
    print("1. RUNNING M-301 DOCUMENT ANALYSIS VERIFICATION")
    print("=" * 70)
    pipeline = get_rag_pipeline()
    
    # Locate M-301 document in store
    m301_doc = None
    for doc in pipeline.doc_store.list_documents():
        if "M-301" in getattr(doc, "machine_id", "") or "044" in getattr(doc, "filename", ""):
            m301_doc = doc
            break
            
    if not m301_doc:
        print("ERROR: M-301 document not found in DocumentStore!")
        sys.exit(1)
        
    print(f"Found document: {m301_doc.filename} (ID: {m301_doc.document_id})")
    analysis_resp = pipeline.analyze_document(m301_doc.document_id)
    analysis_dict = analysis_resp.model_dump()
    
    markdown_report = generate_document_analysis_markdown(analysis_dict)
    
    print("\n--- GENERATED MARKDOWN REPORT PREVIEW (First 2000 chars) ---")
    print(markdown_report[:2000])
    print("...\n")
    
    expected_sections = [
        "## 1. Incident Overview",
        "## 2. Executive Summary",
        "## 3. Incident Timeline",
        "## 4. Physical Observations & Telemetry",
        "## 5. Root Cause Analysis",
        "## 6. Contributing Factors",
        "## 7. Corrective & Preventive Actions",
        "## 8. Missing Evidence & Uncertainties",
        "## 9. Evidence & Citations",
    ]
    
    print("Verifying 9 sections in exact order:")
    pos = 0
    for sec in expected_sections:
        idx = markdown_report.find(sec, pos)
        if idx == -1:
            print(f"FAILED: Missing section: {sec}")
            sys.exit(1)
        print(f"  [OK] Found {sec} at char {idx}")
        pos = idx + len(sec)
        
    print("\nVerifying dynamic values from M-301 report:")
    assert "148" in markdown_report, "Missing 148 C winding temperature"
    assert "84 A" in markdown_report or "84" in markdown_report, "Missing Phase B current 84 A"
    assert "450" in markdown_report, "Missing insulation resistance 450 MΩ"
    assert "B-301" in markdown_report, "Missing bearing B-301"
    print("  [OK] All M-301 telemetry and physical observations verified.")
    
    print("\n" + "=" * 70)
    print("2. RUNNING HP-17 QUESTION MISMATCH VERIFICATION")
    print("=" * 70)
    
    # User asks about HP-17 while examining M-301 or against M-301 document evidence:
    hp17_query = "What caused high radial vibration and bearing wiping on steam turbine HP-17?"
    print(f"Query: {hp17_query}")
    investigate_resp = pipeline.investigate(InvestigationRequest(
        query=hp17_query,
        document_id_filter=m301_doc.document_id
    ))
    
    print("\nGenerated Investigation Response (with M-301 filter / evidence):")
    print(f"  is_sufficient_evidence: {investigate_resp.is_sufficient_evidence}")
    print(f"  executive_summary: {investigate_resp.executive_summary}")
    print(f"  missing_evidence_and_uncertainties: {investigate_resp.missing_evidence_and_uncertainties}")
    
    # Verify that equipment mismatch was detected and flagged
    assert investigate_resp.is_sufficient_evidence is False, "Evidence should be marked insufficient for mismatched equipment!"
    assert "HP-17" in investigate_resp.executive_summary, "Executive summary should mention target machine HP-17!"
    assert "does not establish" in investigate_resp.executive_summary or "mismatch" in investigate_resp.executive_summary.lower(), "Mismatch warning should be clearly explained!"
    print("\n  [OK] HP-17 mismatch detection and warning verified successfully!")
    print("\nAll verifications passed!")

if __name__ == "__main__":
    main()
