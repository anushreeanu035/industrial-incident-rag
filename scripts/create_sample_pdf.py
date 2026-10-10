"""Utility script to generate a multi-page sample industrial safety bulletin PDF."""

from pathlib import Path
import pymupdf


def create_sample_pdf(output_path: Path):
    doc = pymupdf.open()

    # Page 1: Incident Description & Background
    page1 = doc.new_page()
    text_page_1 = """SAFETY INVESTIGATION BULLETIN: SIB-2024-500
Machine ID: TR-200
Incident Type: Electrical Transformer Arc Flash & Explosion
Facility: High-Voltage Substation Yard B, Transformer Bay 3
Incident Date: 2024-06-12
Classification: High Potential Catastrophic Near-Miss

1. INCIDENT OVERVIEW
On June 12, 2024, at 11:23 AM, the 50 MVA 132/33 kV step-down power transformer TR-200 suffered a catastrophic internal arc flash and tank rupture. The pressure relief device PRD-200 operated, but the internal pressure wave bowed the main tank sidewalls and ignited an insulating oil pool fire. The automated Deluge water spray system extinguished the flames in 8 minutes. No personnel were inside the blast radius.

2. TIMELINE OF SUBSTATION METRICS
- 11:05 AM: Dissolved gas analysis (DGA) monitor reported hydrogen (H2) concentration climb to 1,200 ppm and acetylene (C2H2) at 45 ppm.
- 11:18 AM: Buchholz relay stage 1 warning alarm activated on control room annunciator panel.
- 11:23 AM: High-energy internal arc occurred; differential protection relay 87T executed breaker trips on primary circuit breaker CB-101 and secondary breaker CB-201.
"""
    rect1 = pymupdf.Rect(50, 50, 550, 750)
    page1.insert_textbox(rect1, text_page_1, fontsize=11, fontname="helv")

    # Page 2: Root Causes & Corrective Actions
    page2 = doc.new_page()
    text_page_2 = """SAFETY INVESTIGATION BULLETIN: SIB-2024-500 (Page 2)
Machine ID: TR-200
Incident Type: Electrical Transformer Arc Flash & Explosion

3. PHYSICAL INVESTIGATION FINDINGS
Teardown inspection of transformer TR-200 at the high-voltage repair facility revealed:
- High-voltage Bushing B-Phase had severe electrical tracking across its lower oil-immersed porcelain skirt.
- Moisture ingress had penetrated the top bushing capacitance tap cap because of a hardened and cracked nitrile gasket.
- Dielectric breakdown of the paper-oil insulation barrier occurred adjacent to the B-phase tap lead.
- The Buchholz relay gas accumulation was ignored by shift operators as a "nuisance alarm" because previous maintenance notes reported false alarms.

4. ROOT CAUSE SUMMARY
- Primary Root Cause: Moisture penetration through a degraded capacitance tap gasket on Bushing B-Phase resulted in dielectric breakdown, initiating an intense phase-to-ground internal arc flash.
- Organizational Root Cause: Inadequate procedural response to Buchholz relay warnings; operators failed to immediately de-energize the unit upon gas alarm confirmation.

5. MANDATORY CORRECTIVE MEASURES
1. Retrofit all substation power transformers with online capacitance and dielectric dissipation factor (Tan Delta) bushing monitors.
2. Replace all legacy nitrile bushing tap gaskets with fluorosilicone O-rings during upcoming outages.
3. Update standard emergency procedure SOP-HV-09 to mandate immediate tripping of transformers upon unverified Buchholz gas alarms.
4. Conduct refresher training for all substation operators regarding Dissolved Gas Analysis interpretation.
"""
    rect2 = pymupdf.Rect(50, 50, 550, 750)
    page2.insert_textbox(rect2, text_page_2, fontsize=11, fontname="helv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    print(f"Sample PDF created successfully at: {output_path}")


if __name__ == "__main__":
    pdf_file = Path("data/raw/INC-2024-500_transformer_explosion_safety_bulletin.pdf")
    create_sample_pdf(pdf_file)
