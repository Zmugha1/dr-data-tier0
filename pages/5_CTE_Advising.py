"""
CTE Program Lead Advising Dashboard
Parses Stellic degree audit PDFs and generates structured advising notes per official rules.
"""

from pathlib import Path

import streamlit as st

from core.stellic_parser import parse_stellic_audit

# PDF path from user - support /mnt/data (Linux/WSL/Cursor), project root, and local alternatives
_AUDIT_FILENAME = "University of Wisconsin-Stout - Degree Audit _ James Frye _ University of Wisconsin-Stout.pdf"
PDF_PATHS = [
    Path("/mnt/data") / _AUDIT_FILENAME,
    Path("mnt/data") / _AUDIT_FILENAME,
    Path("data/degree_audits") / _AUDIT_FILENAME,
    Path(_AUDIT_FILENAME),
    Path.home() / "Documents" / _AUDIT_FILENAME,
    Path.cwd() / _AUDIT_FILENAME,
    Path.cwd() / "data" / "degree_audits" / _AUDIT_FILENAME,
]
Path("data/degree_audits").mkdir(parents=True, exist_ok=True)  # ensure dir exists for uploads

st.set_page_config(page_title="CTE Advising Output", layout="wide")

st.title("🎓 CTE Program Lead Advising Dashboard")
st.markdown("""
**What this page does**: Parse a Stellic degree audit PDF and generate Program Lead advising notes.  
Applies official CTE rules: 120-credit minimum, Stout Core logic, RES/GLP, double-counting, CTET rotation.
""")

# Load PDF
pdf_content = None
pdf_path_used = None

uploaded = st.file_uploader("Upload Stellic Degree Audit PDF", type=["pdf"], help="Or use pre-loaded audit below.")

if uploaded:
    pdf_content = uploaded.read()
    st.success(f"Loaded: {uploaded.name}")
else:
    for p in PDF_PATHS:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    pdf_content = f.read()
                pdf_path_used = str(p)
                st.success(f"Loaded pre-uploaded audit: `{p.name}`")
                break
            except Exception as e:
                continue
    if not pdf_content:
        st.info("Upload a degree audit PDF above, or place it at `data/degree_audits/` with the expected filename.")

if pdf_content:
    try:
        audit = parse_stellic_audit(pdf_content)
    except Exception as e:
        st.error(f"Parse error: {e}")
        st.stop()

    # Apply CTE rules and build output
    creds = audit["total_credits_earned"]
    remaining = max(0, 120 - creds)
    name = audit["student_name"]
    res = audit["res_status"]["count"]
    glp = audit["glp_status"]["count"]
    engl = audit["engl102_status"]
    risks = audit["risk_flags_2028_2029"]
    placeholders = audit["unmet_placeholders"]

    # POWER MOVE identification
    power_moves = []
    if "RES" in str(placeholders) and "SBSC" in str(placeholders):
        power_moves.append("RES course that also satisfies SBSC")
    if "GLP" in str(placeholders) and "ARHU" in str(placeholders):
        power_moves.append("GLP course that also satisfies ARHU")
    if not power_moves:
        power_moves.append("Seek RES/GLP courses that double-count with ARHU or SBSC")

    # Summer plan (next logical term)
    summer_courses = []
    if engl == "unmet" or engl == "planned_spring_29":
        summer_courses.append(("ENGL-102", "Cannot delay ENGL-102; foundational for Stout Core.", "Communications"))
    if res < 2:
        summer_courses.append(("RES option", "One of two RES required. Prefer RES+SBSC double-count.", "RES"))
    if glp < 2:
        summer_courses.append(("GLP option", "One of two GLP required. Prefer GLP+ARHU double-count.", "GLP"))
    if not summer_courses:
        summer_courses.append(("Major/elective", "Core milestones met; advance major or reduce Fall load.", "Major"))

    # Fall plan
    fall_courses = []
    for ph in placeholders[:3]:
        if "Natural Science" in ph or "NSLAB" in ph:
            fall_courses.append(("Natural Science with Lab", "ARNS requires 1 math/stat + 1 NSLAB.", "ARNS/NSLAB"))
        elif "ARHU" in ph or "SBSC" in ph:
            fall_courses.append((f"{ph} course", f"Stout Core placeholder. Seek double-count with GLP/RES if possible.", ph))
        elif "SRER" in ph:
            fall_courses.append(("SRER option", "Senior Research. Plan for Spring if not yet scheduled.", "SRER"))

    # Build output
    output = f"""
--------------------------------------------------
ADVISING NOTES — {name}
--------------------------------------------------

**Progress Summary**
- Credits earned: **{creds}** / 120 (remaining: {remaining})
- RES: {res}/2 | GLP: {glp}/2 | ENGL-102: {engl}
- Unmet Stout Core placeholders: {", ".join(placeholders) if placeholders else "None identified"}
- Major progression: Review planned courses vs. CTET matrix

**Strategic Priorities**
1) **Reduce Stout Core risk** — Address placeholders in 2028-2029 before senior year
2) **Maximize double-counting** — RES/GLP + category whenever possible
3) **Maintain manageable load** — 15-16 cr standard; 12-14 if balancing work/internship
4) **Align with CTET matrix** — Confirm course availability per rotation

---

**SUMMER PLAN (next logical term)**
"""
    for course, why, req in summer_courses:
        output += f"\n- **{course}** — {why} *(Satisfies: {req})*"

    output += "\n\n---\n\n**FALL PLAN**\n"
    if fall_courses:
        for course, why, req in fall_courses:
            output += f"\n- **{course}** — {why} *(Satisfies: {req})*"
        output += "\n\n*Double-count impact: Prioritize courses that satisfy RES/GLP and a Stout Core category. Cannot double-count a course for both major AND Stout Core.*"
    else:
        output += "\n- Advance major requirements and/or remaining electives."
        output += "\n- Confirm no additional Stout Core gaps from catalog."

    output += "\n\n---\n\n**Two Strategic Pathways**\n\n"
    output += "**PATH A – Fast Track Core Completion**\n"
    output += "Take 2–3 Stout Core placeholders in Summer + Fall to clear risk by end of Fall '28. Heavier load short-term, lighter senior year.\n\n"
    output += "**PATH B – Lighter Summer**\n"
    output += "1–2 courses in Summer (e.g., ENGL-102 if needed). Spread remaining Core across Fall '28 and Spring '29. More balanced load.\n\n"

    if power_moves:
        output += "**POWER MOVE** courses (close multiple gaps):\n"
        for pm in power_moves:
            output += f"- {pm}\n"

    output += "\n---\n\n**Risk Flags**\n"
    for r in risks:
        output += f"- ⚠️ {r}\n"
    if not risks:
        output += "- No specific 2028-2029 placeholder risks identified from audit.\n"

    output += "\n**Field Experience Planning**\n"
    output += "- Confirm internship/capstone sequence per major requirements.\n"
    output += "- CTE-360 Summer availability remains; apply academic year rules for sequencing.\n\n"

    output += "**Substitution/Waiver Notes**\n"
    output += "- None indicated in audit. Document any approved substitutions in student record.\n\n"

    output += "**Graduation Outlook**\n"
    pct = round(100 * creds / 120, 1) if creds else 0
    output += f"- {pct}% credit progress. "
    if remaining <= 30:
        output += "On track for timely graduation with planned courses."
    elif remaining <= 60:
        output += "Monitor load; ensure 15-16 cr/semester to stay on pace."
    else:
        output += "Early-mid progression; prioritize Core and major sequencing."

    output += "\n\n--------------------------------------------------"

    # Display
    st.subheader("Advising Output")
    st.markdown(output)

    st.download_button(
        label="📥 Download Advising Notes",
        data=output,
        file_name=f"CTE_Advising_{name.replace(' ', '_')}.txt",
        mime="text/plain",
    )

    # Raw audit summary (collapsible)
    with st.expander("Raw parsed audit data"):
        st.json({k: v for k, v in audit.items() if k != "raw_text"})
else:
    st.markdown("---")
    st.caption("Upload a Stellic degree audit PDF to generate advising notes.")
