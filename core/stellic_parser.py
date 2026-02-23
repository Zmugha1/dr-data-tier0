"""
Stellic Degree Audit PDF Parser for UW-Stout CTE Advising.
Extracts student info, credits, planned courses, Stout Core, RES/GLP, major requirements.
"""

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


# Stout Core category patterns
STOUT_CORE_CATEGORIES = [
    "ARNS", "ARHU", "SBSC", "MATH", "NSLAB", "GLP", "RES",
    "Stout Core", "Communications", "ENGL", "COMM",
]
PLACEHOLDER_PATTERNS = [
    r"Natural Science with Lab",
    r"Natural Science",
    r"ARHU",
    r"SBSC",
    r"SRER",
    r"RES\s*(\d)?",
    r"GLP\s*(\d)?",
    r"ENGL-102",
    r"Math/Statistics",
    r"ARNS",
]
COURSE_PATTERN = re.compile(
    r"([A-Z]{2,4})-(\d{3})(?:\s+([^|]+))?(?:\s*\|\s*(\w+\s*'?\d{2}))?",
    re.IGNORECASE
)
TERM_PATTERN = re.compile(
    r"(Fall|Spring|Summer)\s*'?(\d{2,4})",
    re.IGNORECASE
)
CREDIT_PATTERN = re.compile(
    r"(\d+)\s*(?:credit|credits|cr\.?)",
    re.IGNORECASE
)


def extract_text_from_pdf(source):  # bytes | Path | str
    """Extract raw text and table data from a Stellic degree audit PDF."""
    if pdfplumber is None:
        raise ImportError("pdfplumber required. pip install pdfplumber")

    if isinstance(source, (Path, str)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        with open(path, "rb") as f:
            content = f.read()
    else:
        content = source

    text_parts = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
            # Also extract tables (Stellic uses tables for course lists)
            for table in page.extract_tables() or []:
                if table:
                    text_parts.append("\n".join(" | ".join(cell or "" for cell in row) for row in table))
    return "\n\n".join(text_parts) if text_parts else ""


def parse_stellic_audit(source):  # bytes | Path | str
    """
    Parse a Stellic degree audit PDF and return structured data.
    """
    text = extract_text_from_pdf(source)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    result: Dict[str, Any] = {
        "student_name": "",
        "catalog_term": "",
        "total_credits_earned": 0,
        "planned_courses": [],
        "unmet_placeholders": [],
        "res_status": {"count": 0, "courses": [], "planned": []},
        "glp_status": {"count": 0, "courses": [], "planned": []},
        "engl102_status": "unknown",
        "major_plans": [],
        "internship_status": "unknown",
        "capstone_status": "unknown",
        "stout_core_unmet": [],
        "arns_status": {"math_stat": False, "nslab": False},
        "raw_text": text,
        "risk_flags_2028_2029": [],
    }

    # Student name - often in "Student: Name" or "Degree Audit for Name" or filename
    for line in lines[:50]:
        if re.search(r"student\s*[:=]", line, re.I):
            result["student_name"] = re.sub(r"student\s*[:=]\s*", "", line, flags=re.I).strip()
            break
        if "Degree Audit" in line or "degree audit" in line:
            # "University of Wisconsin-Stout - Degree Audit _ James Frye _ ..."
            parts = re.split(r"\s+_\s+", line)
            if len(parts) >= 2:
                result["student_name"] = parts[1].strip()
                break

    if not result["student_name"] and isinstance(source, (Path, str)):
        fn = Path(source).stem
        if "_" in fn or " - " in fn:
            parts = re.split(r"\s*[_-]\s*", fn)
            for p in parts:
                if "James" in p or "Frye" in p or (len(p) > 3 and p.replace(" ", "").isalpha()):
                    result["student_name"] = p.strip()
                    break

    if not result["student_name"]:
        result["student_name"] = "Student"

    # Catalog term
    for line in lines[:80]:
        if re.search(r"catalog\s*term|term\s*[:=]|effective\s*term", line, re.I):
            result["catalog_term"] = line
            break
        m = TERM_PATTERN.search(line)
        if m and not result["catalog_term"]:
            result["catalog_term"] = f"{m.group(1)} {m.group(2)}"

    # Total credits earned
    for line in lines:
        if re.search(r"total\s+credit|credit\s+earned|credits\s+earned", line, re.I):
            m = CREDIT_PATTERN.search(line)
            if m:
                result["total_credits_earned"] = int(m.group(1))
                break

    if result["total_credits_earned"] == 0:
        m = re.search(r"(\d+)\s*/\s*120", text)
        if m:
            result["total_credits_earned"] = int(m.group(1))

    # Planned courses and terms
    current_term = ""
    for i, line in enumerate(lines):
        tm = TERM_PATTERN.search(line)
        if tm:
            current_term = f"{tm.group(1)} '{tm.group(2)[-2:]}"
        cm = COURSE_PATTERN.search(line)
        if cm and current_term:
            result["planned_courses"].append({
                "course": f"{cm.group(1)}-{cm.group(2)}",
                "term": current_term,
                "raw": line,
            })

    # ENGL-102
    if re.search(r"ENGL-?102|engl\s*102", text, re.I):
        if re.search(r"ENGL-?102.*(?:satisfied|complete|done|✓)", text, re.I):
            result["engl102_status"] = "satisfied"
        elif re.search(r"Spring\s*'?29|Spring\s*2029", text, re.I) and re.search(r"ENGL-?102", text, re.I):
            result["engl102_status"] = "planned_spring_29"
            result["risk_flags_2028_2029"].append("ENGL-102 planned Spring '29 - consider moving earlier")
        else:
            result["engl102_status"] = "unmet"

    # RES and GLP
    res_matches = re.findall(r"RES\s*(?:requirement)?\s*(\d)?|(?:RES\s*)([A-Z]{2,4}-\d{3})", text, re.I)
    glp_matches = re.findall(r"GLP\s*(?:requirement)?\s*(\d)?|(?:GLP\s*)([A-Z]{2,4}-\d{3})", text, re.I)
    result["res_status"]["count"] = min(2, len([m for m in res_matches if any(m)]))
    result["glp_status"]["count"] = min(2, len([m for m in glp_matches if any(m)]))

    # Unmet placeholders
    for pat in ["Natural Science with Lab", "Natural Science", "ARHU", "SBSC", "SRER", "ARNS", "Math/Statistics"]:
        if re.search(pat, text, re.I) and re.search(r"unmet|remaining|planned|placeholder", text, re.I):
            result["unmet_placeholders"].append(pat)
    result["unmet_placeholders"] = list(dict.fromkeys(result["unmet_placeholders"]))

    # Risk flags from user spec
    if re.search(r"Natural Science with Lab.*Fall\s*'?28|Fall\s*'?28.*Natural Science", text, re.I):
        result["risk_flags_2028_2029"].append("Natural Science with Lab planned Fall '28")
    if re.search(r"SRER.*Spring\s*'?29|Spring\s*'?29.*SRER", text, re.I):
        result["risk_flags_2028_2029"].append("SRER planned Spring '29")
    if not result["risk_flags_2028_2029"] and result["unmet_placeholders"]:
        result["risk_flags_2028_2029"] = [
            f"Placeholder in 2028-2029: {p}" for p in result["unmet_placeholders"][:4]
        ]

    return result
