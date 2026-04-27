"""
Role synonym mappings used by scoring.py for ICP fit scoring.

TITLE_EQUIVALENTS: canonical title → all equivalent titles
FUNCTION_SYNONYMS: functional area → all synonym terms
"""

TITLE_EQUIVALENTS = {
    "ceo": ["ceo", "chief executive officer", "chief executive"],
    "cto": ["cto", "chief technology officer", "chief technical officer"],
    "cfo": ["cfo", "chief financial officer"],
    "coo": ["coo", "chief operating officer"],
    "cmo": ["cmo", "chief marketing officer"],
    "cro": ["cro", "chief revenue officer"],
    "cio": ["cio", "chief information officer"],
    "cdo": ["cdo", "chief data officer"],
    "cpo": ["cpo", "chief product officer"],
    "vp": ["vp", "vice president"],
    "svp": ["svp", "senior vice president"],
    "evp": ["evp", "executive vice president"],
    "avp": ["avp", "assistant vice president"],
    "director": ["director", "dir"],
    "head": ["head", "head of"],
    "manager": ["manager", "mgr"],
    "partner": ["partner", "managing partner"],
    "founder": ["founder", "co-founder", "cofounder"],
    "president": ["president", "pres"],
    "gm": ["gm", "general manager"],
    "md": ["md", "managing director"],
}

FUNCTION_SYNONYMS = {
    "sales": ["sales", "business development", "revenue", "commercial", "account executive", "account management"],
    "marketing": ["marketing", "brand", "demand generation", "growth", "digital marketing", "content marketing"],
    "engineering": ["engineering", "software engineering", "development", "r&d", "research and development"],
    "product": ["product", "product management", "product development"],
    "finance": ["finance", "financial", "accounting", "treasury", "fp&a"],
    "operations": ["operations", "ops", "supply chain", "logistics"],
    "hr": ["hr", "human resources", "people", "talent", "people operations"],
    "it": ["it", "information technology", "infrastructure", "devops", "sre"],
    "legal": ["legal", "compliance", "regulatory", "general counsel"],
    "security": ["security", "cybersecurity", "infosec", "information security"],
    "data": ["data", "analytics", "data science", "business intelligence", "bi"],
    "design": ["design", "ux", "ui", "user experience", "creative"],
    "customer success": ["customer success", "client success", "customer experience", "cx"],
    "support": ["support", "customer support", "technical support", "service"],
    "pr": ["pr", "public relations", "communications", "corporate communications", "media relations"],
    "partnerships": ["partnerships", "alliances", "channel", "strategic partnerships", "ecosystem"],
    "procurement": ["procurement", "purchasing", "sourcing", "vendor management"],
}
