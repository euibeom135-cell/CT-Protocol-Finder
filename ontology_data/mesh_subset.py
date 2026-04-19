"""Curated MeSH condition mapping for CP/FIH-relevant therapeutic areas.

Maps free-text condition strings to MeSH tree codes, and computes
tree-distance similarity between conditions using lowest common ancestor.

MeSH tree structure (simplified):
  C01 - Infections
  C04 - Neoplasms
  C05 - Musculoskeletal Diseases
  C06 - Digestive System Diseases
  C07 - Stomatognathic Diseases
  C08 - Respiratory Tract Diseases
  C10 - Nervous System Diseases
  C14 - Cardiovascular Diseases
  C17 - Skin and Connective Tissue Diseases
  C18 - Nutritional and Metabolic Diseases
  C19 - Endocrine System Diseases
  C20 - Immune System Diseases
  F03 - Mental Disorders
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Sentinel for healthy volunteers (maximum distance from all disease codes)
# ---------------------------------------------------------------------------
HEALTHY_VOLUNTEER = "HEALTHY"

# ---------------------------------------------------------------------------
# Condition -> MeSH tree code mapping
# Covers top conditions found in CP/FIH Phase 1 trial indices
# ---------------------------------------------------------------------------
MESH_TREE: dict[str, str] = {
    # -- Healthy volunteers --
    "healthy": HEALTHY_VOLUNTEER,
    "healthy volunteers": HEALTHY_VOLUNTEER,
    "healthy subjects": HEALTHY_VOLUNTEER,
    "healthy adults": HEALTHY_VOLUNTEER,
    "healthy participants": HEALTHY_VOLUNTEER,
    "healthy male": HEALTHY_VOLUNTEER,
    "healthy female": HEALTHY_VOLUNTEER,
    "normal volunteers": HEALTHY_VOLUNTEER,

    # -- C01: Infections --
    "hiv": "C01.778.640.400",
    "hiv infection": "C01.778.640.400",
    "hiv infections": "C01.778.640.400",
    "hiv-1": "C01.778.640.400",
    "hepatitis b": "C01.925.440.435",
    "hepatitis c": "C01.925.440.440",
    "chronic hepatitis b": "C01.925.440.435",
    "chronic hepatitis c": "C01.925.440.440",
    "tuberculosis": "C01.150.252.410.890",
    "influenza": "C01.925.782.580.600.500",
    "covid-19": "C01.925.782.600.550.200.163",
    "sars-cov-2": "C01.925.782.600.550.200.163",
    "bacterial infection": "C01.150",
    "fungal infection": "C01.150.703",
    "malaria": "C01.610.752.530",
    "sepsis": "C01.539.757",
    "pneumonia": "C01.150.252.410.700",
    "urinary tract infection": "C01.150.252.410.868",
    "clostridium difficile": "C01.150.252.410.222",
    "respiratory syncytial virus": "C01.925.782.580.600.680",

    # -- C04: Neoplasms --
    "cancer": "C04",
    "neoplasm": "C04",
    "solid tumor": "C04.588",
    "solid tumors": "C04.588",
    "solid tumour": "C04.588",
    "advanced solid tumor": "C04.588",
    "advanced solid tumors": "C04.588",
    "metastatic solid tumor": "C04.588",
    "breast cancer": "C04.588.180",
    "breast neoplasm": "C04.588.180",
    "triple negative breast cancer": "C04.588.180",
    "non-small cell lung cancer": "C04.588.894.797.520",
    "nsclc": "C04.588.894.797.520",
    "lung cancer": "C04.588.894.797",
    "small cell lung cancer": "C04.588.894.797.640",
    "colorectal cancer": "C04.588.274.476.205",
    "colon cancer": "C04.588.274.476.205",
    "pancreatic cancer": "C04.588.274.761",
    "gastric cancer": "C04.588.274.843",
    "stomach cancer": "C04.588.274.843",
    "hepatocellular carcinoma": "C04.588.274.623.160",
    "liver cancer": "C04.588.274.623",
    "prostate cancer": "C04.588.945.440.770",
    "ovarian cancer": "C04.588.945.418.685",
    "endometrial cancer": "C04.588.945.418.948",
    "renal cell carcinoma": "C04.588.945.947.535",
    "kidney cancer": "C04.588.945.947",
    "bladder cancer": "C04.588.945.947.100",
    "melanoma": "C04.557.465.625.650.510",
    "glioblastoma": "C04.557.470.670.380",
    "brain tumor": "C04.557.470",
    "head and neck cancer": "C04.588.443",
    "thyroid cancer": "C04.588.322.894",
    "mesothelioma": "C04.557.470.035.510",
    "sarcoma": "C04.557.450",
    "osteosarcoma": "C04.557.450.565.575",
    "acute myeloid leukemia": "C04.557.337.539.275",
    "aml": "C04.557.337.539.275",
    "acute lymphoblastic leukemia": "C04.557.337.539.250",
    "all": "C04.557.337.539.250",
    "chronic lymphocytic leukemia": "C04.557.337.539.375",
    "cll": "C04.557.337.539.375",
    "chronic myeloid leukemia": "C04.557.337.539.400",
    "cml": "C04.557.337.539.400",
    "leukemia": "C04.557.337",
    "lymphoma": "C04.557.386",
    "non-hodgkin lymphoma": "C04.557.386.480",
    "diffuse large b-cell lymphoma": "C04.557.386.480.350",
    "dlbcl": "C04.557.386.480.350",
    "follicular lymphoma": "C04.557.386.480.425",
    "hodgkin lymphoma": "C04.557.386.355",
    "multiple myeloma": "C04.557.595.500",
    "myelodysplastic syndrome": "C04.557.337.555",
    "myelofibrosis": "C04.557.337.600",

    # -- C05: Musculoskeletal --
    "rheumatoid arthritis": "C05.550.114.865",
    "osteoarthritis": "C05.550.114.606",
    "osteoporosis": "C05.116.198.579",
    "gout": "C05.550.354",
    "ankylosing spondylitis": "C05.550.114.845",
    "psoriatic arthritis": "C05.550.114.154",
    "lupus": "C05.550.114.154.720",

    # -- C06: Digestive --
    "crohn's disease": "C06.405.205.265",
    "crohn disease": "C06.405.205.265",
    "ulcerative colitis": "C06.405.205.731",
    "inflammatory bowel disease": "C06.405.205",
    "ibd": "C06.405.205",
    "nonalcoholic steatohepatitis": "C06.552.241.500",
    "nash": "C06.552.241.500",
    "nonalcoholic fatty liver disease": "C06.552.241",
    "nafld": "C06.552.241",
    "cirrhosis": "C06.552.630",
    "liver fibrosis": "C06.552.630",
    "celiac disease": "C06.405.469.163",
    "irritable bowel syndrome": "C06.405.469.452",
    "pancreatitis": "C06.689.587",

    # -- C08: Respiratory --
    "asthma": "C08.127.108",
    "copd": "C08.381.495.389",
    "chronic obstructive pulmonary disease": "C08.381.495.389",
    "idiopathic pulmonary fibrosis": "C08.381.765.500",
    "ipf": "C08.381.765.500",
    "cystic fibrosis": "C08.381.187",
    "pulmonary arterial hypertension": "C08.381.423.500",
    "pah": "C08.381.423.500",

    # -- C10: Nervous System --
    "alzheimer's disease": "C10.228.140.380.100",
    "alzheimer disease": "C10.228.140.380.100",
    "parkinson's disease": "C10.228.140.380.614",
    "parkinson disease": "C10.228.140.380.614",
    "multiple sclerosis": "C10.114.375.500",
    "epilepsy": "C10.228.140.490",
    "migraine": "C10.228.140.546.399.750",
    "neuropathic pain": "C10.668.829.600",
    "pain": "C10.668.829",
    "stroke": "C10.228.140.300.775",
    "amyotrophic lateral sclerosis": "C10.228.854.139",
    "als": "C10.228.854.139",
    "huntington disease": "C10.228.140.380.410",
    "schizophrenia": "F03.700.750",
    "depression": "F03.600.300.375",
    "major depressive disorder": "F03.600.300.375",
    "bipolar disorder": "F03.600.150",
    "anxiety": "F03.080",
    "adhd": "F03.625.094.150",
    "insomnia": "C10.886.425.800.400",

    # -- C14: Cardiovascular --
    "heart failure": "C14.280.434",
    "atrial fibrillation": "C14.280.067.198",
    "hypertension": "C14.907.489",
    "pulmonary hypertension": "C14.907.489.750",
    "coronary artery disease": "C14.280.647.250",
    "atherosclerosis": "C14.907.137.126.307",
    "deep vein thrombosis": "C14.907.355.830.925",
    "venous thromboembolism": "C14.907.355.830",
    "pulmonary embolism": "C14.907.355.830.600",
    "peripheral artery disease": "C14.907.137.126.612",
    "myocardial infarction": "C14.280.647.500",
    "dyslipidemia": "C14.907.137.126.339",
    "hypercholesterolemia": "C14.907.137.126.339",

    # -- C17: Skin --
    "psoriasis": "C17.800.859.675",
    "atopic dermatitis": "C17.800.174.255",
    "eczema": "C17.800.174",
    "urticaria": "C17.800.862.945",
    "alopecia": "C17.800.329.050",
    "vitiligo": "C17.800.621.902",
    "systemic sclerosis": "C17.300.799",
    "scleroderma": "C17.300.799",

    # -- C18: Metabolic --
    "diabetes": "C18.452.394.750",
    "type 2 diabetes": "C18.452.394.750.149",
    "type 1 diabetes": "C18.452.394.750.124",
    "diabetes mellitus": "C18.452.394.750",
    "obesity": "C18.654.726.500",
    "metabolic syndrome": "C18.452.394.968",
    "hyperuricemia": "C18.452.648.398",

    # -- C19: Endocrine --
    "hypothyroidism": "C19.874.397",
    "hyperthyroidism": "C19.874.397.500",
    "cushing syndrome": "C19.053.347",
    "acromegaly": "C19.700.355.100",
    "growth hormone deficiency": "C19.700.355.528",

    # -- C20: Immune --
    "systemic lupus erythematosus": "C20.111.590",
    "sle": "C20.111.590",
    "sjogren syndrome": "C20.111.850",
    "graft versus host disease": "C20.452.399",
    "gvhd": "C20.452.399",
    "transplant rejection": "C20.452.300",
    "autoimmune disease": "C20.111",
    "allergy": "C20.543",
    "anaphylaxis": "C20.543.480.099",

    # -- Hematologic --
    "anemia": "C15.378.071",
    "sickle cell disease": "C15.378.071.141.150.875",
    "hemophilia": "C15.378.100.425",
    "thrombocytopenia": "C15.378.140.855",
    "iron deficiency anemia": "C15.378.071.085",

    # -- Other common in CP trials --
    "chronic kidney disease": "C12.777.419.780",
    "ckd": "C12.777.419.780",
    "end stage renal disease": "C12.777.419.780.500",
    "organ transplant": "E04.936",
    "kidney transplant": "E04.936.580",
}


# ---------------------------------------------------------------------------
# Tree hierarchy for LCA (Lowest Common Ancestor) computation
# Maps each tree code to its list of ancestors (root → self)
# ---------------------------------------------------------------------------
def _build_ancestors(code: str) -> list[str]:
    """Build ancestor list from a MeSH tree code."""
    parts = code.split(".")
    ancestors = []
    for i in range(len(parts)):
        ancestors.append(".".join(parts[: i + 1]))
    return ancestors


def _lowest_common_ancestor_depth(code_a: str, code_b: str) -> int:
    """Find depth of lowest common ancestor of two MeSH tree codes.

    Returns 0 if codes are in completely different branches.
    """
    ancestors_a = set(_build_ancestors(code_a))
    ancestors_b = _build_ancestors(code_b)

    lca_depth = 0
    for anc in ancestors_b:
        if anc in ancestors_a:
            depth = anc.count(".") + 1
            if depth > lca_depth:
                lca_depth = depth
    return lca_depth


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def condition_to_mesh(condition_text: str) -> str | None:
    """Map a free-text condition to its MeSH tree code.

    Tries exact match first, then substring matching against known conditions.

    Returns:
        MeSH tree code string, or None if no match found.
    """
    if not condition_text:
        return None

    text = condition_text.lower().strip()

    # Exact match
    if text in MESH_TREE:
        return MESH_TREE[text]

    # Substring match: find the longest matching key
    best_match = None
    best_len = 0
    for key, code in MESH_TREE.items():
        if key in text and len(key) > best_len:
            best_match = code
            best_len = len(key)

    if best_match:
        return best_match

    # Reverse substring: check if condition is contained in any key
    for key, code in MESH_TREE.items():
        if text in key:
            return code

    return None


def mesh_tree_distance(code_a: str, code_b: str) -> float:
    """Compute normalized tree distance between two MeSH codes.

    Returns:
        0.0 if codes are identical,
        1.0 if codes are in completely unrelated branches,
        Value in between based on LCA depth.
    """
    if code_a == code_b:
        return 0.0

    # Healthy volunteer sentinel — max distance from all diseases
    if code_a == HEALTHY_VOLUNTEER or code_b == HEALTHY_VOLUNTEER:
        if code_a == code_b:
            return 0.0
        return 1.0

    # Compute LCA depth
    max_depth = max(code_a.count(".") + 1, code_b.count(".") + 1)
    if max_depth == 0:
        return 1.0

    lca_depth = _lowest_common_ancestor_depth(code_a, code_b)
    if lca_depth == 0:
        return 1.0

    # Distance: 1 - (LCA_depth / max_depth)
    # Deeper LCA = closer relationship = lower distance
    return 1.0 - (lca_depth / max_depth)


def condition_set_similarity(conditions_a: list[str], conditions_b: list[str]) -> tuple[float, str]:
    """Compute similarity between two sets of conditions using MeSH tree distance.

    For each condition in set A, finds the closest match in set B (min distance).
    Averages across all conditions. Symmetrizes by averaging both directions.

    Returns:
        (similarity_score, detail_string)
        similarity_score: 0.0 to 1.0 where 1.0 = identical conditions
        detail_string: human-readable explanation (e.g., "Both oncology (breast)")
    """
    if not conditions_a and not conditions_b:
        return 0.5, "No conditions specified"
    if not conditions_a or not conditions_b:
        return 0.0, "Missing conditions"

    # Map conditions to MeSH codes
    codes_a = [(c, condition_to_mesh(c)) for c in conditions_a]
    codes_b = [(c, condition_to_mesh(c)) for c in conditions_b]

    # Filter to only mapped conditions
    mapped_a = [(c, code) for c, code in codes_a if code is not None]
    mapped_b = [(c, code) for c, code in codes_b if code is not None]

    if not mapped_a and not mapped_b:
        return 0.5, "Conditions not in MeSH subset"
    if not mapped_a or not mapped_b:
        return 0.3, "Partial MeSH mapping"

    # Compute directional distance: A -> B
    def _directional_distance(from_set, to_set):
        total_dist = 0.0
        for _, code_from in from_set:
            min_dist = 1.0
            for _, code_to in to_set:
                dist = mesh_tree_distance(code_from, code_to)
                min_dist = min(min_dist, dist)
            total_dist += min_dist
        return total_dist / len(from_set)

    dist_ab = _directional_distance(mapped_a, mapped_b)
    dist_ba = _directional_distance(mapped_b, mapped_a)
    avg_dist = (dist_ab + dist_ba) / 2.0
    similarity = 1.0 - avg_dist

    # Build detail string
    detail = _build_condition_detail(mapped_a, mapped_b)

    return similarity, detail


def _build_condition_detail(mapped_a: list, mapped_b: list) -> str:
    """Build human-readable condition similarity detail."""
    codes_a = {code for _, code in mapped_a}
    codes_b = {code for _, code in mapped_b}

    # Check for exact overlap
    overlap = codes_a & codes_b
    if overlap:
        # Find condition names for overlapping codes
        names = [c for c, code in mapped_a if code in overlap][:2]
        return f"Shared: {', '.join(names)}"

    # Check for same top-level branch
    branches_a = {code.split(".")[0] for code in codes_a if code != HEALTHY_VOLUNTEER}
    branches_b = {code.split(".")[0] for code in codes_b if code != HEALTHY_VOLUNTEER}
    shared_branches = branches_a & branches_b

    branch_names = {
        "C01": "Infections", "C04": "Oncology", "C05": "Musculoskeletal",
        "C06": "GI", "C08": "Respiratory", "C10": "Neurology",
        "C14": "Cardiovascular", "C17": "Dermatology", "C18": "Metabolic",
        "C19": "Endocrine", "C20": "Immunology", "F03": "Psychiatry",
        "C12": "Renal", "C15": "Hematology",
    }

    if shared_branches:
        names = [branch_names.get(b, b) for b in shared_branches]
        return f"Both {', '.join(names[:2])}"

    if HEALTHY_VOLUNTEER in codes_a and HEALTHY_VOLUNTEER in codes_b:
        return "Both healthy volunteers"

    name_a = mapped_a[0][0] if mapped_a else "?"
    name_b = mapped_b[0][0] if mapped_b else "?"
    return f"{name_a} vs {name_b}"
