"""Drug-to-Mechanism Class Mapping for Clinical Trial Similarity.

Maps drug names to their mechanism of action and broader pharmacological class.
Two-level hierarchy enables nuanced similarity scoring:
  - Same mechanism (e.g., both PD-1 inhibitors) → distance 0.0
  - Same broader class (e.g., PD-1 vs CTLA-4, both immune checkpoint) → distance 0.3
  - Different class → distance 1.0

Covers ~150 commonly encountered drugs in CP/FIH Phase 1 trials plus
common PK probe substrates/inhibitors/inducers used in DDI studies.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Drug -> (specific_mechanism, broader_class)
# ---------------------------------------------------------------------------
DRUG_TO_CLASS: dict[str, tuple[str, str]] = {
    # === Immune Checkpoint Inhibitors ===
    "pembrolizumab": ("PD-1 inhibitor", "immune checkpoint inhibitor"),
    "keytruda": ("PD-1 inhibitor", "immune checkpoint inhibitor"),
    "nivolumab": ("PD-1 inhibitor", "immune checkpoint inhibitor"),
    "opdivo": ("PD-1 inhibitor", "immune checkpoint inhibitor"),
    "cemiplimab": ("PD-1 inhibitor", "immune checkpoint inhibitor"),
    "dostarlimab": ("PD-1 inhibitor", "immune checkpoint inhibitor"),
    "tislelizumab": ("PD-1 inhibitor", "immune checkpoint inhibitor"),
    "toripalimab": ("PD-1 inhibitor", "immune checkpoint inhibitor"),
    "sintilimab": ("PD-1 inhibitor", "immune checkpoint inhibitor"),
    "camrelizumab": ("PD-1 inhibitor", "immune checkpoint inhibitor"),
    "atezolizumab": ("PD-L1 inhibitor", "immune checkpoint inhibitor"),
    "tecentriq": ("PD-L1 inhibitor", "immune checkpoint inhibitor"),
    "durvalumab": ("PD-L1 inhibitor", "immune checkpoint inhibitor"),
    "imfinzi": ("PD-L1 inhibitor", "immune checkpoint inhibitor"),
    "avelumab": ("PD-L1 inhibitor", "immune checkpoint inhibitor"),
    "ipilimumab": ("CTLA-4 inhibitor", "immune checkpoint inhibitor"),
    "yervoy": ("CTLA-4 inhibitor", "immune checkpoint inhibitor"),
    "tremelimumab": ("CTLA-4 inhibitor", "immune checkpoint inhibitor"),
    "relatlimab": ("LAG-3 inhibitor", "immune checkpoint inhibitor"),

    # === Kinase Inhibitors ===
    "imatinib": ("BCR-ABL inhibitor", "kinase inhibitor"),
    "gleevec": ("BCR-ABL inhibitor", "kinase inhibitor"),
    "dasatinib": ("BCR-ABL inhibitor", "kinase inhibitor"),
    "nilotinib": ("BCR-ABL inhibitor", "kinase inhibitor"),
    "bosutinib": ("BCR-ABL inhibitor", "kinase inhibitor"),
    "ponatinib": ("BCR-ABL inhibitor", "kinase inhibitor"),
    "ibrutinib": ("BTK inhibitor", "kinase inhibitor"),
    "acalabrutinib": ("BTK inhibitor", "kinase inhibitor"),
    "zanubrutinib": ("BTK inhibitor", "kinase inhibitor"),
    "pirtobrutinib": ("BTK inhibitor", "kinase inhibitor"),
    "erlotinib": ("EGFR inhibitor", "kinase inhibitor"),
    "gefitinib": ("EGFR inhibitor", "kinase inhibitor"),
    "osimertinib": ("EGFR inhibitor", "kinase inhibitor"),
    "tagrisso": ("EGFR inhibitor", "kinase inhibitor"),
    "lapatinib": ("HER2 inhibitor", "kinase inhibitor"),
    "tucatinib": ("HER2 inhibitor", "kinase inhibitor"),
    "neratinib": ("HER2 inhibitor", "kinase inhibitor"),
    "crizotinib": ("ALK inhibitor", "kinase inhibitor"),
    "ceritinib": ("ALK inhibitor", "kinase inhibitor"),
    "alectinib": ("ALK inhibitor", "kinase inhibitor"),
    "lorlatinib": ("ALK inhibitor", "kinase inhibitor"),
    "brigatinib": ("ALK inhibitor", "kinase inhibitor"),
    "vemurafenib": ("BRAF inhibitor", "kinase inhibitor"),
    "dabrafenib": ("BRAF inhibitor", "kinase inhibitor"),
    "encorafenib": ("BRAF inhibitor", "kinase inhibitor"),
    "trametinib": ("MEK inhibitor", "kinase inhibitor"),
    "cobimetinib": ("MEK inhibitor", "kinase inhibitor"),
    "binimetinib": ("MEK inhibitor", "kinase inhibitor"),
    "selpercatinib": ("RET inhibitor", "kinase inhibitor"),
    "pralsetinib": ("RET inhibitor", "kinase inhibitor"),
    "sorafenib": ("multi-kinase inhibitor", "kinase inhibitor"),
    "sunitinib": ("multi-kinase inhibitor", "kinase inhibitor"),
    "lenvatinib": ("multi-kinase inhibitor", "kinase inhibitor"),
    "cabozantinib": ("multi-kinase inhibitor", "kinase inhibitor"),
    "regorafenib": ("multi-kinase inhibitor", "kinase inhibitor"),
    "axitinib": ("VEGFR inhibitor", "kinase inhibitor"),
    "pazopanib": ("VEGFR inhibitor", "kinase inhibitor"),
    "ruxolitinib": ("JAK inhibitor", "JAK inhibitor"),
    "tofacitinib": ("JAK inhibitor", "JAK inhibitor"),
    "baricitinib": ("JAK inhibitor", "JAK inhibitor"),
    "upadacitinib": ("JAK inhibitor", "JAK inhibitor"),
    "abemaciclib": ("CDK4/6 inhibitor", "kinase inhibitor"),
    "palbociclib": ("CDK4/6 inhibitor", "kinase inhibitor"),
    "ribociclib": ("CDK4/6 inhibitor", "kinase inhibitor"),
    "olaparib": ("PARP inhibitor", "DNA repair inhibitor"),
    "niraparib": ("PARP inhibitor", "DNA repair inhibitor"),
    "rucaparib": ("PARP inhibitor", "DNA repair inhibitor"),
    "talazoparib": ("PARP inhibitor", "DNA repair inhibitor"),

    # === Monoclonal Antibodies (non-checkpoint) ===
    "trastuzumab": ("HER2 antibody", "targeted antibody"),
    "herceptin": ("HER2 antibody", "targeted antibody"),
    "pertuzumab": ("HER2 antibody", "targeted antibody"),
    "bevacizumab": ("VEGF antibody", "targeted antibody"),
    "avastin": ("VEGF antibody", "targeted antibody"),
    "ramucirumab": ("VEGFR2 antibody", "targeted antibody"),
    "cetuximab": ("EGFR antibody", "targeted antibody"),
    "panitumumab": ("EGFR antibody", "targeted antibody"),
    "rituximab": ("CD20 antibody", "targeted antibody"),
    "obinutuzumab": ("CD20 antibody", "targeted antibody"),
    "daratumumab": ("CD38 antibody", "targeted antibody"),
    "tocilizumab": ("IL-6R antibody", "immunomodulatory antibody"),
    "actemra": ("IL-6R antibody", "immunomodulatory antibody"),
    "sarilumab": ("IL-6R antibody", "immunomodulatory antibody"),
    "adalimumab": ("TNF-alpha antibody", "immunomodulatory antibody"),
    "humira": ("TNF-alpha antibody", "immunomodulatory antibody"),
    "infliximab": ("TNF-alpha antibody", "immunomodulatory antibody"),
    "certolizumab": ("TNF-alpha antibody", "immunomodulatory antibody"),
    "golimumab": ("TNF-alpha antibody", "immunomodulatory antibody"),
    "secukinumab": ("IL-17A antibody", "immunomodulatory antibody"),
    "ixekizumab": ("IL-17A antibody", "immunomodulatory antibody"),
    "ustekinumab": ("IL-12/23 antibody", "immunomodulatory antibody"),
    "guselkumab": ("IL-23 antibody", "immunomodulatory antibody"),
    "risankizumab": ("IL-23 antibody", "immunomodulatory antibody"),
    "dupilumab": ("IL-4R antibody", "immunomodulatory antibody"),
    "omalizumab": ("IgE antibody", "immunomodulatory antibody"),
    "mepolizumab": ("IL-5 antibody", "immunomodulatory antibody"),
    "benralizumab": ("IL-5R antibody", "immunomodulatory antibody"),
    "denosumab": ("RANKL antibody", "bone-targeted antibody"),
    "eculizumab": ("C5 complement antibody", "complement inhibitor"),
    "ravulizumab": ("C5 complement antibody", "complement inhibitor"),

    # === ADCs ===
    "trastuzumab emtansine": ("HER2 ADC", "ADC"),
    "t-dm1": ("HER2 ADC", "ADC"),
    "trastuzumab deruxtecan": ("HER2 ADC", "ADC"),
    "t-dxd": ("HER2 ADC", "ADC"),
    "enfortumab vedotin": ("Nectin-4 ADC", "ADC"),
    "sacituzumab govitecan": ("Trop-2 ADC", "ADC"),
    "brentuximab vedotin": ("CD30 ADC", "ADC"),
    "polatuzumab vedotin": ("CD79b ADC", "ADC"),
    "loncastuximab tesirine": ("CD19 ADC", "ADC"),
    "mirvetuximab soravtansine": ("FRalpha ADC", "ADC"),

    # === Bispecifics ===
    "blinatumomab": ("CD19xCD3 bispecific", "bispecific antibody"),
    "mosunetuzumab": ("CD20xCD3 bispecific", "bispecific antibody"),
    "glofitamab": ("CD20xCD3 bispecific", "bispecific antibody"),
    "teclistamab": ("BCMAxCD3 bispecific", "bispecific antibody"),
    "elranatamab": ("BCMAxCD3 bispecific", "bispecific antibody"),
    "epcoritamab": ("CD20xCD3 bispecific", "bispecific antibody"),
    "amivantamab": ("EGFRxMET bispecific", "bispecific antibody"),

    # === CAR-T ===
    "tisagenlecleucel": ("CD19 CAR-T", "CAR-T cell therapy"),
    "kymriah": ("CD19 CAR-T", "CAR-T cell therapy"),
    "axicabtagene ciloleucel": ("CD19 CAR-T", "CAR-T cell therapy"),
    "yescarta": ("CD19 CAR-T", "CAR-T cell therapy"),
    "lisocabtagene maraleucel": ("CD19 CAR-T", "CAR-T cell therapy"),
    "idecabtagene vicleucel": ("BCMA CAR-T", "CAR-T cell therapy"),
    "ciltacabtagene autoleucel": ("BCMA CAR-T", "CAR-T cell therapy"),

    # === Cytotoxic Chemotherapy ===
    "cisplatin": ("platinum agent", "cytotoxic chemotherapy"),
    "carboplatin": ("platinum agent", "cytotoxic chemotherapy"),
    "oxaliplatin": ("platinum agent", "cytotoxic chemotherapy"),
    "docetaxel": ("taxane", "cytotoxic chemotherapy"),
    "paclitaxel": ("taxane", "cytotoxic chemotherapy"),
    "nab-paclitaxel": ("taxane", "cytotoxic chemotherapy"),
    "gemcitabine": ("antimetabolite", "cytotoxic chemotherapy"),
    "capecitabine": ("antimetabolite", "cytotoxic chemotherapy"),
    "5-fluorouracil": ("antimetabolite", "cytotoxic chemotherapy"),
    "pemetrexed": ("antimetabolite", "cytotoxic chemotherapy"),
    "methotrexate": ("antimetabolite", "cytotoxic chemotherapy"),
    "cyclophosphamide": ("alkylating agent", "cytotoxic chemotherapy"),
    "ifosfamide": ("alkylating agent", "cytotoxic chemotherapy"),
    "temozolomide": ("alkylating agent", "cytotoxic chemotherapy"),
    "doxorubicin": ("anthracycline", "cytotoxic chemotherapy"),
    "epirubicin": ("anthracycline", "cytotoxic chemotherapy"),
    "irinotecan": ("topoisomerase I inhibitor", "cytotoxic chemotherapy"),
    "topotecan": ("topoisomerase I inhibitor", "cytotoxic chemotherapy"),
    "etoposide": ("topoisomerase II inhibitor", "cytotoxic chemotherapy"),
    "vincristine": ("vinca alkaloid", "cytotoxic chemotherapy"),
    "vinblastine": ("vinca alkaloid", "cytotoxic chemotherapy"),

    # === PK Probe Drugs (DDI studies) ===
    "midazolam": ("CYP3A4 substrate", "PK probe"),
    "alprazolam": ("CYP3A4 substrate", "PK probe"),
    "triazolam": ("CYP3A4 substrate", "PK probe"),
    "simvastatin": ("CYP3A4 substrate", "PK probe"),
    "itraconazole": ("CYP3A4 inhibitor", "CYP inhibitor"),
    "ketoconazole": ("CYP3A4 inhibitor", "CYP inhibitor"),
    "voriconazole": ("CYP3A4 inhibitor", "CYP inhibitor"),
    "rifampin": ("CYP3A4 inducer", "CYP inducer"),
    "rifampicin": ("CYP3A4 inducer", "CYP inducer"),
    "rifabutin": ("CYP3A4 inducer", "CYP inducer"),
    "carbamazepine": ("CYP3A4 inducer", "CYP inducer"),
    "phenytoin": ("CYP3A4 inducer", "CYP inducer"),
    "omeprazole": ("CYP2C19 substrate", "PK probe"),
    "lansoprazole": ("CYP2C19 substrate", "PK probe"),
    "warfarin": ("CYP2C9 substrate", "PK probe"),
    "s-warfarin": ("CYP2C9 substrate", "PK probe"),
    "tolbutamide": ("CYP2C9 substrate", "PK probe"),
    "dextromethorphan": ("CYP2D6 substrate", "PK probe"),
    "metoprolol": ("CYP2D6 substrate", "PK probe"),
    "caffeine": ("CYP1A2 substrate", "PK probe"),
    "theophylline": ("CYP1A2 substrate", "PK probe"),
    "fluvoxamine": ("CYP1A2 inhibitor", "CYP inhibitor"),
    "fluconazole": ("CYP2C9 inhibitor", "CYP inhibitor"),
    "quinidine": ("CYP2D6 inhibitor", "CYP inhibitor"),
    "paroxetine": ("CYP2D6 inhibitor", "CYP inhibitor"),
    "digoxin": ("P-gp substrate", "transporter probe"),
    "rosuvastatin": ("OATP substrate", "transporter probe"),
    "metformin": ("OCT2/MATE substrate", "transporter probe"),
    "furosemide": ("OAT substrate", "transporter probe"),

    # === Lipid-Lowering ===
    "evolocumab": ("PCSK9 inhibitor", "lipid-lowering biologic"),
    "alirocumab": ("PCSK9 inhibitor", "lipid-lowering biologic"),
    "inclisiran": ("PCSK9 siRNA", "lipid-lowering biologic"),
    "atorvastatin": ("HMG-CoA reductase inhibitor", "statin"),
    "rosuvastatin": ("HMG-CoA reductase inhibitor", "statin"),
    "ezetimibe": ("cholesterol absorption inhibitor", "lipid-lowering small molecule"),

    # === Anticoagulants ===
    "rivaroxaban": ("Factor Xa inhibitor", "anticoagulant"),
    "apixaban": ("Factor Xa inhibitor", "anticoagulant"),
    "edoxaban": ("Factor Xa inhibitor", "anticoagulant"),
    "dabigatran": ("direct thrombin inhibitor", "anticoagulant"),
    "enoxaparin": ("LMWH", "anticoagulant"),
    "heparin": ("unfractionated heparin", "anticoagulant"),

    # === Antidiabetics ===
    "semaglutide": ("GLP-1 receptor agonist", "incretin-based therapy"),
    "liraglutide": ("GLP-1 receptor agonist", "incretin-based therapy"),
    "dulaglutide": ("GLP-1 receptor agonist", "incretin-based therapy"),
    "tirzepatide": ("GIP/GLP-1 dual agonist", "incretin-based therapy"),
    "empagliflozin": ("SGLT2 inhibitor", "SGLT2 inhibitor"),
    "dapagliflozin": ("SGLT2 inhibitor", "SGLT2 inhibitor"),
    "canagliflozin": ("SGLT2 inhibitor", "SGLT2 inhibitor"),
    "sitagliptin": ("DPP-4 inhibitor", "incretin-based therapy"),
    "linagliptin": ("DPP-4 inhibitor", "incretin-based therapy"),
    "insulin": ("insulin", "insulin"),
    "insulin glargine": ("long-acting insulin", "insulin"),
    "insulin lispro": ("rapid-acting insulin", "insulin"),

    # === Antivirals ===
    "sofosbuvir": ("NS5B polymerase inhibitor", "direct-acting antiviral"),
    "ledipasvir": ("NS5A inhibitor", "direct-acting antiviral"),
    "velpatasvir": ("NS5A inhibitor", "direct-acting antiviral"),
    "glecaprevir": ("NS3/4A protease inhibitor", "direct-acting antiviral"),
    "pibrentasvir": ("NS5A inhibitor", "direct-acting antiviral"),
    "tenofovir": ("nucleotide reverse transcriptase inhibitor", "antiviral NRTI"),
    "emtricitabine": ("nucleoside reverse transcriptase inhibitor", "antiviral NRTI"),
    "dolutegravir": ("integrase inhibitor", "antiviral integrase inhibitor"),
    "bictegravir": ("integrase inhibitor", "antiviral integrase inhibitor"),
    "remdesivir": ("RNA polymerase inhibitor", "antiviral"),
    "nirmatrelvir": ("3CL protease inhibitor", "antiviral"),
    "paxlovid": ("3CL protease inhibitor", "antiviral"),

    # === Other Common ===
    "dexamethasone": ("corticosteroid", "corticosteroid"),
    "prednisone": ("corticosteroid", "corticosteroid"),
    "prednisolone": ("corticosteroid", "corticosteroid"),
    "methylprednisolone": ("corticosteroid", "corticosteroid"),
    "tacrolimus": ("calcineurin inhibitor", "immunosuppressant"),
    "cyclosporine": ("calcineurin inhibitor", "immunosuppressant"),
    "mycophenolate": ("IMPDH inhibitor", "immunosuppressant"),
    "sirolimus": ("mTOR inhibitor", "mTOR inhibitor"),
    "everolimus": ("mTOR inhibitor", "mTOR inhibitor"),
    "temsirolimus": ("mTOR inhibitor", "mTOR inhibitor"),
    "lenalidomide": ("IMiD", "immunomodulatory agent"),
    "pomalidomide": ("IMiD", "immunomodulatory agent"),
    "thalidomide": ("IMiD", "immunomodulatory agent"),
    "venetoclax": ("BCL-2 inhibitor", "apoptosis modulator"),
    "bortezomib": ("proteasome inhibitor", "proteasome inhibitor"),
    "carfilzomib": ("proteasome inhibitor", "proteasome inhibitor"),
    "ixazomib": ("proteasome inhibitor", "proteasome inhibitor"),
}

# ---------------------------------------------------------------------------
# Broader class hierarchy distances
# ---------------------------------------------------------------------------
# Same mechanism = 0.0
# Same broader class = 0.3
# Related classes = 0.6
# Unrelated = 1.0
_RELATED_CLASSES: dict[frozenset, float] = {
    frozenset({"immune checkpoint inhibitor", "immunomodulatory antibody"}): 0.5,
    frozenset({"immune checkpoint inhibitor", "targeted antibody"}): 0.6,
    frozenset({"kinase inhibitor", "DNA repair inhibitor"}): 0.6,
    frozenset({"cytotoxic chemotherapy", "DNA repair inhibitor"}): 0.6,
    frozenset({"PK probe", "CYP inhibitor"}): 0.2,
    frozenset({"PK probe", "CYP inducer"}): 0.2,
    frozenset({"CYP inhibitor", "CYP inducer"}): 0.3,
    frozenset({"PK probe", "transporter probe"}): 0.3,
    frozenset({"statin", "lipid-lowering biologic"}): 0.4,
    frozenset({"statin", "lipid-lowering small molecule"}): 0.3,
    frozenset({"targeted antibody", "ADC"}): 0.4,
    frozenset({"targeted antibody", "bispecific antibody"}): 0.4,
    frozenset({"ADC", "bispecific antibody"}): 0.5,
    frozenset({"CAR-T cell therapy", "bispecific antibody"}): 0.5,
    frozenset({"immunomodulatory antibody", "immunomodulatory agent"}): 0.5,
    frozenset({"GLP-1 receptor agonist", "incretin-based therapy"}): 0.2,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def drug_to_mechanism(drug_name: str) -> tuple[str, str] | None:
    """Look up a drug's mechanism class.

    Args:
        drug_name: Drug name (case-insensitive).

    Returns:
        (specific_mechanism, broader_class) or None if not found.
    """
    if not drug_name:
        return None

    name = drug_name.lower().strip()

    # Exact match
    if name in DRUG_TO_CLASS:
        return DRUG_TO_CLASS[name]

    # Try substring: check if drug name contains a known drug
    for key, value in DRUG_TO_CLASS.items():
        if key in name or name in key:
            return value

    return None


def mechanism_class_distance(
    class_a: tuple[str, str] | None,
    class_b: tuple[str, str] | None,
) -> float:
    """Compute distance between two drug mechanism classes.

    Args:
        class_a: (specific_mechanism, broader_class) or None
        class_b: (specific_mechanism, broader_class) or None

    Returns:
        0.0 if same mechanism,
        0.3 if same broader class,
        0.2-0.6 if related classes,
        1.0 if unrelated or unknown.
    """
    if class_a is None or class_b is None:
        return 0.5  # Unknown drug — neutral

    mech_a, broad_a = class_a
    mech_b, broad_b = class_b

    # Same specific mechanism
    if mech_a == mech_b:
        return 0.0

    # Same broader class
    if broad_a == broad_b:
        return 0.3

    # Check related classes
    pair = frozenset({broad_a, broad_b})
    if pair in _RELATED_CLASSES:
        return _RELATED_CLASSES[pair]

    return 1.0


def intervention_set_similarity(
    interventions_a: list[str],
    interventions_b: list[str],
) -> tuple[float, str]:
    """Compute similarity between two sets of interventions using drug class distance.

    For each intervention in set A, finds the closest match in set B.
    Averages across all. Symmetrizes.

    Returns:
        (similarity_score, detail_string)
    """
    if not interventions_a and not interventions_b:
        return 0.5, "No interventions specified"
    if not interventions_a or not interventions_b:
        return 0.0, "Missing interventions"

    # Map interventions to mechanisms
    mapped_a = [(i, drug_to_mechanism(i)) for i in interventions_a]
    mapped_b = [(i, drug_to_mechanism(i)) for i in interventions_b]

    known_a = [(i, m) for i, m in mapped_a if m is not None]
    known_b = [(i, m) for i, m in mapped_b if m is not None]

    if not known_a and not known_b:
        return 0.5, "Drugs not in class mapping"
    if not known_a or not known_b:
        return 0.3, "Partial drug mapping"

    # Directional distance
    def _dir_dist(from_set, to_set):
        total = 0.0
        for _, mech_from in from_set:
            min_d = 1.0
            for _, mech_to in to_set:
                d = mechanism_class_distance(mech_from, mech_to)
                min_d = min(min_d, d)
            total += min_d
        return total / len(from_set)

    dist_ab = _dir_dist(known_a, known_b)
    dist_ba = _dir_dist(known_b, known_a)
    avg_dist = (dist_ab + dist_ba) / 2.0
    similarity = 1.0 - avg_dist

    # Build detail
    detail = _build_drug_detail(known_a, known_b)

    return similarity, detail


def _build_drug_detail(known_a: list, known_b: list) -> str:
    """Build human-readable drug similarity detail."""
    mechs_a = {m[0] for _, m in known_a}
    mechs_b = {m[0] for _, m in known_b}

    overlap = mechs_a & mechs_b
    if overlap:
        return f"Both {list(overlap)[0]}"

    broads_a = {m[1] for _, m in known_a}
    broads_b = {m[1] for _, m in known_b}
    broad_overlap = broads_a & broads_b
    if broad_overlap:
        return f"Both {list(broad_overlap)[0]}"

    name_a = known_a[0][1][0] if known_a else "?"
    name_b = known_b[0][1][0] if known_b else "?"
    return f"{name_a} vs {name_b}"
