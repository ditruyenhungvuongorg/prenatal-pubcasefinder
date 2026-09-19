"""Agent 3: Clinical Decision Support & Genetic Testing Recommender for Prenatal Ultrasound.

Combines knowledge-grounded clinical pattern synthesis, prenatal disease reranking,
and tiered genetic testing recommendations according to ACMG/ACOG guidelines.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


# ============================================================================
# 1. ORGAN SYSTEM MAPPINGS (Based on HPO Top-Level Phenotypic Abnormalities)
# ============================================================================

ORGAN_SYSTEM_ROOTS: Dict[str, str] = {
    "HP:0001627": "Tim mạch (Cardiovascular)",
    "HP:0000707": "Thần kinh trung ương (CNS)",
    "HP:0000152": "Sọ mặt (Craniofacial)",
    "HP:0001438": "Thành bụng (Abdominal wall)",
    "HP:0025031": "Tiêu hóa (Gastrointestinal)",
    "HP:0000924": "Cơ xương & Chi (Musculoskeletal)",
    "HP:0000119": "Tiết niệu & Sinh dục (Genitourinary)",
    "HP:0001507": "Tăng trưởng (Growth)",
    "HP:0001197": "Phụ sản & Ối (Prenatal / Amniotic)",
    "HP:0000598": "Tai & Mắt (Sensory)",
    "HP:0001939": "Huyết học & Miễn dịch (Hematology/Immune)",
    "HP:0002086": "Hô hấp & Lồng ngực (Respiratory/Thorax)",
}

# Key prenatal HPO patterns mapped directly to organ systems for speed
HPO_TO_ORGAN_FAST_MAP: Dict[str, str] = {
    # Heart
    "HP:0001674": "Tim mạch (Cardiovascular)", # AVSD
    "HP:0001636": "Tim mạch (Cardiovascular)", # Tetralogy of Fallot
    "HP:0001629": "Tim mạch (Cardiovascular)", # Ventricular septal defect
    "HP:0001631": "Tim mạch (Cardiovascular)", # Atrial septal defect
    "HP:0001640": "Tim mạch (Cardiovascular)", # Cardiomegaly
    "HP:0001639": "Tim mạch (Cardiovascular)", # HCM
    "HP:0009729": "Tim mạch (Cardiovascular)", # Cardiac rhabdomyoma
    "HP:0012020": "Tim mạch (Cardiovascular)", # Right aortic arch
    "HP:0031632": "Tim mạch (Cardiovascular)", # ARSA
    "HP:0012304": "Tim mạch (Cardiovascular)", # Hypoplastic aortic arch
    "HP:0001680": "Tim mạch (Cardiovascular)", # Coarctation of aorta
    "HP:0001683": "Tim mạch (Cardiovascular)", # Transposition of great arteries
    "HP:0001698": "Tim mạch (Cardiovascular)", # Pericardial effusion
    "HP:0001719": "Tim mạch (Cardiovascular)", # Ebstein anomaly
    # CNS
    "HP:0001274": "Thần kinh trung ương (CNS)", # Agenesis of corpus callosum
    "HP:0001320": "Thần kinh trung ương (CNS)", # Cerebellar vermis hypoplasia
    "HP:0002119": "Thần kinh trung ương (CNS)", # Ventriculomegaly
    "HP:0001360": "Thần kinh trung ương (CNS)", # Holoprosencephaly
    "HP:0002416": "Thần kinh trung ương (CNS)", # Subependymal cysts
    "HP:0030724": "Thần kinh trung ương (CNS)", # CNS cyst
    "HP:0002059": "Thần kinh trung ương (CNS)", # Cerebral atrophy
    "HP:0040196": "Thần kinh trung ương (CNS)", # Microcephaly
    "HP:0002269": "Thần kinh trung ương (CNS)", # Dandy-Walker malformation
    "HP:0002414": "Thần kinh trung ương (CNS)", # Spina bifida
    "HP:0002084": "Thần kinh trung ương (CNS)", # Encephalocele
    # Abdominal wall & GI
    "HP:0001539": "Thành bụng & Tiêu hóa", # Omphalocele
    "HP:0001543": "Thành bụng & Tiêu hóa", # Gastroschisis
    "HP:0005214": "Thành bụng & Tiêu hóa", # Intestinal obstruction
    "HP:0002247": "Thành bụng & Tiêu hóa", # Duodenal atresia
    "HP:0030656": "Thành bụng & Tiêu hóa", # Hyperechoic bowel
    "HP:0002575": "Thành bụng & Tiêu hóa", # Tracheoesophageal fistula
    # Craniofacial
    "HP:0000202": "Sọ mặt (Craniofacial)", # Cleft lip
    "HP:0000175": "Sọ mặt (Craniofacial)", # Cleft palate
    "HP:0000347": "Sọ mặt (Craniofacial)", # Micrognathia
    "HP:0000154": "Sọ mặt (Craniofacial)", # Wide mouth
    "HP:0000343": "Sọ mặt (Craniofacial)", # Long philtrum
    "HP:0000316": "Sọ mặt (Craniofacial)", # Hypertelorism
    "HP:0000369": "Sọ mặt (Craniofacial)", # Low-set ears
    # Musculoskeletal
    "HP:0002758": "Cơ xương & Chi", # Osteogenesis imperfecta / bone fragility
    "HP:0002650": "Cơ xương & Chi", # Scoliosis
    "HP:0025481": "Cơ xương & Chi", # Cervical hemivertebrae
    "HP:0001156": "Cơ xương & Chi", # Brachydactyly
    "HP:0001166": "Cơ xương & Chi", # Arachnodactyly
    "HP:0010049": "Cơ xương & Chi", # Short femur
    "HP:0001762": "Cơ xương & Chi", # Talipes equinovarus (clubfoot)
    "HP:0001162": "Cơ xương & Chi", # Polydactyly
    "HP:0001180": "Cơ xương & Chi", # Clenched hands
    "HP:0011843": "Cơ xương & Chi", # Rhizomelic micromelia
    # Genitourinary
    "HP:0000107": "Tiết niệu & Thận", # Renal cyst
    "HP:0000079": "Tiết niệu & Thận", # Hydronephrosis
    "HP:0000110": "Tiết niệu & Thận", # Renal hypoplasia
    "HP:0000104": "Tiết niệu & Thận", # Renal agenesis
    "HP:0000028": "Tiết niệu & Thận", # Cryptorchidism
    "HP:0000047": "Tiết niệu & Thận", # Hypospadias
    # Growth & Prenatal markers
    "HP:0001511": "Tăng trưởng & Ối", # IUGR
    "HP:0001562": "Tăng trưởng & Ối", # Oligohydramnios
    "HP:0001561": "Tăng trưởng & Ối", # Polyhydramnios
    "HP:0001789": "Tăng trưởng & Ối", # Hydrops fetalis
    "HP:0010880": "Tăng trưởng & Ối", # Increased nuchal translucency
    "HP:0000474": "Tăng trưởng & Ối", # Cystic hygroma
    "HP:0032548": "Tăng trưởng & Ối", # Increased placental thickness
    "HP:0030658": "Phụ sản", # Marginal umbilical cord insertion
}


# ============================================================================
# 2. PRENATAL DISEASE KNOWLEDGE BASE (Priors & High-Yield Syndromes)
# ============================================================================

@dataclass(frozen=True)
class PrenatalSyndromeProfile:
    canonical_id: str
    name: str
    category: str # "aneuploidy", "cnv_microdeletion", "monogenic", "ciliopathy", "rasopathy"
    common_genes: Tuple[str, ...]
    hallmark_patterns: Tuple[str, ...]
    recommended_first_tier: str
    recommended_second_tier: str
    base_prior_weight: float = 1.0


PRENATAL_SYNDROME_CATALOG: Dict[str, PrenatalSyndromeProfile] = {
    "trisomy_18": PrenatalSyndromeProfile(
        canonical_id="ORPHA:3380",
        name="Trisomy 18 (Edwards syndrome)",
        category="aneuploidy",
        common_genes=(),
        hallmark_patterns=("Tim mạch", "Thần kinh trung ương", "Thành bụng", "Cơ xương", "Tăng trưởng & Ối"),
        recommended_first_tier="QF-PCR nhanh dị bội + Karyotype / CMA từ mẫu ối",
        recommended_second_tier="Không cần nếu đã xác định Trisomy 18",
        base_prior_weight=2.5,
    ),
    "trisomy_13": PrenatalSyndromeProfile(
        canonical_id="ORPHA:3378",
        name="Trisomy 13 (Patau syndrome)",
        category="aneuploidy",
        common_genes=(),
        hallmark_patterns=("Thần kinh trung ương", "Sọ mặt", "Tim mạch", "Tiết niệu", "Thành bụng"),
        recommended_first_tier="QF-PCR nhanh dị bội + Karyotype / CMA từ mẫu ối",
        recommended_second_tier="Không cần nếu đã xác định Trisomy 13",
        base_prior_weight=2.2,
    ),
    "trisomy_21": PrenatalSyndromeProfile(
        canonical_id="ORPHA:870",
        name="Trisomy 21 (Down syndrome)",
        category="aneuploidy",
        common_genes=(),
        hallmark_patterns=("Tim mạch", "Tiêu hóa", "Tăng trưởng & Ối", "Sọ mặt"),
        recommended_first_tier="QF-PCR nhanh dị bội + Karyotype / CMA từ mẫu ối",
        recommended_second_tier="Không cần nếu đã xác định Trisomy 21",
        base_prior_weight=2.8,
    ),
    "turner": PrenatalSyndromeProfile(
        canonical_id="ORPHA:886",
        name="Turner syndrome (Monosomy X)",
        category="aneuploidy",
        common_genes=(),
        hallmark_patterns=("Tim mạch", "Tăng trưởng & Ối", "Tiết niệu"),
        recommended_first_tier="QF-PCR / Karyotype + CMA khảo sát thể khảm",
        recommended_second_tier="Không cần nếu đã xác định Monosomy X",
        base_prior_weight=2.0,
    ),
    "digeorge": PrenatalSyndromeProfile(
        canonical_id="ORPHA:567",
        name="22q11.2 deletion syndrome (DiGeorge / VCFS)",
        category="cnv_microdeletion",
        common_genes=("TBX1",),
        hallmark_patterns=("Tim mạch", "Sọ mặt", "Tiết niệu"),
        recommended_first_tier="CMA (Chromosomal Microarray) - xét nghiệm nền tảng hàng đầu",
        recommended_second_tier="Nếu CMA âm tính và đa dị tật: Trio WES / Panel tim bẩm sinh",
        base_prior_weight=2.6,
    ),
    "tuberous_sclerosis": PrenatalSyndromeProfile(
        canonical_id="ORPHA:805",
        name="Tuberous Sclerosis Complex (TSC1/TSC2)",
        category="monogenic",
        common_genes=("TSC1", "TSC2"),
        hallmark_patterns=("Tim mạch", "Thần kinh trung ương"),
        recommended_first_tier="Giải trình tự gen TSC1/TSC2 + phân tích MLPA / del-dup",
        recommended_second_tier="CMA khảo sát mất đoạn liên quan hoặc Trio WES",
        base_prior_weight=3.0,
    ),
    "noonan": PrenatalSyndromeProfile(
        canonical_id="ORPHA:648",
        name="Noonan syndrome & RASopathy spectrum",
        category="rasopathy",
        common_genes=("PTPN11", "SOS1", "RAF1", "RIT1", "KRAS", "BRAF"),
        hallmark_patterns=("Tim mạch", "Tăng trưởng & Ối", "Sọ mặt"),
        recommended_first_tier="CMA loại trừ CNV + Panel RASopathy / Trio WES",
        recommended_second_tier="Trio WES phân tích sâu biến thể khảm / de novo",
        base_prior_weight=2.2,
    ),
    "beckwith_wiedemann": PrenatalSyndromeProfile(
        canonical_id="ORPHA:116",
        name="Beckwith–Wiedemann syndrome (BWS)",
        category="cnv_microdeletion",
        common_genes=("CDKN1C", "KCNQ1OT1", "H19"),
        hallmark_patterns=("Thành bụng & Tiêu hóa", "Tăng trưởng & Ối"),
        recommended_first_tier="CMA + Xét nghiệm Methylation-specific MLPA vùng 11p15",
        recommended_second_tier="Giải trình tự CDKN1C nếu nghi ngờ di truyền trội từ mẹ",
        base_prior_weight=2.0,
    ),
    "cystic_fibrosis": PrenatalSyndromeProfile(
        canonical_id="ORPHA:586",
        name="Cystic fibrosis (CFTR-related)",
        category="monogenic",
        common_genes=("CFTR",),
        hallmark_patterns=("Thành bụng & Tiêu hóa",),
        recommended_first_tier="Sàng lọc người mang gen CFTR bố mẹ; giải trình tự CFTR thai",
        recommended_second_tier="Trio WES nếu nghi ngờ teo ruột non do nguyên nhân khác (TTC7A)",
        base_prior_weight=2.4,
    ),
    "meckel_gruber": PrenatalSyndromeProfile(
        canonical_id="ORPHA:564",
        name="Meckel–Gruber syndrome (MKS)",
        category="ciliopathy",
        common_genes=("TMEM67", "RPGRIP1L", "CC2D2A", "CEP290"),
        hallmark_patterns=("Thần kinh trung ương", "Tiết niệu & Thận", "Cơ xương & Chi"),
        recommended_first_tier="CMA loại trừ lệch bội + Trio WES / Panel Ciliopathy",
        recommended_second_tier="Trio WES giải trình tự toàn bộ vùng mã hóa cho thai và bố mẹ",
        base_prior_weight=2.1,
    ),
    "skeletal_dysplasia": PrenatalSyndromeProfile(
        canonical_id="ORPHA:15",
        name="Loạn sản xương (FGFR3 / COL1A1 / COL1A2 / Thanatophoric)",
        category="monogenic",
        common_genes=("FGFR3", "COL1A1", "COL1A2", "FLNB"),
        hallmark_patterns=("Cơ xương & Chi", "Tăng trưởng & Ối"),
        recommended_first_tier="CMA + Panel gen Loạn sản xương (ưu tiên FGFR3 hot-spot)",
        recommended_second_tier="Trio WES nếu nghi ngờ các thể loạn sản xương hiếm gặp",
        base_prior_weight=2.5,
    ),
    "charge": PrenatalSyndromeProfile(
        canonical_id="ORPHA:138",
        name="CHARGE syndrome",
        category="monogenic",
        common_genes=("CHD7",),
        hallmark_patterns=("Tim mạch", "Sọ mặt", "Thần kinh trung ương", "Tai & Mắt"),
        recommended_first_tier="CMA loại trừ CNV + Giải trình tự gen CHD7",
        recommended_second_tier="Trio WES nếu giải trình tự gen mục tiêu âm tính",
        base_prior_weight=2.0,
    ),
    "ritscher_schinzel": PrenatalSyndromeProfile(
        canonical_id="OMIM:220210",
        name="Ritscher–Schinzel / 3C syndrome",
        category="monogenic",
        common_genes=("WASHC5", "CCDC22"),
        hallmark_patterns=("Thần kinh trung ương", "Tim mạch", "Sọ mặt"),
        recommended_first_tier="CMA loại trừ bất thường NST/CNV",
        recommended_second_tier="Trio WES phân tích sâu WASHC5 và CCDC22",
        base_prior_weight=2.2,
    ),
    "heterotaxy": PrenatalSyndromeProfile(
        canonical_id="ORPHA:45353",
        name="Heterotaxy / Situs inversus / Isomerism spectrum",
        category="monogenic",
        common_genes=("ZIC3", "NODAL", "CFC1", "GDF1"),
        hallmark_patterns=("Tim mạch", "Thành bụng & Tiêu hóa"),
        recommended_first_tier="CMA loại trừ CNV + Panel gen Laterality / Heterotaxy",
        recommended_second_tier="Trio WES nếu nghi ngờ đột biến gen đơn lẻ chưa xác định",
        base_prior_weight=2.4,
    ),
    "pentalogy_of_cantrell": PrenatalSyndromeProfile(
        canonical_id="ORPHA:1376",
        name="Pentalogy of Cantrell",
        category="cnv_microdeletion",
        common_genes=(),
        hallmark_patterns=("Tim mạch", "Thành bụng & Tiêu hóa"),
        recommended_first_tier="CMA khảo sát bất thường số lượng và vi mất đoạn NST",
        recommended_second_tier="Trio WES nếu có thêm bất thường ngoài đường giữa",
        base_prior_weight=3.0,
    ),
    "spondylocostal_dysostosis": PrenatalSyndromeProfile(
        canonical_id="ORPHA:139",
        name="Spondylocostal dysostosis / Bất thường phân đoạn đốt sống",
        category="monogenic",
        common_genes=("DLL3", "MESP2", "HES7", "LFNG", "TBX6"),
        hallmark_patterns=("Cơ xương & Chi",),
        recommended_first_tier="CMA loại trừ vi mất đoạn + Panel gen đốt sống / Trio WES",
        recommended_second_tier="Trio WES phân tích sâu các gen phân đoạn phôi trục",
        base_prior_weight=2.5,
    ),
    "van_der_woude": PrenatalSyndromeProfile(
        canonical_id="ORPHA:889",
        name="Van der Woude syndrome (IRF6 / GRHL3)",
        category="monogenic",
        common_genes=("IRF6", "GRHL3"),
        hallmark_patterns=("Sọ mặt",),
        recommended_first_tier="CMA + Giải trình tự gen IRF6 / GRHL3",
        recommended_second_tier="Trio WES nếu có thêm dị tật tim hoặc thần kinh",
        base_prior_weight=2.5,
    ),
    "arpkd": PrenatalSyndromeProfile(
        canonical_id="ORPHA:731",
        name="Thận đa nang lặn (ARPKD - PKHD1)",
        category="monogenic",
        common_genes=("PKHD1",),
        hallmark_patterns=("Tiết niệu & Thận", "Tăng trưởng & Ối"),
        recommended_first_tier="Giải trình tự gen PKHD1 (bố mẹ và thai) + CMA",
        recommended_second_tier="Trio WES nếu nghi ngờ ciliopathy thận thể khác",
        base_prior_weight=2.5,
    ),
    "hnf1b_renal": PrenatalSyndromeProfile(
        canonical_id="ORPHA:34516",
        name="Vi mất đoạn 17q12 / Bệnh thận liên quan HNF1B",
        category="cnv_microdeletion",
        common_genes=("HNF1B",),
        hallmark_patterns=("Tiết niệu & Thận",),
        recommended_first_tier="CMA phát hiện vi mất đoạn 17q12 hoặc giải trình tự HNF1B",
        recommended_second_tier="Trio WES nếu CMA âm tính",
        base_prior_weight=2.6,
    ),
    "vacterl": PrenatalSyndromeProfile(
        canonical_id="ORPHA:887",
        name="VACTERL / VATER association",
        category="cnv_microdeletion",
        common_genes=(),
        hallmark_patterns=("Cơ xương & Chi", "Tim mạch", "Thành bụng & Tiêu hóa", "Tiết niệu & Thận"),
        recommended_first_tier="CMA + Karyotype loại trừ lệch bội và CNV",
        recommended_second_tier="Trio WES nếu kiểu hình tiến triển nghi hội chứng đơn gen",
        base_prior_weight=2.2,
    ),
    "fryns": PrenatalSyndromeProfile(
        canonical_id="ORPHA:2059",
        name="Fryns syndrome",
        category="monogenic",
        common_genes=(),
        hallmark_patterns=("Thành bụng & Tiêu hóa", "Thần kinh trung ương", "Sọ mặt"),
        recommended_first_tier="CMA + Trio WES",
        recommended_second_tier="Trio WES",
        base_prior_weight=2.2,
    ),
    "multiple_intestinal_atresia": PrenatalSyndromeProfile(
        canonical_id="OMIM:243150",
        name="Gastrointestinal defects and immunodeficiency (TTC7A)",
        category="monogenic",
        common_genes=("TTC7A",),
        hallmark_patterns=("Thành bụng & Tiêu hóa",),
        recommended_first_tier="Giải trình tự gen TTC7A + CMA",
        recommended_second_tier="Trio WES",
        base_prior_weight=2.5,
    ),
}

# High-yield Hallmark Clinical Rules: HPO -> List of (syndrome_key, boost_multiplier)
HALLMARK_INDICATOR_RULES: Dict[str, List[Tuple[str, float]]] = {
    # Cardiac rhabdomyoma -> Tuberous sclerosis (boost áp đảo x15)
    "HP:0009729": [("tuberous_sclerosis", 15.0)],
    # Conotruncal heart defects -> 22q11.2 deletion (boost x8)
    "HP:0001636": [("digeorge", 8.0), ("trisomy_18", 3.0), ("charge", 4.0)], # TOF
    "HP:0012020": [("digeorge", 8.0)], # Right aortic arch
    "HP:0001683": [("digeorge", 5.0), ("pentalogy_of_cantrell", 5.0)], # TGA
    # Complete AVSD -> Trisomy 21 / Trisomy 18 (boost x8)
    "HP:0001674": [("trisomy_21", 8.0), ("trisomy_18", 7.0), ("ritscher_schinzel", 5.0)],
    # Aortic coarctation / Hypoplastic arch / Cystic hygroma -> Turner (boost x8)
    "HP:0012304": [("turner", 8.0), ("digeorge", 5.0)],
    "HP:0001680": [("turner", 8.0)],
    "HP:0000474": [("turner", 8.0), ("trisomy_21", 5.0), ("noonan", 5.0)],
    # Hypertrophic cardiomyopathy (HCM) -> Noonan / RASopathy (boost x8)
    "HP:0001639": [("noonan", 8.0)],
    # Omphalocele -> Trisomy 18, Beckwith-Wiedemann (boost x7)
    "HP:0001539": [("trisomy_18", 7.0), ("trisomy_13", 5.0), ("beckwith_wiedemann", 7.0)],
    # Holoprosencephaly -> Trisomy 13 (boost x10)
    "HP:0001360": [("trisomy_13", 10.0)],
    # Echogenic bowel / Intestinal obstruction -> Cystic fibrosis, Trisomy 21 (boost x8)
    "HP:0030656": [("cystic_fibrosis", 8.0), ("trisomy_21", 5.0)],
    "HP:0005214": [("cystic_fibrosis", 8.0), ("multiple_intestinal_atresia", 7.0)],
    # Duodenal atresia -> Trisomy 21 (boost x10)
    "HP:0002247": [("trisomy_21", 10.0)],
    # Polycystic kidney / Encephalocele -> Meckel-Gruber (boost x10)
    "HP:0002084": [("meckel_gruber", 10.0)],
    # Cleft lip/palate -> Van der Woude, Trisomy 13, 22q11 (boost x6)
    "HP:0000202": [("van_der_woude", 8.0), ("trisomy_13", 5.0), ("digeorge", 4.0)],
    # Hemivertebrae / Scoliosis -> Spondylocostal dysostosis, VACTERL (boost x8)
    "HP:0025481": [("spondylocostal_dysostosis", 8.0), ("vacterl", 6.0)],
    "HP:0002650": [("spondylocostal_dysostosis", 6.0)],
    # Micromelia / Short femur severe -> Skeletal dysplasia (boost x10)
    "HP:0011843": [("skeletal_dysplasia", 10.0)],
    "HP:0010049": [("skeletal_dysplasia", 6.0), ("trisomy_21", 4.0)],
    # Heterotaxy / Dextrocardia -> Heterotaxy syndrome (boost x10)
    "HP:0031855": [("heterotaxy", 10.0)],
    # Ectopia cordis -> Pentalogy of Cantrell (boost x15)
    "HP:0010866": [("pentalogy_of_cantrell", 15.0)],
    # Renal cysts -> ARPKD, HNF1B / 17q12 (boost x7)
    "HP:0000107": [("arpkd", 7.0), ("hnf1b_renal", 6.0)],
}


# ============================================================================
# 3. CORE PRENATAL DECISION AGENT IMPLEMENTATION
# ============================================================================

@dataclass
class ClinicalPatternResult:
    affected_organs: List[str]
    is_multiple: bool
    pattern_summary: str
    primary_suspicion: str # "Lệch bội NST (Aneuploidy)", "Vi mất đoạn / CNV", "Bệnh đơn gen", "Dị tật đơn độc"


@dataclass
class RerankedCandidate:
    rank: int
    disease_id: str
    disease_name: str
    original_rank: int
    original_score: float
    rerank_score: float
    clinical_rationale: str
    recommended_tests: str


@dataclass
class Agent3ClinicalReport:
    case_id: str
    clinical_pattern: str
    syndromes_text: str
    genetic_tests_text: str
    top_candidates: List[RerankedCandidate]
    structured_payload: Dict[str, Any]


class PrenatalDecisionAgent:
    """Agent 3: Clinical Decision Support & Genetic Testing Recommender."""

    def __init__(self, gene_associations_file: Optional[str] = None):
        self.gene_map: Dict[str, Set[str]] = defaultdict(set)
        if gene_associations_file and Path(gene_associations_file).is_file():
            self._load_genes(gene_associations_file)

    def _load_genes(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4 and parts[0] != "ncbi_gene_id":
                    gene = parts[1].strip().upper()
                    disease_id = parts[3].strip()
                    self.gene_map[gene].add(disease_id)

    def synthesize_pattern(
        self,
        observations: Sequence[Any],
    ) -> ClinicalPatternResult:
        """Synthesize clinical malformation pattern from HPO observations."""
        organs_seen: Set[str] = set()
        hpo_terms: List[str] = []

        for obs in observations:
            hid = getattr(obs, "hpo_id", None) or (obs.get("hpo_id") if isinstance(obs, dict) else str(obs))
            if not hid:
                continue
            hpo_terms.append(hid)
            organ = HPO_TO_ORGAN_FAST_MAP.get(hid)
            if organ:
                organs_seen.add(organ)

        organs_list = sorted(organs_seen)
        is_multiple = len(organs_list) >= 2 or len(hpo_terms) >= 3

        # Formulate clinical summary
        if not organs_list:
            if not hpo_terms:
                summary = "Chưa ghi nhận bất thường cấu trúc thai rõ ràng trên siêu âm."
                suspicion = "Chưa có chỉ định hội chứng"
            else:
                summary = f"Bất thường cấu trúc thai với {len(hpo_terms)} dấu hiệu HPO."
                suspicion = "Bất thường cấu trúc thai"
        elif len(organs_list) == 1:
            summary = f"Bất thường đơn hệ: {organs_list[0]}."
            suspicion = "Dị tật đơn hệ / Đơn gen"
        else:
            joined_organs = " + ".join(organs_list)
            summary = f"Đa dị tật phối hợp nhiều hệ cơ quan: {joined_organs}."
            suspicion = "Bất thường NST / Vi mất đoạn CNV hoặc Hội chứng đa dị tật"

        return ClinicalPatternResult(
            affected_organs=organs_list,
            is_multiple=is_multiple,
            pattern_summary=summary,
            primary_suspicion=suspicion,
        )

    def rerank_candidates(
        self,
        disease_candidates: Sequence[Any],
        pattern: ClinicalPatternResult,
        top_k: int = 5,
        observations: Optional[Sequence[Any]] = None,
    ) -> List[RerankedCandidate]:
        """Rerank disease candidates using prenatal epidemiology priors and clinical pattern compatibility."""
        observed_hpos: Set[str] = set()
        if observations:
            for obs in observations:
                hid = getattr(obs, "hpo_id", None) or (obs.get("hpo_id") if isinstance(obs, dict) else str(obs))
                if hid:
                    observed_hpos.add(hid)

        # Pre-check hallmark clinical indicator rules
        hallmark_matched_syndromes: List[Tuple[str, float]] = []
        for hid in observed_hpos:
            if hid in HALLMARK_INDICATOR_RULES:
                for syn_key, boost in HALLMARK_INDICATOR_RULES[hid]:
                    hallmark_matched_syndromes.append((syn_key, boost))

        reranked = []

        for original_rank, cand in enumerate(disease_candidates, 1):
            cand_id = getattr(cand, "disease_id", "") or cand.get("disease_id", "")
            cand_name = getattr(cand, "disease_name", "") or cand.get("disease_name", "")
            orig_score = getattr(cand, "score", 0.0) or cand.get("score", 0.0)
            orig_ic = getattr(cand, "ic_weighted_coverage", 0.0) or cand.get("ic_weighted_coverage", 0.0)

            cand_clean = cand_name.lower()
            bonus = 1.0
            rationale_parts = []
            test_rec = "CMA trên mẫu thai; xem xét Trio WES nếu CMA âm tính"

            # Check matching against prenatal syndrome profiles
            for syn_key, profile in PRENATAL_SYNDROME_CATALOG.items():
                is_this_syndrome = (
                    profile.canonical_id.lower() == cand_id.lower()
                    or any(w in cand_clean for w in profile.name.lower().split()[:2] if len(w) >= 5)
                )
                if is_this_syndrome:
                    bonus *= profile.base_prior_weight
                    # Apply hallmark indicator boost if present
                    for h_key, h_boost in hallmark_matched_syndromes:
                        if h_key == syn_key:
                            bonus *= h_boost
                            rationale_parts.append(f"Dấu hiệu chỉ điểm lâm sàng đặc hiệu tiền sản (boost x{h_boost:g})")
                    # Pattern compatibility bonus
                    pattern_overlap = sum(1 for p in profile.hallmark_patterns if any(p in org for org in pattern.affected_organs))
                    if pattern_overlap > 0:
                        bonus *= (1.0 + 0.3 * pattern_overlap)
                        rationale_parts.append(f"Khớp mô thức tiền sản ({pattern_overlap} hệ cơ quan tương thích)")
                    test_rec = f"{profile.recommended_first_tier}. {profile.recommended_second_tier}"
                    break

            # Penalize adult-only / non-prenatal obscure terms
            if any(term in cand_clean for term in [
                "adult", "late-onset", "susceptibility", "q fever", "fibrosis, retroperitoneal",
                "carcinoma", "adenoma", "melanoma", "glioma", "infection"
            ]):
                bonus *= 0.1
                rationale_parts.append("Hội chứng ít gặp/không điển hình ở giai đoạn trước sinh")

            final_score = orig_score * bonus + (orig_ic * 5.0 * bonus)
            rationale = "; ".join(rationale_parts) if rationale_parts else "Xếp hạng dựa trên độ tương đồng kiểu hình HPO"

            reranked.append({
                "cand_id": cand_id,
                "cand_name": cand_name,
                "original_rank": original_rank,
                "original_score": orig_score,
                "rerank_score": final_score,
                "rationale": rationale,
                "tests": test_rec,
            })

        # Sort by rerank_score descending
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        results = []
        for new_rank, item in enumerate(reranked[:top_k], 1):
            results.append(
                RerankedCandidate(
                    rank=new_rank,
                    disease_id=item["cand_id"],
                    disease_name=item["cand_name"],
                    original_rank=item["original_rank"],
                    original_score=item["original_score"],
                    rerank_score=round(item["rerank_score"], 2),
                    clinical_rationale=item["rationale"],
                    recommended_tests=item["tests"],
                )
            )

        return results

    def recommend_genetic_tests(
        self,
        pattern: ClinicalPatternResult,
        top_candidates: List[RerankedCandidate],
    ) -> str:
        """Formulate tiered genetic testing recommendations according to ACMG/ACOG guidelines."""
        recs = []

        # 1. First-tier recommendation
        has_aneuploidy_sign = any(
            "trisomy" in c.disease_name.lower() or "turner" in c.disease_name.lower()
            for c in top_candidates[:3]
        )
        if has_aneuploidy_sign or pattern.is_multiple:
            recs.append("• **Bậc 1 (First-tier):** QF-PCR nhanh dị bội (loại trừ Trisomy 13, 18, 21, monosomy X) kết hợp Karyotype hoặc CMA (Chromosomal Microarray) từ dịch ối.")
        else:
            recs.append("• **Bậc 1 (First-tier):** CMA (Chromosomal Microarray) trên mẫu thai là xét nghiệm nền tảng để khảo sát các biến thể số lượng bản sao (CNV).")

        # 2. Specific gene sequencing if pathognomonic
        targeted_genes = []
        for c in top_candidates[:3]:
            c_lower = c.disease_name.lower()
            if "tuberous sclerosis" in c_lower:
                targeted_genes.append("TSC1 / TSC2 (giải trình tự và del/dup; xét nghiệm bố mẹ kèm theo)")
            elif "cystic fibrosis" in c_lower:
                targeted_genes.append("CFTR (sàng lọc đột biến bố mẹ và thai)")
            elif "22q11" in c_lower or "digeorge" in c_lower:
                targeted_genes.append("Vùng vi mất đoạn 22q11.2 (khảo sát ưu tiên bằng CMA)")
            elif "noonan" in c_lower:
                targeted_genes.append("Panel RASopathy (PTPN11, SOS1, RAF1, RIT1...)")

        if targeted_genes:
            recs.append(f"• **Xét nghiệm gen mục tiêu:** {'; '.join(targeted_genes)}.")

        # 3. Second-tier recommendation
        recs.append("• **Bậc 2 (Second-tier):** Nếu CMA/karyotype âm tính nhưng thai có đa dị tật hoặc phenotype tiến triển: Đề xuất Trio WES (giải trình tự toàn bộ vùng mã hóa thai và bố mẹ) để tìm đột biến đơn gen.")

        # 4. Non-genetic differentiation if applicable
        if any("CNS" in org for org in pattern.affected_organs) and any("cyst" in c.disease_name.lower() for c in top_candidates):
            recs.append("• **Phân biệt ngoài di truyền:** Cân nhắc CMV-PCR từ dịch ối nếu có tổn thương hệ thần kinh trung ương/vôi hóa nghi ngờ nhiễm trùng bẩm sinh.")

        return "\n".join(recs)

    def analyze_case(
        self,
        case_id: str,
        observations: Sequence[Any],
        disease_candidates: Sequence[Any],
        top_k: int = 5,
    ) -> Agent3ClinicalReport:
        """Run complete Agent 3 pipeline: Pattern -> Reranked Syndromes -> Genetic Tests."""
        # Step 1: Pattern
        pattern = self.synthesize_pattern(observations)

        # Step 2: Rerank
        top_candidates = self.rerank_candidates(
            disease_candidates, pattern, top_k=top_k, observations=observations
        )

        # Step 3: Genetic Tests
        tests_text = self.recommend_genetic_tests(pattern, top_candidates)

        # Step 4: Format Syndromes text
        syndromes_lines = []
        for idx, cand in enumerate(top_candidates, 1):
            syndromes_lines.append(
                f"{idx}) **{cand.disease_name}** [{cand.disease_id}]: {cand.clinical_rationale}."
            )
        syndromes_text = "\n".join(syndromes_lines)

        # Step 5: Format Pattern text
        pattern_text = f"{pattern.pattern_summary} Hướng phân loại ưu tiên: {pattern.primary_suspicion}."

        structured_payload = {
            "case_id": case_id,
            "pattern": {
                "summary": pattern.pattern_summary,
                "affected_organs": pattern.affected_organs,
                "is_multiple": pattern.is_multiple,
                "primary_suspicion": pattern.primary_suspicion,
            },
            "reranked_syndromes": [
                {
                    "rank": c.rank,
                    "id": c.disease_id,
                    "name": c.disease_name,
                    "original_rank": c.original_rank,
                    "rerank_score": c.rerank_score,
                    "rationale": c.clinical_rationale,
                }
                for c in top_candidates
            ],
            "genetic_tests": tests_text,
        }

        return Agent3ClinicalReport(
            case_id=case_id,
            clinical_pattern=pattern_text,
            syndromes_text=syndromes_text,
            genetic_tests_text=tests_text,
            top_candidates=top_candidates,
            structured_payload=structured_payload,
        )
