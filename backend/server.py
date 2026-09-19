"""Prenatal PubCaseFinder Standalone Web Server.

Integrates Model 1 (NER HPO Extraction), Model 2 (PubCaseFinder Phenotype Matcher),
and Model 3 (ACMG/ACOG Prenatal Decision & Genetic Test Recommender).
Modeled directly after the PubCaseFinder UI and Dr. Hung Vuong clinical protocols.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

# Ensure local package path resolution
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from hpo_agents.agent2_matcher import HPOAgent2Matcher
from hpo_agents.agent2_schema import PhenotypeObservation, PhenotypeStatus, RankingStrategy
from hpo_agents.agent3_prenatal_recommender import PrenatalDecisionAgent
from stage2_assertion.assertion_engine import ClinicalAssertionEngine

# ============================================================================
# INHERITANCE DICTIONARY (HPO Term -> Human Readable String)
# ============================================================================
INHERITANCE_NAMES: Dict[str, str] = {
    "HP:0000006": "Autosomal dominant inheritance (HP:0000006)",
    "HP:0000007": "Autosomal recessive inheritance (HP:0000007)",
    "HP:0001417": "X-linked inheritance (HP:0001417)",
    "HP:0001419": "X-linked recessive inheritance (HP:0001419)",
    "HP:0001423": "X-linked dominant inheritance (HP:0001423)",
    "HP:0001427": "Mitochondrial inheritance (HP:0001427)",
    "HP:0001428": "Somatic mutation / Mosaicism (HP:0001428)",
    "HP:0010985": "Gonosomal inheritance (HP:0010985)",
    "HP:0001425": "Heterogeneous inheritance (HP:0001425)",
    "HP:0003745": "Sporadic (HP:0003745)",
    "HP:0001426": "Multifactorial inheritance (HP:0001426)",
    "HP:0001442": "Polygenic inheritance (HP:0001442)",
}


def normalize_text(text: str) -> str:
    """Normalize Vietnamese text for resilient matching."""
    text = unicodedata.normalize("NFKC", str(text)).casefold().replace("đ", "d")
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9:+]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================================================================
# GLOBAL SYSTEM STATE & CACHE
# ============================================================================
class PubCaseFinderSystem:
    def __init__(self):
        self.matcher: Optional[HPOAgent2Matcher] = None
        self.decider: Optional[PrenatalDecisionAgent] = None
        self.disease_to_genes: Dict[str, List[str]] = defaultdict(list)
        self.hpo_catalog: List[Dict[str, Any]] = []
        self.hpo_id_to_term: Dict[str, Dict[str, str]] = {}
        self.sample_cases: List[Dict[str, Any]] = []
        self.search_index: List[Tuple[str, str, Dict[str, Any]]] = []
        self.assertion_engine = ClinicalAssertionEngine()

    def initialize(self):
        print("=" * 75)
        print("   KHỞI ĐỘNG PRENATAL PUBCASEFINDER BACKEND (3-AGENT PIPELINE)   ")
        print("=" * 75)
        t0 = time.time()

        data_dir = BASE_DIR / "data"
        hpoa_path = data_dir / "phenotype.hpoa"
        obo_path = data_dir / "hp.obo"
        genes_path = data_dir / "genes_to_disease.txt"
        catalog_path = data_dir / "hpo_catalog_vi.json"
        sample_cases_path = data_dir / "sample_cases.json"

        # 1. Load HPO Matcher (Model 2)
        print(f"[Model 2] Nạp cơ sở tri thức đồ thị bệnh hiếm từ {hpoa_path.name} & {obo_path.name}...")
        self.matcher = HPOAgent2Matcher.from_files(
            phenotype_hpoa=str(hpoa_path),
            hp_obo=str(obo_path),
        )
        print(f"[Model 2] Đã nạp thành công {len(self.matcher.profiles):,} hồ sơ bệnh OMIM/Orphanet.")

        # 2. Load Prenatal Decision Agent (Model 3) & Gene Map
        print(f"[Model 3] Nạp bảng tương tác gen - bệnh từ {genes_path.name}...")
        self.decider = PrenatalDecisionAgent(gene_associations_file=str(genes_path))

        # Build disease -> genes reverse lookup
        with open(genes_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4 and parts[0] != "ncbi_gene_id":
                    gene = parts[1].strip().upper()
                    dis_id = parts[3].strip()
                    if gene not in self.disease_to_genes[dis_id]:
                        self.disease_to_genes[dis_id].append(gene)
        print(f"[Model 3] Đã ánh xạ {len(self.disease_to_genes):,} bệnh với các gen gây bệnh tương ứng.")

        # 3. Load Vietnamese HPO Catalog (Model 1)
        doc_phrases_path = data_dir / "doctor_clinical_phrases.json"
        if doc_phrases_path.is_file():
            print(f"[Model 1] Nạp {doc_phrases_path.name} (Bộ từ vựng lâm sàng 234 ca bác sĩ)...")
            with open(doc_phrases_path, "r", encoding="utf-8") as f:
                doc_terms = json.load(f)
            for item in doc_terms:
                hid = item["id"]
                vi_name = item.get("vi", "")
                en_name = item.get("en", "")
                self.hpo_id_to_term[hid] = {"vi": vi_name, "en": en_name}
                norm_vi = normalize_text(vi_name)
                norm_en = normalize_text(en_name)
                self.search_index.append((norm_vi, norm_en, item))

        if catalog_path.is_file():
            print(f"[Model 1] Nạp từ điển chuẩn hóa HPO tiếng Việt từ {catalog_path.name}...")
            with open(catalog_path, "r", encoding="utf-8") as f:
                self.hpo_catalog = json.load(f)

            for item in self.hpo_catalog:
                hid = item["id"]
                vi_name = item.get("vi", "")
                en_name = item.get("en", "")
                if hid not in self.hpo_id_to_term:
                    self.hpo_id_to_term[hid] = {"vi": vi_name, "en": en_name}
                
                # Build fast normalized search index
                norm_vi = normalize_text(vi_name)
                norm_en = normalize_text(en_name)
                self.search_index.append((norm_vi, norm_en, item))

            # Sort longest terms first so more specific phrases match first
            self.search_index.sort(key=lambda x: max(len(x[0]), len(x[1])), reverse=True)
            print(f"[Model 1] Đã lập chỉ mục tìm kiếm {len(self.search_index):,} cụm từ HPO (ưu tiên cụm dài).")

        # 4. Load Sample Cases
        if sample_cases_path.is_file():
            with open(sample_cases_path, "r", encoding="utf-8") as f:
                self.sample_cases = json.load(f)
            print(f"[Hệ thống] Đã nạp {len(self.sample_cases)} ca lâm sàng mẫu (Gold 234 Benchmark).")

        print(f"--> Khởi động hoàn tất trong {time.time()-t0:.2f}s! Sẵn sàng phục vụ yêu cầu lâm sàng.\n")

    def search_hpo(self, query: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Instant search in HPO catalog by ID, Vietnamese or English name."""
        q_clean = query.strip()
        if not q_clean:
            return []

        # If direct HP:xxxxxxx ID
        if q_clean.upper().startswith("HP:"):
            exact_id = q_clean.upper()
            matches = [item for item in self.hpo_catalog if item["id"].startswith(exact_id)]
            if matches:
                return matches[:limit]

        q_norm = normalize_text(q_clean)
        results = []
        for norm_vi, norm_en, item in self.search_index:
            if q_norm in norm_vi or q_norm in norm_en:
                results.append(item)
                if len(results) >= limit:
                    break
        return results

    def extract_hpo_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Deterministic Model 1 extractor for Vietnamese ultrasound descriptions with Stage 2 Assertion."""
        norm_input = normalize_text(text)
        found_mentions: List[Dict[str, Any]] = []
        seen_hpos: Set[str] = set()

        # Search against indexed catalog terms (longest match first with word boundaries)
        for norm_vi, norm_en, item in self.search_index:
            hid = item["id"]
            if hid in seen_hpos:
                continue

            term_match = None
            if len(norm_vi) >= 3 and re.search(r'\b' + re.escape(norm_vi) + r'\b', norm_input):
                term_match = item.get("vi")
            elif len(norm_en) >= 4 and re.search(r'\b' + re.escape(norm_en) + r'\b', norm_input):
                term_match = item.get("en")

            if term_match:
                seen_hpos.add(hid)
                # Locate character span in original text
                idx = text.lower().find(term_match.lower())
                span_start = idx if idx != -1 else 0
                span_end = idx + len(term_match) if idx != -1 else len(term_match)

                # Stage 2 Clinical Assertion Engine (3 doctor keywords: 1 phần, theo dõi, nghi ngờ)
                status, reason = self.assertion_engine.predict(
                    text=text,
                    span_start=span_start,
                    span_end=span_end,
                    mention_text=term_match
                )

                found_mentions.append({
                    "hpo_id": hid,
                    "hpo_term": item.get("en", ""),
                    "doctor_phrase": term_match,
                    "mention_text": term_match,
                    "span_start": span_start,
                    "span_end": span_end,
                    "status": status,
                    "explanation": reason,
                })

        return found_mentions


GLOBAL_SYSTEM = PubCaseFinderSystem()


# ============================================================================
# HTTP REQUEST HANDLER
# ============================================================================
class PubCaseFinderHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR / "static"), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.path = "/index.html"
            return super().do_GET()

        if path == "/api/status":
            return self._send_json({
                "status": "online",
                "profiles_count": len(GLOBAL_SYSTEM.matcher.profiles) if GLOBAL_SYSTEM.matcher else 0,
                "hpo_terms_count": len(GLOBAL_SYSTEM.hpo_catalog),
                "genes_mapped_diseases": len(GLOBAL_SYSTEM.disease_to_genes),
            })

        if path == "/api/sample_cases":
            return self._send_json(GLOBAL_SYSTEM.sample_cases)

        if path == "/api/hpo_search":
            qs = parse_qs(parsed.query)
            q = qs.get("q", [""])[0]
            matches = GLOBAL_SYSTEM.search_hpo(q)
            return self._send_json(matches)

        # Fallback to static files
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            payload = json.loads(post_body.decode("utf-8"))
        except Exception:
            payload = {}

        if path == "/api/extract_hpo":
            text = payload.get("text", "")
            mentions = GLOBAL_SYSTEM.extract_hpo_from_text(text)
            return self._send_json({"mentions": mentions})

        if path == "/api/match_diseases":
            return self._handle_match_diseases(payload)

        self.send_error(404, "Endpoint not found")

    def _handle_match_diseases(self, payload: Dict[str, Any]):
        """PubCaseFinder Matching Engine."""
        hpos_input = payload.get("hpos", [])
        clinical_text = payload.get("clinical_text", "")

        if not hpos_input or not GLOBAL_SYSTEM.matcher:
            return self._send_json({"candidates": [], "clinical_pattern": ""})

        # 1. Prepare PhenotypeObservation list
        observations = []
        for item in hpos_input:
            hid = item.get("id") or item.get("hpo_id")
            if not hid:
                continue
            status_str = item.get("status", "CÓ")
            pstatus = PhenotypeStatus.SUSPECTED if status_str == "NGHI NGỜ" else PhenotypeStatus.PRESENT
            observations.append(PhenotypeObservation(hid, pstatus))

        # 2. Run Model 2 Matcher (IC_COVERAGE like PubCaseFinder)
        raw_candidates = GLOBAL_SYSTEM.matcher.rank(
            observations,
            top_k=20,
            ranking_strategy=RankingStrategy.IC_COVERAGE,
        )

        # 3. Run Model 3 Prenatal Decision Agent
        clinical_pattern_text = ""
        model3_report = None
        if GLOBAL_SYSTEM.decider:
            model3_report = GLOBAL_SYSTEM.decider.analyze_case(
                case_id="WEB_CASE",
                observations=observations,
                disease_candidates=raw_candidates,
                top_k=5,
            )
            clinical_pattern_text = model3_report.clinical_pattern

        # Build Model 3 lookup for top candidates
        m3_lookup = {}
        if model3_report:
            for rc in model3_report.top_candidates:
                m3_lookup[rc.disease_id] = {
                    "rationale": rc.clinical_rationale,
                    "recommended_tests": [t.strip() for t in rc.recommended_tests.split(";") if t.strip()],
                }

        # 4. Format Candidate Cards exactly like PubCaseFinder
        formatted_candidates = []
        for rank_idx, cand in enumerate(raw_candidates, 1):
            dis_id = cand.disease_id
            dis_name = cand.disease_name
            profile = GLOBAL_SYSTEM.matcher.profiles.get(dis_id)

            # Match percentage calculation (PubCaseFinder semantic coverage)
            # If all input findings match exactly -> 100%
            if cand.exact_coverage >= 0.999:
                match_pct = 100.0
            else:
                match_pct = round(min(1.0, cand.ic_weighted_coverage) * 100, 1)

            # Matched Phenotypes (Blue tags)
            matched_phenotypes = []
            matched_hpo_ids: Set[str] = set()
            for ev in cand.evidence:
                m_id = ev.matched_hpo_id or ev.observed_hpo_id
                if m_id and m_id not in matched_hpo_ids:
                    matched_hpo_ids.add(m_id)
                    term_info = GLOBAL_SYSTEM.hpo_id_to_term.get(m_id, {})
                    matched_phenotypes.append({
                        "hpo_id": m_id,
                        "name_vi": term_info.get("vi", ""),
                        "name_en": term_info.get("en", m_id),
                        "relation": ev.relation,
                    })

            # Modes of inheritance (Green tags)
            inheritance_modes = []
            if profile and profile.annotations:
                for a_hid, a_list in profile.annotations.items():
                    for ann in a_list:
                        if ann.aspect == "I":
                            inh_str = INHERITANCE_NAMES.get(a_hid, f"{a_hid} (Inheritance)")
                            if inh_str not in inheritance_modes:
                                inheritance_modes.append(inh_str)

            # Causative Genes (Grey tags)
            causative_genes = GLOBAL_SYSTEM.disease_to_genes.get(dis_id, [])

            # Clinical features unobserved (Amber tags for ultrasound re-check)
            clinical_features_to_check = []
            if profile and profile.positive_frequencies:
                # Sort unobserved terms by frequency descending
                unobserved_pairs = [
                    (hid, freq)
                    for hid, freq in profile.positive_frequencies.items()
                    if hid not in matched_hpo_ids and hid != "HP:0000118"
                ]
                unobserved_pairs.sort(key=lambda x: x[1], reverse=True)

                for u_hid, _ in unobserved_pairs[:25]:  # show top 25 features
                    t_info = GLOBAL_SYSTEM.hpo_id_to_term.get(u_hid, {})
                    clinical_features_to_check.append({
                        "hpo_id": u_hid,
                        "name_vi": t_info.get("vi", ""),
                        "name_en": t_info.get("en", u_hid),
                    })

            # Model 3 additions
            m3_info = m3_lookup.get(dis_id, {})
            model3_rationale = m3_info.get("rationale", "")
            model3_tests = m3_info.get("recommended_tests", [
                "CMA (Chromosomal Microarray) khảo sát vi mất/lặp đoạn",
                "Trio WES (Whole Exome Sequencing) nếu CMA âm tính"
            ])

            formatted_candidates.append({
                "rank": rank_idx,
                "disease_id": dis_id,
                "disease_name": dis_name,
                "match_percentage": match_pct,
                "ic_coverage": round(cand.ic_weighted_coverage, 3),
                "matched_phenotypes": matched_phenotypes,
                "inheritance_modes": inheritance_modes,
                "causative_genes": causative_genes,
                "clinical_features_to_check": clinical_features_to_check,
                "model3_rationale": model3_rationale,
                "model3_recommended_tests": model3_tests,
            })

        return self._send_json({
            "clinical_pattern": clinical_pattern_text,
            "candidates": formatted_candidates,
        })

    def _send_json(self, data: Any, status_code: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def run_server(port: int = 8000):
    GLOBAL_SYSTEM.initialize()
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, PubCaseFinderHandler)
    print("=" * 75)
    print(f"   SERVER ĐANG CHẠY TẠI: http://localhost:{port}/ hoặc http://127.0.0.1:{port}/")
    print("   Nhấn Ctrl+C để dừng server.")
    print("=" * 75)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang tắt server...")
        httpd.server_close()
        print("Đã tắt server an toàn.")


if __name__ == "__main__":
    port_arg = 8000
    if len(sys.argv) > 1:
        try:
            port_arg = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port=port_arg)
