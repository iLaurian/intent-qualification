import json
import os
import ast
import re
import numpy as np
import pandas as pd
import sys
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder


class LocalCompanyQualifierSystem:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.companies_df = self._load_dataset(data_path)

        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.reranker_model = CrossEncoder('BAAI/bge-reranker-large')

        print("Computing boosted company vector indices...")
        self.company_corpus = self._build_corpus()
        self.company_embeddings = self.embedding_model.encode(
            self.company_corpus,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def _load_dataset(self, path: str) -> pd.DataFrame:
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        df = pd.DataFrame(records)

        for col in ['address', 'primary_naics']:
            if col in df.columns:
                df[col] = df[col].apply(self._safe_eval_dict)
        return df

    def _safe_eval_dict(self, val: Any) -> Dict:
        if pd.isna(val) or val is None:
            return {}
        if isinstance(val, dict):
            return val
        try:
            if isinstance(val, str):
                return ast.literal_eval(val)
        except Exception:
            pass
        return {"raw_text": str(val)}

    def _build_corpus(self) -> List[str]:
        corpus = []
        for _, row in self.companies_df.iterrows():
            offerings = " | ".join(row.get('core_offerings', [])) if isinstance(row.get('core_offerings'), list) else ""
            markets = " | ".join(row.get('target_markets', [])) if isinstance(row.get('target_markets'), list) else ""
            model = " | ".join(row.get('business_model', [])) if isinstance(row.get('business_model'), list) else ""

            profile_str = (
                f"OFFERINGS: {offerings}. "
                f"MARKETS: {markets}. "
                f"MODEL: {model}. "
                f"DESC: {row.get('description', '')}"
            )
            corpus.append(profile_str)
        return corpus

    def _extract_query_constraints(self, query: str) -> Dict[str, Any]:
        constraints = {
            "min_revenue": None,
            "max_employee_count": None,
            "min_employee_count": None,
            "country_query": None,
            "semantic_query": query
        }

        clean_query = query.lower()

        money_pattern = r'(?:with revenue\s*)?(?:over|above|more than|>\s*)\s*\$?\s*([\d\.]+)\s*(billion|million|b|m)?'
        match_money = re.search(money_pattern, clean_query)
        if match_money:
            value = float(match_money.group(1))
            modifier = match_money.group(2)
            if modifier in ['billion', 'b']:
                value *= 1_000_000_000
            elif modifier in ['million', 'm']:
                value *= 1_000_000
            constraints["min_revenue"] = value
            clean_query = re.sub(money_pattern, '', clean_query)

        emp_less = re.search(r'(?:fewer than|less than|<\s*)\s*([\d,]+)\s*employee[s]?', clean_query)
        if emp_less:
            constraints["max_employee_count"] = int(emp_less.group(1).replace(',', ''))
            clean_query = re.sub(r'(?:fewer than|less than|<\s*)\s*([\d,]+)\s*employee[s]?', '', clean_query)

        emp_more = re.search(r'(?:more than|greater than|over|>\s*)\s*([\d,]+)\s*employee[s]?', clean_query)
        if emp_more:
            constraints["min_employee_count"] = int(emp_more.group(1).replace(',', ''))
            clean_query = re.sub(r'(?:more than|greater than|over|>\s*)\s*([\d,]+)\s*employee[s]?', '', clean_query)

        geo_map = {
            "united states": "us", "usa": "us", "germany": "de",
            "romania": "ro", "france": "fr", "switzerland": "ch"
        }
        for place, code in geo_map.items():
            if place in clean_query:
                constraints["country_query"] = code
                clean_query = clean_query.replace(f"in the {place}", "").replace(f"in {place}", "").replace(place, "")
                break

        clean_query = re.sub(r'\s+', ' ', clean_query).strip().strip('.')
        constraints["semantic_query"] = clean_query if clean_query else "companies"

        return constraints

    def _matches_constraints(self, company: Dict, constraints: Dict) -> bool:
        if constraints["min_revenue"] is not None:
            rev = company.get('revenue')
            if pd.notna(rev) and rev is not None:
                if float(rev) < constraints["min_revenue"]:
                    return False

        if constraints["max_employee_count"] is not None:
            emp = company.get('employee_count')
            if pd.notna(emp) and emp is not None:
                if int(emp) >= constraints["max_employee_count"]:
                    return False

        if constraints["min_employee_count"] is not None:
            emp = company.get('employee_count')
            if pd.notna(emp) and emp is not None:
                if int(emp) <= constraints["min_employee_count"]:
                    return False

        if constraints["country_query"] is not None:
            addr = company.get('address', {})
            country_code = str(addr.get('country_code', '')).lower() if isinstance(addr, dict) else ""
            addr_str = str(addr).lower()

            if country_code:
                if country_code != constraints["country_query"]:
                    return False
            else:
                if constraints["country_query"] not in addr_str:
                    return False

        return True

    def process_query(self, user_query: str) -> List[Dict[str, Any]]:
        constraints = self._extract_query_constraints(user_query)
        semantic_intent = constraints["semantic_query"]

        print(f"\n[Processing Query]: '{user_query}'")
        print(
            f"-> Hard Filters Extracted: Revenue > {constraints['min_revenue']}, Country: {constraints['country_query']}")
        print(f"-> AI Semantic Intent: '{semantic_intent}'")

        query_vector = self.embedding_model.encode([semantic_intent], convert_to_numpy=True)
        dot_product = np.dot(self.company_embeddings, query_vector.T).squeeze()

        top_k_indices = np.argsort(dot_product)[::-1][:200]
        candidates = self.companies_df.iloc[top_k_indices].to_dict(orient='records')

        filtered_candidates = [c for c in candidates if self._matches_constraints(c, constraints)]

        if not filtered_candidates:
            return []

        pairs = []
        for item in filtered_candidates:
            item_text = f"{item.get('description', '')} {item.get('core_offerings', [])}"
            pairs.append([semantic_intent, item_text])

        rerank_scores = self.reranker_model.predict(pairs)

        qualified_results = []
        for i, raw_score in enumerate(rerank_scores):
            confidence = float(1 / (1 + np.exp(-raw_score)))

            if confidence >= 0.60:
                item = filtered_candidates[i]
                item['qualification_confidence'] = confidence
                qualified_results.append(item)

        return sorted(qualified_results, key=lambda x: x['qualification_confidence'], reverse=True)


if __name__ == "__main__":
    DATA_FILE = "companies.jsonl"
    if os.path.exists(DATA_FILE):
        system = LocalCompanyQualifierSystem(DATA_FILE)

        for query in sys.stdin:
            query = query.strip()
            if not query:
                continue
            results = system.process_query(query)

            print(f"\nFound {len(results)} companies exceeding 60% qualification boundary:")
            for rank, comp in enumerate(results[:20], 1):
                print(f"{rank}. {comp['operational_name']} (Confidence: {comp['qualification_confidence'] * 100:.2f}%)")
                print(f"   Revenue: ${comp.get('revenue', 0):,.0f} | Location: {comp.get('address')}")
                print(f"   Offerings: {str(comp.get('core_offerings'))[:110]}...\n")
    else:
        print(f"Error: Could not find '{DATA_FILE}'. Please check the path.")