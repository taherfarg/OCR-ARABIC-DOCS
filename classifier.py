"""
Document classifier - TWO methods combined:
1. Model's own document_type field from JSON (most reliable)
2. Keyword matching with ABSOLUTE thresholds (not percentage)
"""
import re
from typing import Dict, List, Tuple
from loguru import logger

import config


class DocumentClassifier:

    def __init__(self):
        self.document_types = config.DOCUMENT_TYPES
        logger.info(f"📋 Classifier: {len(self.document_types) - 1} types")

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ة', 'ه').replace('ى', 'ي')
        text = text.replace('ؤ', 'و').replace('ئ', 'ي')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _match_keywords(self, text: str, keywords: List[str]) -> Tuple[int, List[str]]:
        normalized = self._normalize(text)
        matched = []
        for kw in keywords:
            if self._normalize(kw) in normalized:
                matched.append(kw)
        return len(matched), matched

    def _classify_from_json(self, json_data: dict) -> Dict:
        """
        METHOD 1: Use the model's own document_type field.
        The model was TRAINED to identify document types!
        """
        if not json_data or not isinstance(json_data, dict):
            return None

        # Look for document_type field (any level)
        doc_type_value = self._find_field(json_data, [
            "document_type", "type", "document_title",
            "title", "form_type", "form_name",
            "نوع_المستند", "نوع_الوثيقة",
        ])

        if not doc_type_value:
            return None

        doc_type_str = str(doc_type_value).lower()
        logger.info(f"📋 Model says document type: {doc_type_value}")

        # Match against known types
        type_patterns = {
            "payment_voucher": [
                "payment voucher", "مستند الصرف", "مستند صرف",
                "سند دفع", "payment", "voucher", "صرف",
                "ap - payment", "ap-payment",
            ],
            "invoice": [
                "invoice", "فاتورة", "فاتوره",
                "tax invoice", "فاتورة ضريبية",
            ],
            "purchase_order": [
                "purchase order", "أمر شراء", "امر شراء",
                "po ", "طلب شراء",
            ],
            "contract": [
                "contract", "عقد", "agreement", "اتفاقية", "اتفاقيه",
            ],
            "letter": [
                "letter", "خطاب", "رسالة", "رساله", "مراسلة",
            ],
            "receipt": [
                "receipt", "إيصال", "ايصال", "سند قبض",
            ],
            "legal_document": [
                "نظام", "قانون", "لائحة", "مرسوم",
                "regulation", "decree", "law",
            ],
            "bank_statement": [
                "bank statement", "كشف حساب",
                "statement", "كشف",
            ],
        }

        for doc_type, patterns in type_patterns.items():
            for pattern in patterns:
                if pattern in self._normalize(doc_type_str):
                    info = self.document_types.get(doc_type, {})
                    return {
                        "document_type": doc_type,
                        "name_ar": info.get("name_ar", ""),
                        "name_en": info.get("name_en", ""),
                        "confidence": 95.0,  # High - model identified it
                        "method": "model_json_field",
                        "model_said": doc_type_value,
                    }

        return None

    def _find_field(self, data: dict, field_names: list) -> str:
        """Recursively find a field in nested dict."""
        if not isinstance(data, dict):
            return None

        for key, value in data.items():
            key_lower = key.lower().replace(" ", "_")
            for name in field_names:
                if name in key_lower:
                    if isinstance(value, str) and len(value.strip()) > 2:
                        return value.strip()

            # Recurse into nested dicts
            if isinstance(value, dict):
                result = self._find_field(value, field_names)
                if result:
                    return result

        return None

    def _classify_from_keywords(self, text: str) -> Dict:
        """
        METHOD 2: Keyword matching with ABSOLUTE thresholds.
        Not percentage-based anymore!
        """
        scores = {}
        details = {}

        for doc_type, info in self.document_types.items():
            if doc_type == "unknown":
                continue
            keywords = info["keywords"]
            if not keywords:
                continue

            count, matched = self._match_keywords(text, keywords)

            # ===== NEW SCORING: Absolute match count =====
            # 1-2 matches = low confidence
            # 3-5 matches = medium confidence
            # 6-10 matches = high confidence
            # 10+ matches = very high confidence
            if count >= 10:
                score = 90.0 + min(count - 10, 10)  # 90-100
            elif count >= 6:
                score = 70.0 + (count - 6) * 5       # 70-90
            elif count >= 3:
                score = 40.0 + (count - 3) * 10       # 40-70
            elif count >= 1:
                score = 10.0 + (count - 1) * 15       # 10-40
            else:
                score = 0.0

            score *= info.get("weight", 1.0)

            scores[doc_type] = score
            details[doc_type] = {
                "score": round(score, 2),
                "matches": count,
                "matched": matched,
            }

        if scores:
            best = max(scores, key=scores.get)
            best_score = scores[best]
            best_info = self.document_types[best]
            best_detail = details[best]
        else:
            best = "unknown"
            best_score = 0
            best_info = self.document_types["unknown"]
            best_detail = {"matched": [], "matches": 0}

        if best_score < config.MIN_CONFIDENCE:
            best = "unknown"
            best_info = self.document_types["unknown"]

        return {
            "document_type": best,
            "name_ar": best_info["name_ar"],
            "name_en": best_info["name_en"],
            "confidence": round(best_score, 2),
            "matched_keywords": best_detail.get("matched", []),
            "keyword_matches": best_detail.get("matches", 0),
            "method": "keyword_matching",
            "all_scores": {
                k: {"score": v["score"], "matches": v["matches"]}
                for k, v in sorted(details.items(), key=lambda x: -x[1]["score"])
            },
        }

    def classify(self, ocr_text: str, json_data: dict = None) -> Dict:
        """
        Combined classification using both methods.
        Priority: Model JSON field > Keyword matching
        """
        # METHOD 1: Try model's own document_type field first
        if json_data:
            json_result = self._classify_from_json(json_data)
            if json_result:
                logger.info(
                    f"📋 [JSON Method] {json_result['document_type']} "
                    f"({json_result['name_en']}) | "
                    f"Confidence: {json_result['confidence']}%"
                )

                # Also run keyword matching for extra info
                kw_result = self._classify_from_keywords(ocr_text)

                # Merge results
                json_result["keyword_matches"] = kw_result.get("keyword_matches", 0)
                json_result["matched_keywords"] = kw_result.get("matched_keywords", [])
                json_result["all_scores"] = kw_result.get("all_scores", {})
                json_result["method"] = "model_json + keywords"

                return json_result

        # METHOD 2: Fall back to keyword matching
        kw_result = self._classify_from_keywords(ocr_text)
        logger.info(
            f"📋 [Keyword Method] {kw_result['document_type']} "
            f"({kw_result['name_en']}) | "
            f"Confidence: {kw_result['confidence']}%"
        )
        return kw_result