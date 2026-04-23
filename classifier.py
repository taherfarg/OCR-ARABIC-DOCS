"""
Document classifier — THREE methods combined for robust classification:
1. Model's own document_type field from JSON (most reliable when available)
2. Keyword matching with weighted scoring (handles missing JSON fields)
3. Structural pattern matching (detects document layout patterns)

Priority: Model JSON field > Keyword matching > Structural patterns
"""
import re
from typing import Dict, List, Tuple, Optional
from loguru import logger

import config


class DocumentClassifier:

    def __init__(self):
        self.document_types = config.DOCUMENT_TYPES
        logger.info(f"📋 Classifier: {len(self.document_types) - 1} types")

    def _normalize(self, text: str) -> str:
        """Normalize Arabic text for matching — strips diacritics, unifies variants."""
        text = text.lower()
        # Remove Arabic diacritics (tashkeel)
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
        # Normalize alef variants
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        # Normalize taa marbuta and other variants
        text = text.replace('ة', 'ه').replace('ى', 'ي')
        text = text.replace('ؤ', 'و').replace('ئ', 'ي')
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _match_keywords(self, text: str, keywords: List[str]) -> Tuple[int, List[str], List[float]]:
        """
        Match keywords against text with weighted scoring.
        Returns (total_weighted_score, matched_keywords, individual_scores).
        """
        normalized = self._normalize(text)
        matched = []
        scores = []

        for kw in keywords:
            kw_norm = self._normalize(kw)
            if kw_norm in normalized:
                matched.append(kw)
                # Weight: longer keywords are more specific and reliable
                # Multi-word keywords get bonus
                word_count = len(kw_norm.split())
                if word_count >= 3:
                    score = 3.0  # Very specific multi-word match
                elif word_count == 2:
                    score = 2.0  # Two-word match
                else:
                    score = 1.0  # Single word match
                scores.append(score)

        total_score = sum(scores)
        return total_score, matched, scores

    def _classify_from_json(self, json_data: dict) -> Optional[Dict]:
        """
        METHOD 1: Use the model's own document_type field.
        The model was TRAINED to identify document types — trust it when confident.
        """
        if not json_data or not isinstance(json_data, dict):
            return None

        # Look for document_type field (any level, multiple possible names)
        doc_type_value = self._find_field(json_data, [
            "document_type", "type", "document_title",
            "title", "form_type", "form_name",
            "نوع_المستند", "نوع_الوثيقة", "نوع المستند",
        ])

        if not doc_type_value:
            return None

        doc_type_str = str(doc_type_value).lower()
        logger.info(f"📋 Model says document type: {doc_type_value}")

        # Match against known types with expanded patterns
        type_patterns = {
            "payment_voucher": [
                "payment voucher", "مستند الصرف", "مستند صرف",
                "سند دفع", "payment", "voucher", "صرف",
                "ap - payment", "ap-payment", "مستندصرف",
                "سند صرف", "مستند صرف / سند دفع",
            ],
            "invoice": [
                "invoice", "فاتورة", "فاتوره",
                "tax invoice", "فاتورة ضريبية", "فاتوره ضريبيه",
                "bill", "فاتورة ضريبة",
            ],
            "purchase_order": [
                "purchase order", "أمر شراء", "امر شراء",
                "po ", "طلب شراء", "أمر شراء",
            ],
            "contract": [
                "contract", "عقد", "agreement", "اتفاقية", "اتفاقيه",
                "ميثاق", "معاهدة",
            ],
            "letter": [
                "letter", "خطاب", "رسالة", "رساله", "مراسلة",
                "مذكرة", "memorandum",
            ],
            "receipt": [
                "receipt", "إيصال", "ايصال", "سند قبض",
                "quittance",
            ],
            "legal_document": [
                "نظام", "قانون", "لائحة", "مرسوم",
                "regulation", "decree", "law", "statute",
                "تشريع", "نظام قانوني",
            ],
            "bank_statement": [
                "bank statement", "كشف حساب", "كشف حساب بنكي",
                "statement", "كشف", "account statement",
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
                        "confidence": 95.0,
                        "method": "model_json_field",
                        "model_said": doc_type_value,
                    }

        # Model gave a type we don't recognize — still useful info
        return {
            "document_type": "unknown",
            "name_ar": "غير مصنف",
            "name_en": "Unknown",
            "confidence": 30.0,
            "method": "model_json_field_unrecognized",
            "model_said": doc_type_value,
        }

    def _find_field(self, data: dict, field_names: list) -> Optional[str]:
        """Recursively find a field in nested dict."""
        if not isinstance(data, dict):
            return None

        for key, value in data.items():
            key_lower = key.lower().replace(" ", "_")
            for name in field_names:
                if name in key_lower:
                    if isinstance(value, str) and len(value.strip()) > 2:
                        return value.strip()
                    elif isinstance(value, list) and value:
                        return str(value[0])

            # Recurse into nested dicts
            if isinstance(value, dict):
                result = self._find_field(value, field_names)
                if result:
                    return result

        return None

    def _classify_from_keywords(self, text: str) -> Dict:
        """
        METHOD 2: Keyword matching with weighted scoring.
        Uses weighted keyword counts instead of raw counts.
        Multi-word keywords score higher (more specific).
        """
        scores = {}
        details = {}

        for doc_type, info in self.document_types.items():
            if doc_type == "unknown":
                continue
            keywords = info["keywords"]
            if not keywords:
                continue

            weighted_score, matched, individual_scores = self._match_keywords(text, keywords)
            raw_count = len(matched)

            # Scoring based on weighted matches
            # weighted_score accounts for keyword specificity
            if weighted_score >= 20:
                score = 92.0 + min(weighted_score - 20, 10) * 0.8  # 92-100
            elif weighted_score >= 12:
                score = 75.0 + (weighted_score - 12) * (17 / 8)     # 75-92
            elif weighted_score >= 6:
                score = 45.0 + (weighted_score - 6) * (30 / 6)      # 45-75
            elif weighted_score >= 3:
                score = 20.0 + (weighted_score - 3) * (25 / 3)      # 20-45
            elif weighted_score >= 1:
                score = 5.0 + (weighted_score - 1) * 15              # 5-20
            else:
                score = 0.0

            score *= info.get("weight", 1.0)

            scores[doc_type] = score
            details[doc_type] = {
                "score": round(score, 2),
                "weighted_matches": round(weighted_score, 1),
                "raw_matches": raw_count,
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
            best_detail = {"matched": [], "raw_matches": 0, "weighted_matches": 0}

        if best_score < config.MIN_CONFIDENCE:
            best = "unknown"
            best_info = self.document_types["unknown"]

        return {
            "document_type": best,
            "name_ar": best_info["name_ar"],
            "name_en": best_info["name_en"],
            "confidence": round(best_score, 2),
            "matched_keywords": best_detail.get("matched", []),
            "keyword_matches": best_detail.get("raw_matches", 0),
            "weighted_matches": best_detail.get("weighted_matches", 0),
            "method": "keyword_matching",
            "all_scores": {
                k: {
                    "score": v["score"],
                    "matches": v["raw_matches"],
                    "weighted": v["weighted_matches"],
                }
                for k, v in sorted(details.items(), key=lambda x: -x[1]["score"])
            },
        }

    def _classify_from_structure(self, text: str) -> Optional[Dict]:
        """
        METHOD 3: Structural pattern matching.
        Detects document type from layout patterns, not just keywords.
        """
        normalized = self._normalize(text)
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        # Payment voucher: typically has debit/credit columns, supplier info, bank details
        if self._has_structure_payment_voucher(normalized, lines):
            info = self.document_types.get("payment_voucher", {})
            return {
                "document_type": "payment_voucher",
                "name_ar": info.get("name_ar", ""),
                "name_en": info.get("name_en", ""),
                "confidence": 55.0,
                "method": "structural_pattern",
            }

        # Invoice: typically has item lines with prices, totals, VAT
        if self._has_structure_invoice(normalized, lines):
            info = self.document_types.get("invoice", {})
            return {
                "document_type": "invoice",
                "name_ar": info.get("name_ar", ""),
                "name_en": info.get("name_en", ""),
                "confidence": 50.0,
                "method": "structural_pattern",
            }

        # Legal document: articles, chapters, numbered provisions
        if self._has_structure_legal(normalized, lines):
            info = self.document_types.get("legal_document", {})
            return {
                "document_type": "legal_document",
                "name_ar": info.get("name_ar", ""),
                "name_en": info.get("name_en", ""),
                "confidence": 55.0,
                "method": "structural_pattern",
            }

        return None

    def _has_structure_payment_voucher(self, normalized: str, lines: List[str]) -> bool:
        """Detect payment voucher structure."""
        indicators = 0
        # Has debit/credit pattern
        if re.search(r'(debit|credit|مدين|دائن)', normalized):
            indicators += 2
        # Has supplier pattern
        if re.search(r'(supplier|vendor|مورد|المورد)', normalized):
            indicators += 2
        # Has bank pattern
        if re.search(r'(bank|بنك|البنك)', normalized):
            indicators += 1
        # Has payment method
        if re.search(r'(cheque|check|wire|transfer|شيك|تحويل)', normalized):
            indicators += 1
        # Has PV/BC number pattern
        if re.search(r'(pv\s*no|bc\s*no)', normalized):
            indicators += 2
        return indicators >= 4

    def _has_structure_invoice(self, normalized: str, lines: List[str]) -> bool:
        """Detect invoice structure."""
        indicators = 0
        # Has quantity/price pattern
        if re.search(r'(quantity|unit\s*price|الكمية|سعر|السعر)', normalized):
            indicators += 2
        # Has subtotal/total/VAT
        if re.search(r'(subtotal|vat|total|الإجمالي|الضريبة|المجموع)', normalized):
            indicators += 2
        # Has bill-to pattern
        if re.search(r'(bill\s*to|ship\s*to|فاتورة|المستلم)', normalized):
            indicators += 1
        # Has item lines with amounts
        amount_lines = sum(1 for l in lines if re.search(r'\d+[\.,]\d{2}', l))
        if amount_lines >= 3:
            indicators += 1
        return indicators >= 4

    def _has_structure_legal(self, normalized: str, lines: List[str]) -> bool:
        """Detect legal document structure."""
        indicators = 0
        # Has article numbering (المادة الأولى, المادة 2, etc.)
        if re.search(r'(المادة|ماده)\s*(الأولى|الثانية|الثالثة|\d+)', normalized):
            indicators += 3
        # Has chapter/section structure
        if re.search(r'(باب|فصل|قسم|chapter|section)', normalized):
            indicators += 2
        # Has legal authority references
        if re.search(r'(مجلس الوزراء|هيئة الخبراء|مرسوم ملكي|نظام)', normalized):
            indicators += 2
        # Has numbered provisions
        numbered_lines = sum(1 for l in lines if re.match(r'^\s*\d+\s*[-–.]', l))
        if numbered_lines >= 3:
            indicators += 1
        return indicators >= 4

    def classify(self, ocr_text: str, json_data: dict = None) -> Dict:
        """
        Combined classification using all three methods.
        Priority: Model JSON field > Keyword matching > Structural patterns

        When multiple methods agree, confidence is boosted.
        When they disagree, the highest-confidence method wins.
        """
        results = []

        # METHOD 1: Try model's own document_type field first
        if json_data:
            json_result = self._classify_from_json(json_data)
            if json_result and json_result["document_type"] != "unknown":
                results.append(json_result)
                logger.info(
                    f"📋 [JSON Method] {json_result['document_type']} "
                    f"({json_result['name_en']}) | "
                    f"Confidence: {json_result['confidence']}%"
                )

        # METHOD 2: Keyword matching (always run for extra info)
        kw_result = self._classify_from_keywords(ocr_text)
        if kw_result["document_type"] != "unknown":
            results.append(kw_result)
        logger.info(
            f"📋 [Keyword Method] {kw_result['document_type']} "
            f"({kw_result['name_en']}) | "
            f"Confidence: {kw_result['confidence']}%"
        )

        # METHOD 3: Structural pattern matching
        struct_result = self._classify_from_structure(ocr_text)
        if struct_result:
            results.append(struct_result)
            logger.info(
                f"📋 [Structural Method] {struct_result['document_type']} "
                f"({struct_result['name_en']}) | "
                f"Confidence: {struct_result['confidence']}%"
            )

        # ===== COMBINE RESULTS =====
        if not results:
            # All methods failed
            return {
                "document_type": "unknown",
                "name_ar": "غير مصنف",
                "name_en": "Unknown",
                "confidence": 0.0,
                "method": "no_match",
                "matched_keywords": [],
                "keyword_matches": 0,
            }

        if len(results) == 1:
            # Only one method produced a result
            result = results[0]
            # Enrich with keyword info if available
            result["keyword_matches"] = kw_result.get("keyword_matches", 0)
            result["matched_keywords"] = kw_result.get("matched_keywords", [])
            result["all_scores"] = kw_result.get("all_scores", {})
            return result

        # Multiple methods — check for agreement
        type_votes = {}
        for r in results:
            dt = r["document_type"]
            if dt not in type_votes:
                type_votes[dt] = []
            type_votes[dt].append(r)

        # If multiple methods agree on the same type → boost confidence
        best_type = max(type_votes.keys(), key=lambda dt: len(type_votes[dt]))
        votes = type_votes[best_type]

        if len(votes) >= 2:
            # Agreement! Use the highest-confidence result and boost it
            best_result = max(votes, key=lambda r: r["confidence"])
            boost = 1.0 + 0.1 * (len(votes) - 1)  # 10% boost per agreeing method
            best_result["confidence"] = min(best_result["confidence"] * boost, 100.0)
            best_result["method"] = " + ".join(r.get("method", "") for r in votes)
            logger.info(
                f"📋 [Agreement] {best_type} — {len(votes)} methods agree | "
                f"Boosted confidence: {best_result['confidence']:.1f}%"
            )
        else:
            # Disagreement — pick highest confidence
            best_result = max(results, key=lambda r: r["confidence"])
            logger.info(
                f"📋 [Disagreement] Picked {best_result['document_type']} "
                f"via {best_result.get('method', '')} | "
                f"Confidence: {best_result['confidence']:.1f}%"
            )

        # Enrich with keyword info
        best_result["keyword_matches"] = kw_result.get("keyword_matches", 0)
        best_result["matched_keywords"] = kw_result.get("matched_keywords", [])
        best_result["all_scores"] = kw_result.get("all_scores", {})

        return best_result
