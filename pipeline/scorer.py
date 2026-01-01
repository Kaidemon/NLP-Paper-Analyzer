"""
Paper quality scoring system for Phase 8.
Evaluates papers on 5 criteria: structure, order, clarity, grammar, consistency.
"""

from typing import Dict, List, Tuple

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.utils import logger


class PaperQualityScorer:
    """Comprehensive scoring system for academic paper quality assessment."""
    
    # Expected sections in a well-structured academic paper (in ideal order)
    EXPECTED_SECTIONS = Config.EXPECTED_SECTIONS
    
    # Weights for each scoring criterion
    DEFAULT_WEIGHTS = Config.SCORING_WEIGHTS
    
    def __init__(self, weights: Dict = None):
        """
        Initialize scorer with optional custom weights.
        
        Args:
            weights: Custom weights for scoring criteria
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.scores = {}
        self.details = {}
    
    def score_structure(self, sections: List[Dict]) -> Tuple[float, Dict]:
        """
        Score 1: Structure Score (out of 10).
        Measures how well the paper covers expected academic sections.
        """
        if not sections:
            return 0.0, "No sections found"
        
        predicted_labels = [s.get('predicted_label', 'Unknown') for s in sections]
        unique_labels = set(predicted_labels)
        
        # Check for essential sections
        essential_sections = {'Introduction', 'Methodology', 'Experiments', 'Conclusion'}
        found_essential = essential_sections.intersection(unique_labels)
        essential_score = len(found_essential) / len(essential_sections) * 5
        
        # Check for optional sections
        optional_sections = {'Related Work', 'Appendix'}
        found_optional = optional_sections.intersection(unique_labels)
        optional_score = len(found_optional) / len(optional_sections) * 2
        
        # Section count penalty (ideal: 5-10 sections)
        num_sections = len(sections)
        if 5 <= num_sections <= 10:
            count_score = 2.0
        elif 3 <= num_sections < 5 or 10 < num_sections <= 15:
            count_score = 1.5
        elif num_sections < 3:
            count_score = 0.5
        else:
            count_score = 1.0
        
        # Diversity bonus
        diversity_score = min(len(unique_labels) / 4, 1.0)
        
        total_score = min(essential_score + optional_score + count_score + diversity_score, 10.0)
        
        details = {
            'essential_found': list(found_essential),
            'essential_missing': list(essential_sections - found_essential),
            'optional_found': list(found_optional),
            'num_sections': num_sections,
            'unique_types': list(unique_labels)
        }
        
        return round(total_score, 2), details
    
    def score_section_order(self, sections: List[Dict]) -> Tuple[float, Dict]:
        """
        Score 2: Section Order Score (out of 10).
        Measures if sections follow logical academic paper structure.
        """
        if not sections:
            return 0.0, "No sections found"
        
        predicted_labels = [s.get('predicted_label', 'Unknown') for s in sections]
        
        # Create order mapping
        order_map = {label: idx for idx, label in enumerate(self.EXPECTED_SECTIONS)}
        
        # Get indices of sections that appear in expected order
        section_indices = []
        for label in predicted_labels:
            if label in order_map:
                section_indices.append(order_map[label])
        
        if len(section_indices) < 2:
            return 5.0, {"note": "Too few classifiable sections to evaluate order"}
        
        # Count inversions (pairs that are out of order)
        inversions = 0
        total_pairs = 0
        for i in range(len(section_indices)):
            for j in range(i + 1, len(section_indices)):
                total_pairs += 1
                if section_indices[i] > section_indices[j]:
                    inversions += 1
        
        # Calculate order score
        if total_pairs > 0:
            order_ratio = 1 - (inversions / total_pairs)
            score = order_ratio * 10
        else:
            score = 5.0
        
        # Check if Introduction is first and Conclusion is last
        bonus = 0
        if section_indices and section_indices[0] == order_map.get('Introduction', -1):
            bonus += 0.5
        if section_indices and section_indices[-1] in [order_map.get('Conclusion', -1), order_map.get('Appendix', -1)]:
            bonus += 0.5
        
        total_score = min(score + bonus, 10.0)
        
        details = {
            'inversions': inversions,
            'total_pairs': total_pairs,
            'order_sequence': predicted_labels,
            'starts_with_intro': predicted_labels[0] == 'Introduction' if predicted_labels else False,
            'ends_with_conclusion': predicted_labels[-1] in ['Conclusion', 'Appendix'] if predicted_labels else False
        }
        
        return round(total_score, 2), details
    
    def score_classification_confidence(self, sections: List[Dict]) -> Tuple[float, Dict]:
        """
        Score 3: Classification Confidence Score (out of 10).
        Measures how confidently sections were classified.
        """
        if not sections:
            return 0.0, "No sections found"
        
        confidences = [s.get('confidence', 0) for s in sections]
        
        if not confidences:
            return 5.0, {"note": "No confidence scores available"}
        
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        # Average confidence contributes 7 points
        avg_score = avg_confidence * 7
        
        # Consistency bonus (small variance = clearer writing)
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        consistency_bonus = max(0, 2 - variance * 10)
        
        # High minimum confidence bonus
        min_bonus = min_confidence * 1
        
        total_score = min(avg_score + consistency_bonus + min_bonus, 10.0)
        
        details = {
            'average_confidence': round(avg_confidence, 4),
            'min_confidence': round(min_confidence, 4),
            'max_confidence': round(max_confidence, 4),
            'variance': round(variance, 4)
        }
        
        return round(total_score, 2), details
    
    def score_grammar_quality(self, sections: List[Dict]) -> Tuple[float, Dict]:
        """
        Score 4: Grammar Quality Score (out of 10).
        Measures writing quality based on grammar correction analysis.
        """
        if not sections:
            return 0.0, "No sections found"
        
        total_chars = 0
        total_changes = 0
        section_scores = []
        
        for section in sections:
            raw_text = section.get('raw_text', section.get('text', ''))
            corrected_text = section.get('corrected_text', raw_text)
            
            if not raw_text:
                continue
            
            raw_len = len(raw_text)
            
            # Calculate change ratio
            if raw_text == corrected_text:
                change_ratio = 0.0
            else:
                # Count word-level changes
                raw_words = set(raw_text.lower().split())
                corrected_words = set(corrected_text.lower().split())
                
                added = len(corrected_words - raw_words)
                removed = len(raw_words - corrected_words)
                total_unique = len(raw_words.union(corrected_words))
                
                change_ratio = (added + removed) / max(total_unique, 1)
            
            # Convert to section score (less change = higher score)
            section_score = max(0, 10 - change_ratio * 20)
            section_scores.append(section_score)
            
            total_chars += raw_len
            total_changes += change_ratio
        
        if not section_scores:
            return 5.0, {"note": "No text available for grammar analysis"}
        
        # Average section scores
        avg_score = sum(section_scores) / len(section_scores)
        
        # Consistency bonus
        variance = sum((s - avg_score) ** 2 for s in section_scores) / len(section_scores)
        consistency_bonus = max(0, 1 - variance / 10)
        
        total_score = min(avg_score + consistency_bonus, 10.0)
        
        details = {
            'sections_analyzed': len(section_scores),
            'average_section_score': round(avg_score, 2),
            'total_chars_analyzed': total_chars,
            'average_change_ratio': round(total_changes / len(section_scores), 4) if section_scores else 0
        }
        
        return round(total_score, 2), details
    
    def score_consistency(self, fact_check_results: List[Dict]) -> Tuple[float, Dict]:
        """
        Score 5: Consistency Score (out of 10).
        Measures internal consistency based on fact-checking results.
        """
        if not fact_check_results:
            return 5.0, {"note": "No fact-check results available"}
        
        supports = 0
        refutes = 0
        not_enough_info = 0
        
        for result in fact_check_results:
            verdict = result.get('verdict', '').lower()
            if 'support' in verdict:
                supports += 1
            elif 'refute' in verdict:
                refutes += 1
            else:
                not_enough_info += 1
        
        total = supports + refutes + not_enough_info
        
        if total == 0:
            return 5.0, {"note": "No claims verified"}
        
        # Base score from support ratio
        support_ratio = supports / total
        base_score = support_ratio * 8
        
        # Penalty for refutations
        refute_penalty = (refutes / total) * 4
        
        # Small penalty for unclear claims
        unclear_penalty = (not_enough_info / total) * 1
        
        # Bonus for having many verified claims
        verification_bonus = min(total / 10, 2)
        
        total_score = max(0, min(base_score - refute_penalty - unclear_penalty + verification_bonus, 10.0))
        
        details = {
            'total_claims': total,
            'supports': supports,
            'refutes': refutes,
            'not_enough_info': not_enough_info,
            'support_ratio': round(support_ratio, 4)
        }
        
        return round(total_score, 2), details
    
    def calculate_all_scores(self, pipeline_results: Dict) -> Tuple[Dict, Dict]:
        """
        Calculate all quality scores from pipeline results.
        
        Args:
            pipeline_results: Results from UnifiedDocumentPipeline
            
        Returns:
            Tuple of (scores, details)
        """
        sections = pipeline_results.get('sections', [])
        fact_results = pipeline_results.get('fact_check_results', [])
        
        # Calculate individual scores
        self.scores['structure'], self.details['structure'] = self.score_structure(sections)
        self.scores['section_order'], self.details['section_order'] = self.score_section_order(sections)
        self.scores['classification_confidence'], self.details['classification_confidence'] = self.score_classification_confidence(sections)
        self.scores['grammar_quality'], self.details['grammar_quality'] = self.score_grammar_quality(sections)
        self.scores['consistency'], self.details['consistency'] = self.score_consistency(fact_results)
        
        # Calculate weighted final score
        total_weight = sum(self.weights.values())
        weighted_sum = sum(self.scores[key] * self.weights[key] for key in self.scores)
        self.scores['final'] = round(weighted_sum / total_weight, 2)
        
        logger.info(f"✓ Quality scores calculated. Final score: {self.scores['final']}/10")
        
        return self.scores, self.details
    
    def get_grade(self, score: float) -> Tuple[str, str]:
        """Convert numerical score to letter grade."""
        if score >= 9.0:
            return 'A+', 'Excellent'
        elif score >= 8.0:
            return 'A', 'Very Good'
        elif score >= 7.0:
            return 'B+', 'Good'
        elif score >= 6.0:
            return 'B', 'Above Average'
        elif score >= 5.0:
            return 'C', 'Average'
        elif score >= 4.0:
            return 'D', 'Below Average'
        else:
            return 'F', 'Needs Improvement'
    
    def generate_report(self) -> str:
        """Generate a comprehensive quality report."""
        if not self.scores:
            return "No scores calculated yet. Run calculate_all_scores() first."
        
        report = []
        report.append("\n" + "█" * 70)
        report.append("           PAPER QUALITY ASSESSMENT REPORT")
        report.append("█" * 70)
        
        # Individual Scores
        report.append("\n📊 INDIVIDUAL SCORES (out of 10):")
        report.append("-" * 50)
        
        score_names = {
            'structure': '1. Structure & Completeness',
            'section_order': '2. Section Order & Flow',
            'classification_confidence': '3. Section Clarity',
            'grammar_quality': '4. Grammar & Writing Quality',
            'consistency': '5. Internal Consistency'
        }
        
        for key, name in score_names.items():
            score = self.scores.get(key, 0)
            grade, desc = self.get_grade(score)
            bar = '█' * int(score) + '░' * (10 - int(score))
            report.append(f"   {name}")
            report.append(f"   [{bar}] {score}/10 ({grade} - {desc})")
            report.append("")
        
        # Final Score
        final_score = self.scores.get('final', 0)
        final_grade, final_desc = self.get_grade(final_score)
        
        report.append("=" * 50)
        report.append(f"\n🏆 FINAL SCORE: {final_score}/10")
        report.append(f"   Grade: {final_grade} ({final_desc})")
        report.append("\n" + "█" * 70)
        
        return "\n".join(report)
