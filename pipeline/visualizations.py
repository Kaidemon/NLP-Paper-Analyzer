"""
Visualization utilities for quality scores (Phase 8).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

from nlp_paper_analyzer.utils import logger


def visualize_scores(scorer, save_path: str = None):
    """
    Create visualizations for paper quality scores.
    
    Args:
        scorer: PaperQualityScorer instance with calculated scores
        save_path: Optional path to save figure
    """
    if not scorer.scores:
        logger.warning("No scores to visualize. Run calculate_all_scores() first.")
        return
    
    # Prepare data
    score_names = {
        'structure': 'Structure',
        'section_order': 'Flow',
        'classification_confidence': 'Clarity',
        'grammar_quality': 'Grammar',
        'consistency': 'Consistency'
    }
    
    categories = list(score_names.values())
    scores = [scorer.scores.get(key, 0) for key in score_names.keys()]
    final_score = scorer.scores.get('final', 0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # --- Plot 1: Bar Chart ---
    ax1 = axes[0]
    colors = ['#2ecc71' if s >= 7 else '#f39c12' if s >= 5 else '#e74c3c' for s in scores]
    bars = ax1.barh(categories, scores, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_xlim(0, 10)
    ax1.set_xlabel('Score (out of 10)', fontsize=12)
    ax1.set_title('Individual Criterion Scores', fontsize=14, fontweight='bold')
    ax1.axvline(x=final_score, color='blue', linestyle='--', linewidth=2, label=f'Final: {final_score}')
    ax1.legend(loc='lower right')
    
    # Add score labels on bars
    for bar, score in zip(bars, scores):
        ax1.text(score + 0.1, bar.get_y() + bar.get_height()/2, f'{score:.1f}',
                va='center', fontsize=11, fontweight='bold')
    
    ax1.grid(axis='x', alpha=0.3)
    
    # --- Plot 2: Radar Chart ---
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    scores_radar = scores + [scores[0]]  # Close the polygon
    angles += angles[:1]
    
    ax2 = fig.add_subplot(132, polar=True)
    ax2.fill(angles, scores_radar, color='#3498db', alpha=0.3)
    ax2.plot(angles, scores_radar, color='#2980b9', linewidth=2, marker='o')
    ax2.set_ylim(0, 10)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_title('Score Distribution', fontsize=14, fontweight='bold', y=1.1)
    
    # --- Plot 3: Final Score Gauge ---
    ax3 = axes[2]
    
    # Create a semi-circle gauge
    theta = np.linspace(0, np.pi, 100)
    
    # Background arcs for score ranges
    for start, end, color, label in [
        (0, 0.4, '#e74c3c', 'Poor'),
        (0.4, 0.6, '#f39c12', 'Average'),
        (0.6, 0.8, '#27ae60', 'Good'),
        (0.8, 1.0, '#2ecc71', 'Excellent')
    ]:
        theta_seg = np.linspace(start * np.pi, end * np.pi, 50)
        ax3.fill_between(theta_seg, 0.7, 1.0, alpha=0.3, color=color)
    
    # Needle
    needle_angle = (1 - final_score/10) * np.pi
    ax3.annotate('', xy=(needle_angle, 0.9), xytext=(np.pi/2, 0),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=3))
    
    ax3.set_xlim(0, np.pi)
    ax3.set_ylim(0, 1.2)
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    # Final score text
    grade, desc = scorer.get_grade(final_score)
    ax3.text(np.pi/2, 0.3, f'{final_score}/10', fontsize=28, fontweight='bold',
            ha='center', va='center', color='#2c3e50')
    ax3.text(np.pi/2, 0.1, f'Grade: {grade}', fontsize=16, ha='center', va='center', color='#7f8c8d')
    ax3.text(np.pi/2, -0.05, f'({desc})', fontsize=12, ha='center', va='center', color='#95a5a6')
    ax3.set_title('Final Paper Score', fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Score visualization saved to: {save_path}")
    
    plt.show()


def display_detailed_scores(scorer):
    """
    Display detailed breakdown of all scores with explanations.
    
    Args:
        scorer: PaperQualityScorer instance with calculated scores
    """
    if not scorer.scores:
        logger.warning("No scores calculated yet.")
        return
    
    print("\n" + "=" * 80)
    print("                    DETAILED SCORE BREAKDOWN")
    print("=" * 80)
    
    details_info = {
        'structure': {
            'name': 'Structure & Completeness',
            'description': 'Evaluates if the paper has all essential academic sections'
        },
        'section_order': {
            'name': 'Section Order & Flow',
            'description': 'Checks if sections follow logical academic paper structure'
        },
        'classification_confidence': {
            'name': 'Section Clarity',
            'description': 'Measures how clearly each section is defined and written'
        },
        'grammar_quality': {
            'name': 'Grammar & Writing Quality',
            'description': 'Assesses writing quality based on grammar corrections needed'
        },
        'consistency': {
            'name': 'Internal Consistency',
            'description': 'Evaluates factual coherence and claim verification'
        }
    }
    
    for key, info in details_info.items():
        score = scorer.scores.get(key, 0)
        details = scorer.details.get(key, {})
        grade, grade_desc = scorer.get_grade(score)
        
        print(f"\n{info['name']}")
        print(f"Score: {score}/10 ({grade} - {grade_desc})")
        print(f"Description: {info['description']}")
        print("-" * 50)
        
        if isinstance(details, dict):
            for detail_key, detail_value in details.items():
                formatted_key = detail_key.replace('_', ' ').title()
                print(f"  • {formatted_key}: {detail_value}")
        else:
            print(f"  Note: {details}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("                         SCORE SUMMARY TABLE")
    print("=" * 80)
    
    summary_data = []
    for key in ['structure', 'section_order', 'classification_confidence', 'grammar_quality', 'consistency']:
        score = scorer.scores.get(key, 0)
        grade, desc = scorer.get_grade(score)
        summary_data.append({
            'Criterion': details_info[key]['name'],
            'Score': f"{score}/10",
            'Grade': grade,
            'Status': desc
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Final score
    final_score = scorer.scores.get('final', 0)
    final_grade, final_desc = scorer.get_grade(final_score)
    print(f"\n{'='*80}")
    print(f"FINAL SCORE: {final_score}/10 | Grade: {final_grade} | {final_desc}")
    print(f"{'='*80}")
