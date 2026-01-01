"""
Training utilities and visualization for Phase 3.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from typing import Dict, List

from nlp_paper_analyzer.utils import logger


def plot_training_history(history, title: str = "Model Training", save_path: str = None):
    """
    Plot training and validation accuracy/loss.
    
    Args:
        history: Keras training history object
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(14, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training plot to {save_path}")
    
    plt.show()


def evaluate_model(model, X_test, y_test, encoder=None):
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        encoder: LabelEncoder for class names
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    
    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    logger.info(f"Model evaluation: Acc={acc:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    
    # Detailed report
    if encoder:
        report = classification_report(y_test, y_pred, target_names=encoder.classes_)
        logger.info(f"\nClassification Report:\n{report}")
    
    return metrics


def compare_models(models_dict: Dict, X_test, y_test) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Args:
        models_dict: Dictionary of {name: model}
        X_test: Test features
        y_test: Test labels
        
    Returns:
        DataFrame with comparison results
    """
    results = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": []
    }
    
    for model_name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test)
        results["Model"].append(model_name)
        results["Accuracy"].append(metrics['accuracy'])
        results["Precision"].append(metrics['precision'])
        results["Recall"].append(metrics['recall'])
        results["F1-Score"].append(metrics['f1_score'])
    
    df_results = pd.DataFrame(results)
    
    logger.info(f"\nModel Comparison:\n{df_results}")
    
    return df_results


def plot_model_comparison(df_results: pd.DataFrame, save_path: str = None):
    """
    Plot comparison of multiple models.
    
    Args:
        df_results: DataFrame from compare_models
        save_path: Optional path to save figure
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(df_results))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        offset = width * (i - 1.5)
        ax.bar([p + offset for p in x], df_results[metric], width, label=metric)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['Model'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    
    plt.show()
