# ml_pipeline/models/evaluator.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np
from pathlib import Path

class ModelEvaluator:
    def __init__(self, results_dir: Path = Path("reports/")):
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)
        plt.style.use('seaborn-v0_8')
    
    def generate_evaluation_report(self, results, X_test, y_test, feature_names=None):
        """Generate comprehensive evaluation report"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curve
        for model_name, model_results in results.items():
            y_pred_proba = model_results['model'].predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            axes[0, 0].plot(fpr, tpr, lw=2, 
                           label=f'{model_name} (AUC = {roc_auc:.2f})')
        
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curves')
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True)
        
        # 2. Precision-Recall Curve
        for model_name, model_results in results.items():
            y_pred_proba = model_results['model'].predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            axes[0, 1].plot(recall, precision, lw=2,
                           label=f'{model_name} (AUC = {pr_auc:.2f})')
        
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curves')
        axes[0, 1].legend(loc="lower left")
        axes[0, 1].grid(True)
        
        # 3. Feature Importance (for best tree-based model)
        best_model_name = max(results.items(), 
                            key=lambda x: x[1]['best_score'])[0]
        best_model = results[best_model_name]['model']
        
        if hasattr(best_model, 'feature_importances_') and feature_names:
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10 features
            
            axes[1, 0].barh(range(len(indices)), importances[indices])
            axes[1, 0].set_yticks(range(len(indices)))
            axes[1, 0].set_yticklabels([feature_names[i] for i in indices])
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title(f'Feature Importance - {best_model_name}')
        
        # 4. Model Comparison
        model_names = list(results.keys())
        scores = [results[m]['best_score'] for m in model_names]
        
        axes[1, 1].bar(model_names, scores)
        axes[1, 1].set_ylabel('CV Score')
        axes[1, 1].set_title('Model Comparison')
        axes[1, 1].set_ylim([0, 1])
        
        for i, score in enumerate(scores):
            axes[1, 1].text(i, score + 0.02, f'{score:.3f}', 
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to JSON
        metrics = {}
        for model_name, model_results in results.items():
            metrics[model_name] = {
                'best_params': model_results['best_params'],
                'best_score': float(model_results['best_score']),
                'test_accuracy': float(model_results['test_accuracy']),
                'roc_auc': float(model_results['roc_auc'])
            }
        
        import json
        with open(self.results_dir / 'model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics