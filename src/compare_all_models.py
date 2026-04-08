import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_all_results():
    """Load results from all models"""
    
    # Load baseline results
    baseline_df = pd.read_csv('results/metrics/baseline_results.csv')
    
    # Load LSTM results
    lstm_df = pd.read_csv('results/metrics/lstm_results.csv')
    
    # Load transformer results (will be available after training)
    try:
        transformer_df = pd.read_csv('results/metrics/transformer_results.csv')
        all_results = pd.concat([baseline_df, lstm_df, transformer_df], ignore_index=True)
    except:
        print("Transformer results not yet available, showing baseline + LSTM only")
        all_results = pd.concat([baseline_df, lstm_df], ignore_index=True)
    
    return all_results

def create_comparison_visualizations(results_df):
    """Create comprehensive comparison visualizations"""
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Accuracy Comparison
    ax1 = plt.subplot(3, 3, 1)
    bars = ax1.bar(results_df['model_name'], results_df['accuracy'], color='skyblue', edgecolor='black')
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim([0.90, 1.0])
    ax1.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Precision Comparison
    ax2 = plt.subplot(3, 3, 2)
    bars = ax2.bar(results_df['model_name'], results_df['precision'], color='lightgreen', edgecolor='black')
    ax2.set_title('Precision Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_ylim([0.90, 1.0])
    ax2.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Recall Comparison
    ax3 = plt.subplot(3, 3, 3)
    bars = ax3.bar(results_df['model_name'], results_df['recall'], color='lightcoral', edgecolor='black')
    ax3.set_title('Recall Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Recall', fontsize=12)
    ax3.set_ylim([0.90, 1.0])
    ax3.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 4. F1 Score Comparison
    ax4 = plt.subplot(3, 3, 4)
    bars = ax4.bar(results_df['model_name'], results_df['f1'], color='gold', edgecolor='black')
    ax4.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('F1 Score', fontsize=12)
    ax4.set_ylim([0.90, 1.0])
    ax4.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Training Time Comparison (log scale)
    ax5 = plt.subplot(3, 3, 5)
    bars = ax5.bar(results_df['model_name'], results_df['train_time'], color='plum', edgecolor='black')
    ax5.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Time (seconds, log scale)', fontsize=12)
    ax5.set_yscale('log')
    ax5.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # 6. Inference Latency Comparison
    ax6 = plt.subplot(3, 3, 6)
    bars = ax6.bar(results_df['model_name'], results_df['latency_ms'], color='orange', edgecolor='black')
    ax6.set_title('Inference Latency per Email', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Latency (milliseconds)', fontsize=12)
    ax6.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms', ha='center', va='bottom', fontsize=9)
    
    # 7. Accuracy vs Training Time Trade-off
    ax7 = plt.subplot(3, 3, 7)
    scatter = ax7.scatter(results_df['train_time'], results_df['accuracy'], 
                         s=200, c=results_df['f1'], cmap='viridis', 
                         edgecolors='black', linewidth=2, alpha=0.7)
    for idx, row in results_df.iterrows():
        ax7.annotate(row['model_name'], 
                    (row['train_time'], row['accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax7.set_xlabel('Training Time (seconds, log scale)', fontsize=12)
    ax7.set_ylabel('Accuracy', fontsize=12)
    ax7.set_xscale('log')
    ax7.set_title('Accuracy vs Training Time Trade-off', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax7, label='F1 Score')
    
    # 8. Accuracy vs Latency Trade-off
    ax8 = plt.subplot(3, 3, 8)
    scatter = ax8.scatter(results_df['latency_ms'], results_df['accuracy'], 
                         s=200, c=results_df['f1'], cmap='plasma', 
                         edgecolors='black', linewidth=2, alpha=0.7)
    for idx, row in results_df.iterrows():
        ax8.annotate(row['model_name'], 
                    (row['latency_ms'], row['accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax8.set_xlabel('Inference Latency (milliseconds)', fontsize=12)
    ax8.set_ylabel('Accuracy', fontsize=12)
    ax8.set_title('Accuracy vs Latency Trade-off', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax8, label='F1 Score')
    
    # 9. Overall Performance Heatmap
    ax9 = plt.subplot(3, 3, 9)
    metrics_for_heatmap = results_df[['model_name', 'accuracy', 'precision', 'recall', 'f1']].set_index('model_name')
    sns.heatmap(metrics_for_heatmap.T, annot=True, fmt='.4f', cmap='YlGnBu', 
                cbar_kws={'label': 'Score'}, ax=ax9, linewidths=0.5)
    ax9.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
    ax9.set_xlabel('')
    ax9.set_ylabel('Metrics', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/plots/comprehensive_model_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Comprehensive comparison plot saved!")
    plt.show()

def print_comparison_table(results_df):
    """Print formatted comparison table"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*100)
    print(results_df.to_string(index=False))
    print("="*100)
    
    # Find best models
    best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
    best_f1 = results_df.loc[results_df['f1'].idxmax()]
    fastest_train = results_df.loc[results_df['train_time'].idxmin()]
    fastest_inference = results_df.loc[results_df['latency_ms'].idxmin()]
    
    print("\n🏆 BEST MODELS BY METRIC:")
    print(f"  Best Accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})")
    print(f"  Best F1 Score: {best_f1['model_name']} ({best_f1['f1']:.4f})")
    print(f"  Fastest Training: {fastest_train['model_name']} ({fastest_train['train_time']:.4f}s)")
    print(f"  Fastest Inference: {fastest_inference['model_name']} ({fastest_inference['latency_ms']:.4f}ms)")
    print("="*100)

def main():
    """Main comparison pipeline"""
    
    print("Loading all model results...")
    results_df = load_all_results()
    
    # Print comparison table
    print_comparison_table(results_df)
    
    # Create visualizations
    print("\nCreating comprehensive visualizations...")
    create_comparison_visualizations(results_df)
    
    # Save combined results
    results_df.to_csv('results/metrics/all_models_comparison.csv', index=False)
    print("\n✓ All results saved to results/metrics/all_models_comparison.csv")

if __name__ == "__main__":
    main()