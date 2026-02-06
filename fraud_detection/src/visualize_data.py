import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Loading data...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    
    # Setup plot grid
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    sns.set_style("whitegrid")
    
    # 1. Class Distribution (Normal vs Fraud)
    # Using specific labels
    fraud_counts = df['is_fraud'].value_counts()
    sns.barplot(x=fraud_counts.index, y=fraud_counts.values, ax=axes[0,0], palette=['#4c72b0', '#c44e52'])
    axes[0,0].set_title('Class Distribution (Normal vs Fraud)')
    axes[0,0].set_xticklabels(['Normal (0)', 'Fraud (1)'])
    axes[0,0].set_ylabel('Count')
    axes[0,0].bar_label(axes[0,0].containers[0])
    
    # 2. Fraud Type Distribution
    # Filter only fraud
    fraud_only = df[df['is_fraud'] == 1]
    type_counts = fraud_only['fraud_type'].value_counts()
    sns.barplot(x=type_counts.index, y=type_counts.values, ax=axes[0,1], palette="viridis")
    axes[0,1].set_title('Distribution of Fraud Types')
    axes[0,1].set_xlabel('Fraud Type')
    axes[0,1].bar_label(axes[0,1].containers[0])
    
    # 3. Scatter Plot: Amount vs Time (Time of day or just index)
    # Let's use Index (order) on X and Amount on Y to show spikes
    # Color by Fraud
    sns.scatterplot(data=df, x=df.index, y='amount', hue='is_fraud', style='is_fraud', 
                    palette=['#4c72b0', '#c44e52'], s=30, alpha=0.7, ax=axes[1,0])
    axes[1,0].set_title('Transaction Amount Sequence (Spikes Highlights)')
    axes[1,0].set_ylabel('Amount ($)')
    axes[1,0].set_xlabel('Transaction Index (Time Order)')
    
    # 4. Boxplot of Amount by Class
    # Log scale for amount often helps
    sns.boxplot(data=df, x='is_fraud', y='amount', ax=axes[1,1], palette=['#4c72b0', '#c44e52'])
    axes[1,1].set_title('Amount Distribution by Class')
    axes[1,1].set_yscale('log')
    axes[1,1].set_xticklabels(['Normal', 'Fraud'])
    axes[1,1].set_ylabel('Amount (Log Scale)')

    # Save
    output_path = os.path.join(MODEL_DIR, "data_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
