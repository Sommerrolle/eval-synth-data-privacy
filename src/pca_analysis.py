#!/usr/bin/env python3
"""
PCA Analysis Tool
----------------
This script performs Principal Component Analysis (PCA) on a dataset to determine
which columns contribute most to variance. It can be used either as a standalone
script or imported as a module.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def analyze_principal_components(df, exclude_cols=None, n_components=None, variance_threshold=0.95, 
                               scale_data=True, display_plots=True, output_dir=None):
    """
    Perform Principal Component Analysis to determine which columns contribute most to variance.
    
    Args:
        df: DataFrame containing the data to analyze
        exclude_cols: List of column names to exclude from analysis (e.g., ID columns)
        n_components: Number of components to return; if None, determined by variance_threshold
        variance_threshold: Cumulative variance threshold for selecting number of components (default: 0.95)
        scale_data: Whether to standardize the data before PCA (default: True)
        display_plots: Whether to display visualizations (default: True)
        
    Returns:
        Dictionary containing:
        - 'pca': Fitted PCA model
        - 'components': DataFrame of principal components with their loadings
        - 'explained_variance': Series of explained variance by component
        - 'feature_importance': DataFrame ranking features by their contribution to variance
        - 'reduced_data': DataFrame with the transformed data (if requested)
    """
    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()
    
    # Handle excluded columns
    if exclude_cols is None:
        exclude_cols = []
    
    # Separate features to analyze
    features = data.drop(columns=exclude_cols, errors='ignore')
    
    # Remove any remaining non-numeric columns
    # numeric_cols = features.select_dtypes(include=['int64', 'float64', 'double']).columns
    # features = features[numeric_cols]
    
    print(f"Analyzing {len(features.columns)} numeric features")
    
    # Standardize the data if requested
    if scale_data:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = features.values
    
    # Determine number of components based on variance threshold if not specified
    if n_components is None:
        # Initially try with all components
        pca_full = PCA()
        pca_full.fit(features_scaled)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"Based on {variance_threshold*100:.1f}% variance threshold, using {n_components} components")
    
    # Perform PCA with determined number of components
    pca = PCA(n_components=n_components)
    pca.fit(features_scaled)
    transformed_data = pca.transform(features_scaled)
    
    # Create DataFrame to show component loadings
    loadings = pd.DataFrame(
        data=pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=features.columns
    )
    
    # Calculate feature importance scores based on absolute loadings weighted by explained variance
    importance_scores = pd.DataFrame(index=features.columns)
    importance_scores['AbsoluteImportance'] = 0
    
    for i, component in enumerate(loadings.columns):
        # Weight the loadings by the explained variance of each component
        weighted_loading = np.abs(loadings[component]) * pca.explained_variance_ratio_[i]
        importance_scores['AbsoluteImportance'] += weighted_loading
    
    # Rank features by importance
    feature_importance = importance_scores.sort_values('AbsoluteImportance', ascending=False)
    feature_importance['CumulativeImportance'] = feature_importance['AbsoluteImportance'].cumsum() / feature_importance['AbsoluteImportance'].sum()
    feature_importance['RelativeImportance'] = feature_importance['AbsoluteImportance'] / feature_importance['AbsoluteImportance'].sum() * 100
    
    # Visualizations if requested
    if display_plots:
        # Create plots directory if saving plots and output_dir is specified
        plots_dir = None
        if output_dir:
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
        
        # Plot explained variance
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7)
        plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', color='red')
        plt.axhline(y=variance_threshold, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.grid(alpha=0.3)
        
        # Plot top feature importance
        top_n = min(20, len(feature_importance))
        plt.subplot(1, 2, 2)
        plt.barh(np.arange(top_n), feature_importance['RelativeImportance'].values[:top_n], alpha=0.7)
        plt.yticks(np.arange(top_n), feature_importance.index[:top_n])
        plt.xlabel('Relative Importance (%)')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        # Save the plot if output_dir is specified
        if plots_dir:
            plt.savefig(os.path.join(plots_dir, 'variance_importance.png'), dpi=300)
            
        plt.show()
        
        # Plot a heatmap of the component loadings
        plt.figure(figsize=(12, 6))
        top_loadings = loadings.loc[feature_importance.index[:min(15, len(feature_importance))]]
        sns.heatmap(top_loadings, cmap='coolwarm', center=0, annot=True, fmt='.2f')
        plt.title('PCA Component Loadings for Top Features')
        plt.tight_layout()
        
        # Save the plot if output_dir is specified
        if plots_dir:
            plt.savefig(os.path.join(plots_dir, 'component_loadings.png'), dpi=300)
            
        plt.show()
    
    # Create dataframe with transformed data
    reduced_df = pd.DataFrame(
        data=transformed_data,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data.index
    )
    
    # Return the results
    return {
        'pca': pca,
        'components': loadings,
        'explained_variance': pd.Series(pca.explained_variance_ratio_, 
                                       index=[f'PC{i+1}' for i in range(n_components)]),
        'feature_importance': feature_importance,
        'reduced_data': reduced_df
    }

def save_pca_results(results, output_dir='pca_results'):
    """
    Save PCA analysis results to files.
    
    Args:
        results: Dictionary of PCA results from analyze_principal_components
        output_dir: Directory to save results (default: 'pca_results')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save feature importance
    results['feature_importance'].to_csv(os.path.join(output_dir, 'feature_importance.csv'))
    
    # Save component loadings
    results['components'].to_csv(os.path.join(output_dir, 'component_loadings.csv'))
    
    # Save explained variance
    results['explained_variance'].to_csv(os.path.join(output_dir, 'explained_variance.csv'))
    
    # Save reduced data
    results['reduced_data'].to_csv(os.path.join(output_dir, 'reduced_data.csv'))
    
    print(f"Results saved to {output_dir}/")

def load_data(filepath):
    """
    Load data from a CSV or Excel file.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        Pandas DataFrame
    """
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.csv':
        return pd.read_csv(filepath)
    elif file_ext in ['.xls', '.xlsx']:
        return pd.read_excel(filepath)
    elif file_ext == '.parquet':
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def main():
    """Main function when script is run directly."""
    parser = argparse.ArgumentParser(description='Perform PCA analysis on a dataset.')
    parser.add_argument('--input', required=True, help='Input data file (CSV, Excel, or Parquet)')
    parser.add_argument('--exclude', nargs='+', default=[], 
                        help='Columns to exclude from analysis (e.g., ID columns)')
    parser.add_argument('--components', type=int, default=None, 
                        help='Number of components to extract')
    parser.add_argument('--variance', type=float, default=0.95, 
                        help='Variance threshold for selecting components (default: 0.95)')
    parser.add_argument('--no-scale', action='store_false', dest='scale', 
                        help='Do not standardize data before PCA')
    parser.add_argument('--no-plots', action='store_false', dest='plots', 
                        help='Do not display plots')
    parser.add_argument('--output-dir', default='pca_results', 
                        help='Directory to save results (default: pca_results)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Randomly sample N rows to analyze (helpful for large datasets)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"Loading data from {args.input}...")
        df = load_data(args.input)
        
        # Sample data if requested
        if args.sample and args.sample < len(df):
            df = df.sample(n=args.sample, random_state=42)
            print(f"Analyzing random sample of {args.sample} rows")
        
        print(f"Data shape: {df.shape}")
        
        # Run PCA analysis
        results = analyze_principal_components(
            df=df,
            exclude_cols=args.exclude,
            n_components=args.components,
            variance_threshold=args.variance,
            scale_data=args.scale,
            display_plots=args.plots
        )
        
        # Print top features
        print("\nTop 10 features by importance:")
        top_features = results['feature_importance'].head(10)
        for idx, row in top_features.iterrows():
            print(f"{idx}: {row['RelativeImportance']:.2f}% importance")
        
        # Save results
        save_pca_results(results, args.output_dir)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())