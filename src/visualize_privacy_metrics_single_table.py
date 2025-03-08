import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import sys

# Set up the visualization style
plt.style.use('seaborn-v0_8-whitegrid')
colors = list(mcolors.TABLEAU_COLORS.values())
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# Dictionary to map table names to meaningful labels
TABLE_NAMES = {
    "joined_1_4_5_6_7": "Inpatient Data",
    "joined_1_8_9_10_11": "Outpatient Data"
}

# *** Set the target table to visualize here ***
TARGET_TABLE = "joined_1_4_5_6_7"  # Change this to filter for a specific table

def load_privacy_metrics(file_path):
    """Load privacy metrics from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_output_directory(dir_name="privacy_visualization"):
    """Create directory for output visualizations."""
    output_dir = Path(dir_name)
    output_dir.mkdir(exist_ok=True)
    return output_dir

def format_large_number(x):
    """Format large numbers for better readability."""
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    elif x >= 1_000:
        return f"{x/1_000:.2f}K"
    else:
        return f"{x:.2f}"

def filter_comparisons_by_table(data, target_table):
    """Filter the comparisons to only include the target table."""
    filtered_comparisons = []
    for comp in data["comparisons"]:
        if comp["table_name"] == target_table:
            filtered_comparisons.append(comp)
    
    if not filtered_comparisons:
        print(f"Error: Table '{target_table}' not found in the data.")
        print(f"Available tables: {[comp['table_name'] for comp in data['comparisons']]}")
        sys.exit(1)
    
    filtered_data = data.copy()
    filtered_data["comparisons"] = filtered_comparisons
    return filtered_data

def plot_k_anonymity_metrics(data, output_dir):
    """Plot k-anonymity metrics comparison focusing on unique and vulnerable records."""
    # Extract data for the plot
    tables = [TABLE_NAMES.get(comp["table_name"], comp["table_name"]) for comp in data["comparisons"]]
    
    # Prepare the data in long format for seaborn
    plot_data = []
    
    for i, comp in enumerate(data["comparisons"]):
        table = tables[i]
        
        # Original data
        orig_unique = comp["dataset1_results"]["unique_records"]
        orig_total = comp["dataset1_results"]["total_groups"]
        orig_unique_pct = orig_unique / orig_total * 100
        
        orig_vulnerable = comp["dataset1_results"]["vulnerable_groups"]
        orig_vulnerable_pct = orig_vulnerable / orig_total * 100
        
        # Synthetic data
        synth_unique = comp["dataset2_results"]["unique_records"]
        synth_total = comp["dataset2_results"]["total_groups"]
        synth_unique_pct = synth_unique / synth_total * 100
        
        synth_vulnerable = comp["dataset2_results"]["vulnerable_groups"]
        synth_vulnerable_pct = synth_vulnerable / synth_total * 100
        
        # Add to plot data
        plot_data.extend([
            {'Table': table, 'Dataset': 'Original', 'Metric': 'Unique Records', 'Value': orig_unique_pct},
            {'Table': table, 'Dataset': 'Synthetic', 'Metric': 'Unique Records', 'Value': synth_unique_pct},
            {'Table': table, 'Dataset': 'Original', 'Metric': 'Vulnerable Groups', 'Value': orig_vulnerable_pct},
            {'Table': table, 'Dataset': 'Synthetic', 'Metric': 'Vulnerable Groups', 'Value': synth_vulnerable_pct}
        ])
    
    df = pd.DataFrame(plot_data)
    
    # For a single table, use regular bar chart instead of catplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Unique Records
    unique_df = df[df["Metric"] == "Unique Records"]
    #sns.barplot(data=unique_df, x="Dataset", y="Value", ax=axes[0], palette="dark", alpha=0.6)
    sns.barplot(data=unique_df, x="Dataset", y="Value", hue="Dataset", ax=axes[0], palette="dark", alpha=0.6, legend=False)
    axes[0].set_title("Unique Records (%)")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Add value labels
    for i, bar in enumerate(axes[0].patches):
        height = bar.get_height()
        if height > 0.01:  # Only label if the value is visible
            axes[0].text(
                bar.get_x() + bar.get_width()/2, height + 0.1,
                f"{height:.2f}%", ha='center', va='bottom',
                color='black', fontsize=10
            )
    
    # Plot Vulnerable Groups
    vulnerable_df = df[df["Metric"] == "Vulnerable Groups"]
    sns.barplot(data=vulnerable_df, x="Dataset", y="Value", ax=axes[1], palette="dark", alpha=0.6)
    axes[1].set_title("Vulnerable Groups (%)")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Add value labels
    for i, bar in enumerate(axes[1].patches):
        height = bar.get_height()
        if height > 0.01:  # Only label if the value is visible
            axes[1].text(
                bar.get_x() + bar.get_width()/2, height + 0.1,
                f"{height:.2f}%", ha='center', va='bottom',
                color='black', fontsize=10
            )
    
    plt.suptitle(f"K-Anonymity Metrics - {tables[0]} (Lower is Better)", y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'k_anonymity_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a log scale version for better visibility of small values
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Unique Records (log scale)
    sns.barplot(data=unique_df, x="Dataset", y="Value", ax=axes[0], palette="dark", alpha=0.6)
    axes[0].set_title("Unique Records (%) - Log Scale")
    axes[0].set_ylabel("Percentage (%) - Log Scale")
    axes[0].set_yscale('log')
    
    # Add value labels
    for i, bar in enumerate(axes[0].patches):
        height = bar.get_height()
        if height > 0:  # Only label if the value is visible
            axes[0].text(
                bar.get_x() + bar.get_width()/2, height * 1.1,
                f"{height:.2f}%", ha='center', va='bottom',
                color='black', fontsize=10
            )
    
    # Plot Vulnerable Groups (log scale)
    #sns.barplot(data=vulnerable_df, x="Dataset", y="Value", ax=axes[1], palette="dark", alpha=0.6)
    sns.barplot(data=vulnerable_df, x="Dataset", y="Value", hue="Dataset", ax=axes[1], palette="dark", alpha=0.6, legend=False)
    axes[1].set_title("Vulnerable Groups (%) - Log Scale")
    axes[1].set_ylabel("Percentage (%) - Log Scale")
    axes[1].set_yscale('log')
    
    # Add value labels
    for i, bar in enumerate(axes[1].patches):
        height = bar.get_height()
        if height > 0:  # Only label if the value is visible
            axes[1].text(
                bar.get_x() + bar.get_width()/2, height * 1.1,
                f"{height:.2f}%", ha='center', va='bottom',
                color='black', fontsize=10
            )
    
    plt.suptitle(f"K-Anonymity Metrics - {tables[0]} - Log Scale (Lower is Better)", y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'k_anonymity_metrics_log.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_group_size_distribution(data, output_dir):
    """Plot group size distribution comparing original and synthetic data."""
    for i, comp in enumerate(data["comparisons"]):
        table_name = TABLE_NAMES.get(comp["table_name"], comp["table_name"])
        
        # Extract data for original dataset
        orig_dist = comp["dataset1_results"]["group_size_distribution"]
        orig_total = comp["dataset1_results"]["total_groups"]
        
        # Extract data for synthetic dataset
        synth_dist = comp["dataset2_results"]["group_size_distribution"]
        synth_total = comp["dataset2_results"]["total_groups"]
        
        # Convert to percentages
        orig_pct = {k: (v / orig_total * 100) for k, v in orig_dist.items()}
        synth_pct = {k: (v / synth_total * 100) for k, v in synth_dist.items()}
        
        # Create bar plots for raw counts
        sizes = list(range(1, 11)) + ['>10']
        orig_values = [orig_dist[f"groups_of_size_{s}"] if s <= 10 else orig_dist["groups_larger_than_10"] for s in sizes[:-1]] + [orig_dist["groups_larger_than_10"]]
        synth_values = [synth_dist[f"groups_of_size_{s}"] if s <= 10 else synth_dist["groups_larger_than_10"] for s in sizes[:-1]] + [synth_dist["groups_larger_than_10"]]
        
        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot raw counts
        ax = axes[0]
        x = np.arange(len(sizes))
        width = 0.35
        
        orig_bars = ax.bar(x - width/2, orig_values, width, label='Original', alpha=0.7, color=colors[0])
        synth_bars = ax.bar(x + width/2, synth_values, width, label='Synthetic', alpha=0.7, color=colors[1])
        
        ax.set_yscale('log')
        ax.set_title(f"Group Size Distribution - Raw Counts (Log Scale)")
        ax.set_xlabel("Group Size")
        ax.set_ylabel("Number of Groups (Log Scale)")
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.legend()
        
        # Add count labels
        for j, bar in enumerate(orig_bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height * 1.05,
                        format_large_number(height), ha='center', va='bottom', 
                        fontsize=8, rotation=45)
                
        for j, bar in enumerate(synth_bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height * 1.05,
                        format_large_number(height), ha='center', va='bottom', 
                        fontsize=8, rotation=45)
        
        # Plot percentages
        ax = axes[1]
        orig_pct_values = [orig_pct[f"groups_of_size_{s}"] if s <= 10 else orig_pct["groups_larger_than_10"] for s in sizes[:-1]] + [orig_pct["groups_larger_than_10"]]
        synth_pct_values = [synth_pct[f"groups_of_size_{s}"] if s <= 10 else synth_pct["groups_larger_than_10"] for s in sizes[:-1]] + [synth_pct["groups_larger_than_10"]]
        
        orig_bars = ax.bar(x - width/2, orig_pct_values, width, label='Original', alpha=0.7, color=colors[0])
        synth_bars = ax.bar(x + width/2, synth_pct_values, width, label='Synthetic', alpha=0.7, color=colors[1])
        
        ax.set_title(f"Group Size Distribution - Percentage")
        ax.set_xlabel("Group Size")
        ax.set_ylabel("Percentage of Groups")
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend()
        
        # Add percentage labels
        for j, bar in enumerate(orig_bars):
            height = bar.get_height()
            if height > 0.5:  # Only label if percentage is visible
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f"{height:.1f}%", ha='center', va='bottom', 
                        fontsize=8, rotation=45)
                
        for j, bar in enumerate(synth_bars):
            height = bar.get_height()
            if height > 0.5:  # Only label if percentage is visible
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f"{height:.1f}%", ha='center', va='bottom', 
                        fontsize=8, rotation=45)
    
        plt.suptitle(f"Group Size Distribution - {table_name}", y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'group_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_l_diversity_metrics(data, output_dir):
    """Plot l-diversity metrics comparing privacy protection between original and synthetic data."""
    # We'll focus on two key metrics: average distinct values and problematic groups percentage
    
    metrics_data = []
    
    for comp in data["comparisons"]:
        table_name = TABLE_NAMES.get(comp["table_name"], comp["table_name"])
        
        # Process each sensitive attribute with l-diversity measurements
        for attr, orig_values in comp["dataset1_results"]["l_diversity"].items():
            if attr in comp["dataset2_results"]["l_diversity"]:
                synth_values = comp["dataset2_results"]["l_diversity"][attr]
                
                # Record average distinct values (higher is better for privacy)
                metrics_data.append({
                    "Table": table_name,
                    "Attribute": attr.replace("_diagnosis", "").replace("_procedure_code", ""),
                    "Metric": "Average Distinct Values",
                    "Dataset": "Original",
                    "Value": orig_values["average_distinct_values"]
                })
                metrics_data.append({
                    "Table": table_name,
                    "Attribute": attr.replace("_diagnosis", "").replace("_procedure_code", ""),
                    "Metric": "Average Distinct Values",
                    "Dataset": "Synthetic",
                    "Value": synth_values["average_distinct_values"]
                })
                
                # Record problematic groups percentage (lower is better for privacy)
                metrics_data.append({
                    "Table": table_name,
                    "Attribute": attr.replace("_diagnosis", "").replace("_procedure_code", ""),
                    "Metric": "Problematic Groups (%)",
                    "Dataset": "Original",
                    "Value": orig_values["problematic_groups_percentage"]
                })
                metrics_data.append({
                    "Table": table_name,
                    "Attribute": attr.replace("_diagnosis", "").replace("_procedure_code", ""),
                    "Metric": "Problematic Groups (%)",
                    "Dataset": "Synthetic",
                    "Value": synth_values["problematic_groups_percentage"]
                })
    
    df = pd.DataFrame(metrics_data)
    
    # Create a pair of plots: one for Average Distinct Values and one for Problematic Groups
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot Average Distinct Values (higher is better)
    avg_df = df[df["Metric"] == "Average Distinct Values"]
    sns.barplot(data=avg_df, x="Attribute", y="Value", hue="Dataset", 
                palette="viridis", alpha=0.8, ax=axes[0])
    axes[0].set_title("L-Diversity: Average Distinct Values by Attribute\n(Higher is Better)")
    axes[0].set_ylabel("Average Distinct Values")
    axes[0].set_xlabel("")
    
    # Add value labels
    for i, bar in enumerate(axes[0].patches):
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width()/2, height + 0.1,
            f"{height:.1f}", ha='center', va='bottom',
            color='black', fontsize=10
        )
    
    # Plot Problematic Groups Percentage (lower is better)
    prob_df = df[df["Metric"] == "Problematic Groups (%)"]
    
    # Determine if we need log scale
    max_val = prob_df["Value"].max()
    min_val = prob_df["Value"][prob_df["Value"] > 0].min()
    use_log = max_val / min_val > 100  # Use log scale if range is more than 100x
    
    sns.barplot(data=prob_df, x="Attribute", y="Value", hue="Dataset", 
                palette="viridis", alpha=0.8, ax=axes[1])
    axes[1].set_title("L-Diversity: Problematic Groups Percentage\n(Lower is Better)")
    axes[1].set_ylabel("Problematic Groups (%)")
    axes[1].set_xlabel("")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    
    if use_log:
        axes[1].set_yscale('log')
        axes[1].set_ylabel("Problematic Groups (%) - Log Scale")
    
    # Add value labels
    for i, bar in enumerate(axes[1].patches):
        height = bar.get_height()
        if height > 0.01:  # Only label if the value is visible
            axes[1].text(
                bar.get_x() + bar.get_width()/2, height * 1.05,
                f"{height:.1f}%", ha='center', va='bottom',
                color='black', fontsize=10, rotation=45 if height < 1 else 0
            )
    
    table_name = TABLE_NAMES.get(data["comparisons"][0]["table_name"], data["comparisons"][0]["table_name"])
    plt.suptitle(f"L-Diversity Metrics - {table_name}", y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'l_diversity_metrics.png', dpi=300, bbox_inches='tight')
    
    # Create a log scale version of problematic groups if not already done
    if not use_log:
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(data=prob_df, x="Attribute", y="Value", hue="Dataset", 
                    palette="viridis", alpha=0.8)
        plt.title(f"L-Diversity: Problematic Groups Percentage - Log Scale\n(Lower is Better)")
        plt.ylabel("Problematic Groups (%) - Log Scale")
        plt.xlabel("")
        ax.set_yscale('log')
        
        # Add value labels
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            if height > 0:  # Only label if the value is visible
                plt.text(
                    bar.get_x() + bar.get_width()/2, height * 1.05,
                    f"{height:.1f}%", ha='center', va='bottom',
                    color='black', fontsize=10, rotation=90
                )
        
        plt.suptitle(f"L-Diversity - {table_name}", y=1.05)        
        plt.tight_layout()
        plt.savefig(output_dir / 'l_diversity_problematic_groups_log.png', dpi=300, bbox_inches='tight')
        
    plt.close('all')

def plot_t_closeness_metrics(data, output_dir):
    """Plot t-closeness metrics comparing original and synthetic data."""
    metrics_data = []
    
    for comp in data["comparisons"]:
        table_name = TABLE_NAMES.get(comp["table_name"], comp["table_name"])
        
        # Process each sensitive attribute with t-closeness measurements
        for attr, orig_values in comp["dataset1_results"]["t_closeness"].items():
            if attr in comp["dataset2_results"]["t_closeness"]:
                synth_values = comp["dataset2_results"]["t_closeness"][attr]
                
                # Record t-closeness value (lower is better)
                metrics_data.append({
                    "Table": table_name,
                    "Attribute": attr.replace("_diagnosis", "").replace("_procedure_code", ""),
                    "Metric": "t-closeness",
                    "Dataset": "Original",
                    "Value": orig_values["t_closeness"]
                })
                metrics_data.append({
                    "Table": table_name,
                    "Attribute": attr.replace("_diagnosis", "").replace("_procedure_code", ""),
                    "Metric": "t-closeness",
                    "Dataset": "Synthetic",
                    "Value": synth_values["t_closeness"]
                })
                
                # Record average distance (lower is better)
                metrics_data.append({
                    "Table": table_name,
                    "Attribute": attr.replace("_diagnosis", "").replace("_procedure_code", ""),
                    "Metric": "Average Distance",
                    "Dataset": "Original",
                    "Value": orig_values["average_distance"]
                })
                metrics_data.append({
                    "Table": table_name,
                    "Attribute": attr.replace("_diagnosis", "").replace("_procedure_code", ""),
                    "Metric": "Average Distance",
                    "Dataset": "Synthetic",
                    "Value": synth_values["average_distance"]
                })
    
    df = pd.DataFrame(metrics_data)
    
    # Create plots for each metric (one for t-closeness and one for average distance)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot t-closeness values (lower is better)
    tc_df = df[df["Metric"] == "t-closeness"]
    sns.barplot(data=tc_df, x="Attribute", y="Value", hue="Dataset", 
                palette="viridis", alpha=0.8, ax=axes[0])
    axes[0].set_title("T-Closeness Values by Attribute\n(Lower is Better)")
    axes[0].set_ylabel("t-closeness")
    axes[0].set_xlabel("")
    axes[0].set_yscale('log')  # Use log scale for better visualization
    
    # Add value labels
    for i, bar in enumerate(axes[0].patches):
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width()/2, height * 1.05,
            format_large_number(height), ha='center', va='bottom',
            color='black', fontsize=10, rotation=45
        )
    
    # Plot average distance (lower is better)
    avg_df = df[df["Metric"] == "Average Distance"]
    sns.barplot(data=avg_df, x="Attribute", y="Value", hue="Dataset", 
                palette="viridis", alpha=0.8, ax=axes[1])
    axes[1].set_title("T-Closeness: Average Distance by Attribute\n(Lower is Better)")
    axes[1].set_ylabel("Average Distance")
    axes[1].set_xlabel("")
    axes[1].set_yscale('log')  # Use log scale for better visualization
    
    # Add value labels
    for i, bar in enumerate(axes[1].patches):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width()/2, height * 1.05,
            format_large_number(height), ha='center', va='bottom',
            color='black', fontsize=10, rotation=45
        )
    
    table_name = TABLE_NAMES.get(data["comparisons"][0]["table_name"], data["comparisons"][0]["table_name"])
    plt.suptitle(f"T-Closeness Metrics - {table_name}", y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 't_closeness_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load the privacy metrics JSON
    data = load_privacy_metrics("results/privacy_calculator/privacy_metrics_claims_data_limebit_mtgan_06032025_232020.json")
    
    # Filter data for the target table
    filtered_data = filter_comparisons_by_table(data, TARGET_TABLE)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Generate plots for the filtered data
    plot_k_anonymity_metrics(filtered_data, output_dir)
    plot_group_size_distribution(filtered_data, output_dir)
    plot_l_diversity_metrics(filtered_data, output_dir)
    plot_t_closeness_metrics(filtered_data, output_dir)
    
    print(f"Visualization completed for table: {TARGET_TABLE}")
    print(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main()