import pandas as pd
import mygene
import numpy as np
from gprofiler import GProfiler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import os

def download_and_prepare_data():
    # Download and extract the dataset
    os.system("wget -O GSE161904_Raw_gene_counts_cortex.txt.gz 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE161904&format=file&file=GSE161904%5FRaw%5Fgene%5Fcounts%5Fcortex%2Etxt%2Egz'")
    os.system("gunzip GSE161904_Raw_gene_counts_cortex.txt.gz")
    
    # Load the dataset
    data = pd.read_csv("GSE161904_Raw_gene_counts_cortex.txt", sep="\t")
    data.to_csv("GSE161904_Raw_gene_counts_cortex.csv", index=False)
    return data

def rename_columns(data):
    # Define current and new column names
    current_columns = [
        "G3R1_Cortex_3xTgAD", "G1R2_Cortex_3xTgAD", "G1R3_Cortex_3xTgAD", "G1R4_Cortex_3xTgAD",
        "G1R5_Cortex_WT", "G1R6_Cortex_WT", "G1R7_Cortex_WT", "G4R1_Cortex_3xTgAD",
        "G4R2_Cortex_3xTgAD", "G4R3_Cortex_3xTgAD", "G3R3_Cortex_3xTgAD", "G4R5_Cortex_WT",
        "G4R6_Cortex_WT", "G4R7_Cortex_WT", "G5R1_Cortex_3xTgAD", "G5R2_Cortex_3xTgAD",
        "G5R3_Cortex_3xTgAD", "G5R4_Cortex_WT", "G5R6_Cortex_WT", "G5R5_Cortex_WT",
        "G3R4_Cortex_3xTgAD", "G2R4_Cortex_WT", "G2R5_Cortex_3xTgAD", "G2R6_Cortex_3xTgAD",
        "G2R7_Cortex_3xTgAD", "G3R10_Cortex_WT", "G2R2_Cortex_WT", "G2R3_Cortex_WT",
        "G3R7_Cortex_WT", "G3R9_Cortex_WT"
    ]

    new_columns = [
        "AD1_8mo", "AD1_2mo", "AD2_2mo", "AD3_2mo", "WT1_2mo", "WT2_2mo", "WT3_2mo",
        "AD1_11mo", "AD2_11mo", "AD3_11mo", "AD2_8mo", "WT1_11mo", "WT2_11mo", "WT3_11mo",
        "AD1_14mo", "AD2_14mo", "AD3_14mo", "WT1_14mo", "WT2_14mo", "WT3_14mo", "AD3_8mo",
        "WT3_7mo", "AD1_7mo", "AD2_7mo", "AD3_7mo", "WT3_8mo", "WT1_7mo", "WT2_7mo",
        "WT1_8mo", "WT2_8mo"
    ]

    # Create a mapping dictionary and rename columns
    column_mapping = dict(zip(current_columns, new_columns))
    data.rename(columns=column_mapping, inplace=True)
    
    return data

def reorder_columns(data):
    # Define the desired column order
    new_column_order = [
        "WT1_2mo", "WT2_2mo", "WT3_2mo", "AD1_2mo", "AD2_2mo", "AD3_2mo",
        "WT1_7mo", "WT2_7mo", "WT3_7mo", "AD1_7mo", "AD2_7mo", "AD3_7mo",
        "WT1_8mo", "WT2_8mo", "WT3_8mo", "AD1_8mo", "AD2_8mo", "AD3_8mo",
        "WT1_11mo", "WT2_11mo", "WT3_11mo", "AD1_11mo", "AD2_11mo", "AD3_11mo",
        "WT1_14mo", "WT2_14mo", "WT3_14mo", "AD1_14mo", "AD2_14mo", "AD3_14mo"
    ]

    # Reorder columns
    return data[new_column_order]

def annotate_genes(data):
    mg = mygene.MyGeneInfo()
    ensembl_ids = data.index.tolist()
    
    # Query gene symbols
    gene_info = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='mouse')
    gene_df = pd.DataFrame(gene_info)

    # Remove duplicates and missing symbols
    gene_df = gene_df.dropna(subset=['symbol']).drop_duplicates(subset='query')
    
    # Map gene symbols to the original dataframe
    data['Gene_Name'] = data.index.map(gene_df.set_index('query')['symbol'])
    cols = ['Gene_Name'] + [col for col in data.columns if col != 'Gene_Name']
    data = data[cols]
    
    # Set Gene_Name as index
    data = data.set_index('Gene_Name')
    return data

def filter_normalize(data, min_cpm=0.70, min_samples=2):
    # Convert raw counts to CPM
    cpm = data.apply(lambda x: (x / x.sum()) * 1e6)
    
    # Filter genes based on CPM thresholds
    mask = (cpm > min_cpm).sum(axis=1) >= min_samples
    filtered_data = data[mask]

    # Compute geometric mean and size factors
    geometric_means = filtered_data.apply(lambda row: np.exp(np.log(row[row > 0]).mean()), axis=1)
    size_factors = filtered_data.div(geometric_means, axis=0).median(axis=0)

    # Normalize data
    normalized_data = filtered_data.div(size_factors, axis=1)
    return pd.DataFrame(normalized_data, index=filtered_data.index, columns=filtered_data.columns)

def plot_normalized_data(data):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Total normalized counts per sample
    total_counts_normalized = data.sum(axis=0)
    axes[0].bar(data.columns, total_counts_normalized, color='lightcoral')
    axes[0].set_ylabel('Total Normalized Counts')
    axes[0].set_title('Total Counts per Sample (Normalized)')
    axes[0].tick_params(axis='x', rotation=85)

    # Log-transformed normalized counts per sample
    log_normalized_data = data.apply(lambda x: np.log2(x + 1))
    log_normalized_data.boxplot(ax=axes[1])
    axes[1].set_ylabel('Log2(Normalized Counts + 1)')
    axes[1].set_title('Log Transformed Counts per Sample (Normalized)')
    axes[1].tick_params(axis='x', rotation=85)

    plt.tight_layout()
    plt.show()

def perform_differential_expression_analysis(data):
    results = []
    time_points = {
        "2mo": (['WT1_2mo', 'WT2_2mo', 'WT3_2mo'], ['AD1_2mo', 'AD2_2mo', 'AD3_2mo']),
        "7mo": (['WT1_7mo', 'WT2_7mo', 'WT3_7mo'], ['AD1_7mo', 'AD2_7mo', 'AD3_7mo']),
        "8mo": (['WT1_8mo', 'WT2_8mo', 'WT3_8mo'], ['AD1_8mo', 'AD2_8mo', 'AD3_8mo']),
        "11mo": (['WT1_11mo', 'WT2_11mo', 'WT3_11mo'], ['AD1_11mo', 'AD2_11mo', 'AD3_11mo']),
        "14mo": (['WT1_14mo', 'WT2_14mo', 'WT3_14mo'], ['AD1_14mo', 'AD2_14mo', 'AD3_14mo']),
    }

    for time, (control_samples, treatment_samples) in time_points.items():
        control = data[control_samples]
        treatment = data[treatment_samples]
        
        for gene in data.index:
            stat, pval = ttest_ind(control.loc[gene], treatment.loc[gene], equal_var=False)
            results.append((gene, time, stat, pval))

    # Create a DataFrame from results
    results_df = pd.DataFrame(results, columns=['Gene_Name', 'Time_Point', 'Statistic', 'P_Value'])
    
    # Adjust p-values for multiple testing
    results_df['FDR'] = multipletests(results_df['P_Value'], method='fdr_bh')[1]
    
    return results_df

def gprofiler_analysis(results_df):
    gp = GProfiler(return_dataframe=True)
    enriched = gp.profile(organism='mmusculus', query=results_df['Gene_Name'].unique(), sources=['GO:BP'], no_evidences=True)
    
    # Merge enriched results with DE results
    enriched_results = results_df.merge(enriched, left_on='Gene_Name', right_on='query', how='left')
    return enriched_results

def main():
    # Download and prepare data
    data = download_and_prepare_data()
    
    # Rename columns and reorder
    data = rename_columns(data)
    data = reorder_columns(data)
    
    # Annotate genes
    data = annotate_genes(data)
    
    # Filter and normalize data
    normalized_data = filter_normalize(data)
    
    # Plot normalized data
    plot_normalized_data(normalized_data)
    
    # Perform differential expression analysis
    results_df = perform_differential_expression_analysis(normalized_data)
    
    # Perform GProfiler analysis
    enriched_results = gprofiler_analysis(results_df)
    
    # Save results
    results_df.to_csv("Differential_Expression_Results.csv", index=False)
    enriched_results.to_csv("GProfiler_Results.csv", index=False)

if __name__ == "__main__":
    main()
