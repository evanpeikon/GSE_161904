# 🧬  Project Introduction: 
## Investigating Transcriptional Changes in Alzheimer's Disease Using 3xTg-AD Mouse Model

Alzheimer’s disease is characterized by progressive cognitive impairment and is associated with complex pathophysiological changes, including amyloid deposition, neuroinflammation, and dysregulated gene expression. 

In this repository, I'll present an analysis of bulk RNA-sequencing data from the 3xTg-AD mouse model, a widely used model for studying Alzheimer’s disease. The objective for this project is to investigate the molecular changes that occur in the insular cortex as the disease advances, with a focus on identifying key transcriptional regulators and pathways that contribute to cognitive decline. The insular cortex is particularly relevant because it is involved in critical functions like emotion regulation, sensory processing, and autonomic control, all of which can be affected in AD.

By leveraging RNA-seq data, we can elucidate the transcriptional landscape associated with neurodegeneration in the 3xTg-AD model. This analysis will help connect changes in gene expression with the functional outcomes seen in Alzheimer’s disease, providing insights into the underlying mechanisms of cognitive decline.

# 🧬 Project Walkthrough 
## Data Availability

The data for this project comes from a study called "Transcriptional, behavioural and biochemical profiling in the 3xTg-AD mouse model reveals a specific signature of amyloid deposition and functional decline in Alzheimer’s disease," which is accessible via the Gene Expression Omnibus (GEO) via the following accession number: [GSE161904](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161904).

In this project we'll analyze bulk RNA-sequencing data from the insular cortex of three 3xTG-AD trasgenic mice and three wild type mice control littermates at five discrete time points (2, 7, 8, 11, and 14 months of age). Additional information about this dataset is provided below:

- **GEO Accension Number:** GSE161904
- **Data type:** Bulk RNA-seq
- **Tissue type:** Insular cortex
- **Mouse Strains:** 3xTg-AD transgenic mice and wid type mice (WT)
- **Age:** Mice were sampled at 2, 7, 8, 11, and 14 months old. 
- **Genome build:** mm10
- **Total Samples:** 30 mice (3 transgenic mice x 3 WT mice x 5 time points)

**Note:** To collect samples mice were euthanized. Following euthanization the insular cortex was bilaterally dissected and tissue from both hemispheres was aggregated into a single sample per animal. Over the course of the study samples were taken from a total of 30 male mice, including three 3xTG-AD mutant mice and wild type mice at each postnatal stage (2,7,8,11, and 14 months). 

## Import Libraries

Before loading the data above, I'll first import the following Python libraries, which will be used in downstream analyses:

```python
import pandas as pd
import mygene
import numpy as np
from gprofiler import GProfiler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
```

## Load, Inspect, and Prepare Data

Next, we'll use Bash's ```wget``` command to retrieve our bulk RNA-seq data, and following that, we'll use the Bash ```gunzip``` command to decompress the files:

```bash
!wget -O GSE161904_Raw_gene_counts_cortex.txt.gz 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE161904&format=file&file=GSE161904%5FRaw%5Fgene%5Fcounts%5Fcortex%2Etxt%2Egz'

!gunzip GSE161904_Raw_gene_counts_cortex.txt.gz
```

Following that, we'll load the data into a Pandas DataFrame and inspect the first 5 rows of data.

```python
# Load the data from the txt file and convert it to a CSV
data = pd.read_csv("GSE161904_Raw_gene_counts_cortex.txt", sep="\t")
data.to_csv("GSE161904_Raw_gene_counts_cortex.csv", index=False)
data.head()
```

<img width="1328" alt="Screenshot 2024-10-31 at 7 05 11 PM" src="https://github.com/user-attachments/assets/41a842d9-26c5-4be0-a7b3-fecf2483433d">

The data above is indexed by Ensemble gene ID (ENSMUSG) with 30 columns of RNA-sequencing expression data (i.e., counts). Before performing downstream analyses we'll want to convert the Ensemble gene IDs to gene names, rename the columns based on the sample group (AD vs. WT) and time points (2,7,8,11, or 14mo), then rearrange the columns so they are ordered by sequentially and by group. 

First, we'll begin by renaming and reordering the columns, as demonstrated in the code block below:

```python
# define the current column names and new column names
current_columns = ["G3R1_Cortex_3xTgAD", "G1R2_Cortex_3xTgAD", "G1R3_Cortex_3xTgAD", "G1R4_Cortex_3xTgAD","G1R5_Cortex_WT", "G1R6_Cortex_WT", "G1R7_Cortex_WT", "G4R1_Cortex_3xTgAD","G4R2_Cortex_3xTgAD", "G4R3_Cortex_3xTgAD", "G3R3_Cortex_3xTgAD", "G4R5_Cortex_WT","G4R6_Cortex_WT", "G4R7_Cortex_WT", "G5R1_Cortex_3xTgAD", "G5R2_Cortex_3xTgAD","G5R3_Cortex_3xTgAD", "G5R4_Cortex_WT", "G5R6_Cortex_WT", "G5R5_Cortex_WT","G3R4_Cortex_3xTgAD", "G2R4_Cortex_WT", "G2R5_Cortex_3xTgAD", "G2R6_Cortex_3xTgAD","G2R7_Cortex_3xTgAD", "G3R10_Cortex_WT", "G2R2_Cortex_WT", "G2R3_Cortex_WT","G3R7_Cortex_WT", "G3R9_Cortex_WT"]

new_columns = ["AD1_8mo", "AD1_2mo", "AD2_2mo", "AD3_2mo", "WT1_2mo", "WT2_2mo", "WT3_2mo","AD1_11mo", "AD2_11mo", "AD3_11mo", "AD2_8mo", "WT1_11mo", "WT2_11mo", "WT3_11mo","AD1_14mo", "AD2_14mo", "AD3_14mo", "WT1_14mo", "WT2_14mo", "WT3_14mo", "AD3_8mo","WT3_7mo", "AD1_7mo", "AD2_7mo", "AD3_7mo", "WT3_8mo", "WT1_7mo", "WT2_7mo","WT1_8mo", "WT2_8mo"]

# create dictionary to map current columns names to new column names 
column_mapping = dict(zip(current_columns, new_columns))

# rename the columns 
data.rename(columns=column_mapping, inplace=True)

# define the desired column order 
new_column_order = [
    "WT1_2mo", "WT2_2mo", "WT3_2mo", "AD1_2mo", "AD2_2mo", "AD3_2mo",
    "WT1_7mo", "WT2_7mo", "WT3_7mo", "AD1_7mo", "AD2_7mo", "AD3_7mo",
    "WT1_8mo", "WT2_8mo", "WT3_8mo", "AD1_8mo", "AD2_8mo", "AD3_8mo",
    "WT1_11mo", "WT2_11mo", "WT3_11mo", "AD1_11mo", "AD2_11mo", "AD3_11mo",
    "WT1_14mo", "WT2_14mo", "WT3_14mo", "AD1_14mo", "AD2_14mo", "AD3_14mo"]

# reorder columns 
data = data[new_column_order]
```

Next, we'll get the correspondong gene names for each Ensemble gene ID in our dataset, then add said gene names to a new column. Then, we'll drop the current index in the DataFrame containing the Ensemble gene IDs and make the first column, containing gene names, our new index, as demonstrated below:

```python
# create MyGeneInfo object
mg = mygene.MyGeneInfo()

# get the ensembl id from index
ensembl_ids = data.index.tolist()  

# query the gene symbols for the ensemble ids and onvert result to dataframe
gene_info = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='mouse')
gene_df = pd.DataFrame(gene_info)

# remove duplicate ensemble ids and rows where symbol is missing or duplicated
gene_df = gene_df.dropna(subset=['symbol']).drop_duplicates(subset='query')

# map gene symbols back to original dataframe and move gene_name column to front column
data['Gene_Name'] = data.index.map(gene_df.set_index('query')['symbol'])
cols = ['Gene_Name'] + [col for col in data.columns if col != 'Gene_Name']
data = data[cols]

# drop current index w/ ensemble IDs, then make column 0 w/ gene names the new index
data = data.reset_index(drop=True)
data = data.set_index(data.columns[0])
```

Now, we'll display the results to ensure our data frame is indexed by gene ID and that the columns are properly renamed and rearranged:

```python
# diplay first 5 rows of data
data.head()
```

<img width="1259" alt="Screenshot 2024-10-31 at 7 07 06 PM" src="https://github.com/user-attachments/assets/eb688e2d-7f0a-4f79-86ca-f2f2f229582a">

Next, we'll check for missing data and perform basic data exploration to understand the distribution and variability of RNA sequencing counts across the samples before performing any downstream analysis. First, let's check out our sample quality:

```python
# check for missing values 
print(data.isnull().sum())
```

<img width="100" alt="Screenshot 2024-10-31 at 7 07 46 PM" src="https://github.com/user-attachments/assets/9e6e76ed-c73d-4640-82ff-c0a5fb0e2b59">

Notably, the dataset has no null (missing) values. Next, we'll explore the distribution and variability in our dataset, as demonstrated in the code block below:

```python
# calcualte total counts per sample and log transform counts
total_counts = data.sum(axis=0)
log_counts = data.apply(lambda x: np.log2(x + 1))

# create subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# plot total counts per sample
axes[0].bar(data.columns, total_counts, color='skyblue')
axes[0].set_ylabel('Total Counts')
axes[0].set_title('Total Counts per Sample')
axes[0].tick_params(axis='x', rotation=85)

# plot log transformed counts per sample
log_counts.boxplot(ax=axes[1])
axes[1].set_ylabel('Log2(Counts + 1)')
axes[1].set_title('Log Transformed Counts per Sample')
axes[1].tick_params(axis='x', rotation=85)

plt.tight_layout()
plt.show()
```

