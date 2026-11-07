# Alpha diversity
#================
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import io

def read_qza_tsv(qza_file):
    """Read the .tsv inside a .qza without qiime2"""
    with zipfile.ZipFile(qza_file, 'r') as z:
        # Find the .tsv file inside
        tsv_name = [n for n in z.namelist() if n.endswith('.tsv')][0]
        print(f"Reading {tsv_name} from {qza_file}")
        data = z.read(tsv_name)
        df = pd.read_csv(io.BytesIO(data), sep="\t", index_col=0)
        return df

# Loading 4 alpha files directly from .qza
shannon = read_qza_tsv("shannon_vector.qza")
observed = read_qza_tsv("observed_features_vector.qza")
faith = read_qza_tsv("faith_pd_vector.qza")
evenness = read_qza_tsv("evenness_vector.qza")

# Combining the files
alpha_df = pd.DataFrame({
    'shannon_entropy': shannon.iloc[:, 0],
    'observed_features': observed.iloc[:, 0],
    'faith_pd': faith.iloc[:, 0],
    'pielou_evenness': evenness.iloc[:, 0]
})

# Loading metadata
metadata = pd.read_csv("metadata.tsv", sep="\t", index_col=0)
df = alpha_df.join(metadata[['condition']])
print(df.head())
print(df.groupby('condition').mean())

# Plotting
metrics = ['shannon_entropy', 'observed_features', 'faith_pd', 'pielou_evenness']
titles = ['Shannon Diversity', 'Observed Features', 'Faith PD', 'Pielou Evenness']

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, metric, title in zip(axes, metrics, titles):
    sns.boxplot(data=df, x='condition', y=metric, ax=ax,
                palette={'healthy': '#2ecc71', 'precancerous': '#e74c3c'})
    sns.stripplot(data=df, x='condition', y=metric, ax=ax,
                  color='black', size=4, alpha=0.5, jitter=True)
    ax.set_xlabel("")
    healthy = df[df['condition']=='healthy'][metric].dropna()
    precancer = df[df['condition']=='precancerous'][metric].dropna()
    stat, p = mannwhitneyu(healthy, precancer)
    ax.set_title(f"{title}\np = {p:.3f}")

plt.tight_layout()
plt.savefig("alpha_diversity.png", dpi=300)
plt.show()

#==================
#Beta diversity
#==================
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import io

def read_qza_tsv(qza_file):
    with zipfile.ZipFile(qza_file, 'r') as z:
        tsv_name = [n for n in z.namelist() if n.endswith('.tsv')][0]
        data = z.read(tsv_name)
        df = pd.read_csv(io.BytesIO(data), sep="\t", index_col=0)
        return df

# Loading distance matrix directly from .qza
dist_df = read_qza_tsv("bray_curtis_distance_matrix.qza")
print(dist_df.shape)
print(dist_df.head())

metadata = pd.read_csv("metadata.tsv", sep="\t", index_col=0)
metadata = metadata.loc[dist_df.index]
dist_matrix = dist_df.values

# PCoA = PCA on distance matrix
pcoa = PCA(n_components=2)
coords = pcoa.fit_transform(dist_matrix)
var_exp = pcoa.explained_variance_ratio_ * 100
print(f"PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%")

# Plotting 
plt.figure(figsize=(8, 6))
colors = {'healthy': '#2ecc71', 'precancerous': '#e74c3c'}
for cond in ['healthy', 'precancerous']:
    idx = metadata['condition'] == cond
    plt.scatter(coords[idx, 0], coords[idx, 1],
                c=colors[cond], label=cond, s=80, alpha=0.8, edgecolors='black')

plt.xlabel(f"PC1 ({var_exp[0]:.1f}%)")
plt.ylabel(f"PC2 ({var_exp[1]:.1f}%)")
plt.title("Beta Diversity - Bray-Curtis PCoA")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("beta_pcoa.png", dpi=300)
plt.show()

# Save for PERMANOVA
dist_df.to_csv("bray_curtis_distance_matrix.csv")