import pandas as pd
import biom
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import mannwhitneyu, entropy
from statsmodels.stats.multitest import multipletests

print("--- 1. LOADING DATA ---")
table = biom.load_table("feature-table.biom")
df = pd.DataFrame(table.matrix_data.toarray().T,
                  index=table.ids(axis='sample'),
                  columns=table.ids(axis='observation'))
print(f"Loaded: {df.shape[0]} samples, {df.shape[1]} ASVs")

metadata = pd.read_csv("metadata.tsv", sep='\t', index_col=0)
taxonomy = pd.read_csv("taxonomy.tsv", sep='\t', index_col=0)
taxonomy['genus'] = taxonomy['Taxon'].str.extract(r'g__([^;]+)')
taxonomy['genus'] = taxonomy['genus'].fillna('Unassigned')

# Relative abundance
rel = df.div(df.sum(axis=1), axis=0)

# ASV -> Genus aggregation
rel_genus = rel.copy()
rel_genus.columns = taxonomy.loc[rel_genus.columns, 'genus'].values
rel_genus = rel_genus.T.groupby(rel_genus.columns).sum().T

common = rel_genus.index.intersection(metadata.index)
rel_genus = rel_genus.loc[common]
metadata = metadata.loc[common]
print(metadata['condition'].value_counts())
print(f"Genus table: {rel_genus.shape} (samples x genera)")

healthy_idx = metadata[metadata['condition']=='healthy'].index
prec_idx = metadata[metadata['condition']=='precancerous'].index
healthy_df = rel_genus.loc[healthy_idx]
prec_df = rel_genus.loc[prec_idx]

# --- PREVALENCE FILTER ---
# Keeping only genera present in >=20% of samples AND mean abundance >0.1%

prevalence = (rel_genus > 0).mean()  # % samples where genus present
mean_abund = rel_genus.mean()

keep = (prevalence >= 0.20) & (mean_abund >= 0.001)
print(f"Before filter: {rel_genus.shape[1]} genera")
print(f"After filter: {keep.sum()} genera kept")
print("Removed:", rel_genus.columns[~keep].tolist()[:10], "...")

rel_genus_filtered = rel_genus.loc[:, keep]

rel_genus = rel_genus_filtered

# ============================================================
# --- 2. STATISTICAL ANALYSIS ---
# ============================================================
print("\n--- 2. STATS: Differential Abundance (Mann-Whitney + Log2FC + BH) ---")
results = []
for genus in rel_genus.columns:
    h = healthy_df[genus].values
    p = prec_df[genus].values
    mean_h = h.mean()
    mean_p = p.mean()
    median_h = np.median(h)
    median_p = np.median(p)
    # Log2FC: Precancerous / Healthy
    log2fc = np.log2((mean_p + 1e-6) / (mean_h + 1e-6))
    try:
        _, pval = mannwhitneyu(h, p, alternative='two-sided')
    except:
        pval = 1.0
    results.append([genus, mean_h, mean_p, median_h, median_p, log2fc, pval])

stats_df = pd.DataFrame(results, columns=['Genus','Mean_Healthy','Mean_Precancerous','Median_Healthy','Median_Precancerous','Log2FC','p_value'])
stats_df['q_value_BH'] = multipletests(stats_df['p_value'], method='fdr_bh')[1]
stats_df = stats_df.sort_values('p_value')
stats_df.to_csv("differential_abundance_results.csv", index=False)

sig_p = stats_df[stats_df['p_value'] < 0.05]
sig_q = stats_df[stats_df['q_value_BH'] < 0.05]
print(f"Genera tested: {len(stats_df)} | p<0.05: {len(sig_p)} | q<0.05 BH: {len(sig_q)}")
print(stats_df.head(15).to_string(index=False))

# Choosing genera for plotting
if len(sig_p) >= 5:
    plot_genera = sig_p.head(10)['Genus'].tolist()
    plot_label = f"Significant Genera (Mann-Whitney p<0.05, n={len(sig_p)})"
else:
    plot_genera = stats_df.head(10)['Genus'].tolist()
    plot_label = "Top 10 Genera by p-value (no BH significance - trend only)"

print(f"\nGenera used for plots: {plot_genera}")

# =================
# --- 3. PLOTS ---
# =================

# 3a. VOLCANO PLOT 
plt.figure(figsize=(9,6))
plt.scatter(stats_df['Log2FC'], -np.log10(stats_df['p_value']+1e-300), c='grey', alpha=0.6, s=30, label='Not significant')
plt.scatter(sig_p['Log2FC'], -np.log10(sig_p['p_value']+1e-300), c='orange', s=60, label='p < 0.05')
if len(sig_q) > 0:
    plt.scatter(sig_q['Log2FC'], -np.log10(sig_q['p_value']+1e-300), c='red', s=80, label='q < 0.05 (BH)')
for _, row in stats_df.head(8).iterrows():
    plt.text(row['Log2FC'], -np.log10(row['p_value']+1e-300)+0.15, row['Genus'], fontsize=9, ha='center')
plt.axhline(-np.log10(0.05), ls='--', c='black', alpha=0.5)
plt.axvline(0, ls='--', c='black', alpha=0.5)
plt.axvline(1, ls=':', c='blue', alpha=0.4)
plt.axvline(-1, ls=':', c='blue', alpha=0.4)
plt.xlabel("Log2 Fold Change (Precancerous / Healthy)")
plt.ylabel("-log10(p-value)")
plt.title("Volcano Plot - Differential Abundance (Mann-Whitney + BH)")
plt.legend()
plt.tight_layout()
plt.savefig("volcano_plot_STATS.png", dpi=300)
plt.show()

#3b. HEATMAP
from scipy.stats import zscore
rel_sig_z = rel_genus[plot_genera].apply(zscore, axis=0) # scale each genus to mean 0, sd 1
rel_sig_z = rel_sig_z.loc[metadata.sort_values('condition').index]

plt.figure(figsize=(10,6))
sns.heatmap(rel_sig_z.T, cmap='RdBu_r', vmin=-2, vmax=2, xticklabels=False, center=0)
plt.title("Heatmap - Z-scored (per genus) - Red = Higher than average, Blue = Lower")
plt.ylabel("Genus")
plt.tight_layout()
plt.savefig("heatmap_ZSCORED.png", dpi=300)
plt.show()

# 3c. BARPLOT 
plot_means = rel_genus[plot_genera].groupby(metadata['condition']).mean().T
# Order by Log2FC for nice visual
order = stats_df.set_index('Genus').loc[plot_genera].sort_values('Log2FC').index.tolist()
plot_means = plot_means.loc[order]

fig, ax = plt.subplots(figsize=(10,6))
plot_means.plot(kind='bar', ax=ax, color=['#2ca02c','#d62728'])
plt.title(f"Healthy vs Precancerous - {plot_label}")
plt.ylabel("Mean Relative Abundance")
plt.xlabel("Genus (ordered by Log2FC)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("barplot_healthy_vs_precancer_STATS.png", dpi=300)
plt.show()




# ================================
# --- 4. MACHINE LEARNING LAST ---
# ================================
print("\n--- 6. ML (Genus-level, 5-Fold CV) ---")
X = rel_genus.values
y = metadata['condition'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print(f"Logistic Regression Test: {accuracy_score(y_test, lr.predict(X_test)):.2f}")
print(classification_report(y_test, lr.predict(X_test)))

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
print(f"Decision Tree Test: {accuracy_score(y_test, dt.predict(X_test)):.2f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"Logistic CV: {cross_val_score(lr, X, y, cv=cv).mean():.2f}")
print(f"Decision Tree CV: {cross_val_score(dt, X, y, cv=cv).mean():.2f}")

# Final merged biomarker table: ML + Stats
ml_imp = pd.DataFrame({'Genus': rel_genus.columns, 'ML_Importance': np.abs(lr.coef_[0])})
final = pd.merge(ml_imp, stats_df, on='Genus', how='left').sort_values('ML_Importance', ascending=False)
final.to_csv("biomarker_final_table_ML_plus_stats.csv", index=False)
print("\n--- FINAL BIOMARKER TABLE: ML Importance + Log2FC + p + q ---")
print(final.head(15)[['Genus','ML_Importance','Mean_Healthy','Mean_Precancerous','Log2FC','p_value','q_value_BH']].to_string(index=False))
print("\nAll files saved: differential_abundance_results.csv, volcano_plot_STATS.png, barplot_healthy_vs_precancer_STATS.png, heatmap_significant_genera_STATS.png, alpha_diversity_stats.csv, biomarker_final_table_ML_plus_stats.csv")