# Activate QIIME 2 Environment
conda activate qiime2-amplicon-2024.10

#Import demultiplexed FASTQ files using manifest
qiime tools import \
  --type 'SampleData[PairedEndSequencesWithQuality]' \
  --input-path manifest.tsv \
  --output-path demux.qza \
  --input-format PairedEndFastqManifestPhred33V2

#Summarize Demultiplexed Sequences - Quality Check
qiime demux summarize \
  --i-data demux.qza \
  --o-visualization demux.qzv

#Denoise with DADA2 
 qiime dada2 denoise-paired \
  --i-demultiplexed-seqs demux.qza \
  --p-trunc-len-f 270 \
  --p-trunc-len-r 220 \
  --p-n-threads 1 \
  --p-n-reads-learn 200000 \
  --p-max-ee-f 2 \
  --p-max-ee-r 2 \
  --o-table table.qza \
  --o-representative-sequences rep-seqs.qza \
  --o-denoising-stats stats.qza

#Tabulating Denoising Statistics
qiime metadata tabulate \
  --m-input-file stats.qza \
  --o-visualization stats.qzv

#Tabulating Representative Sequences
qiime feature-table tabulate-seqs \
  --i-data rep-seqs.qza \
  --o-visualization rep-seqs.qzv

#Summarize Feature Table
 qiime feature-table summarize \
  --i-table table.qza \
  --o-visualization table.qzv \
  --m-sample-metadata-file metadata.tsv

# Phylogenetic Tree
 qiime phylogeny align-to-tree-mafft-fasttree \
  --i-sequences rep-seqs.qza \
  --o-alignment aligned-rep-seqs.qza \
  --o-masked-alignment masked-aligned-rep-seqs.qza \
  --o-tree unrooted-tree.qza \
  --o-rooted-tree rooted-tree.qza

#Taxonomic Classification - SILVA V1-V2 with VSEARCH
qiime feature-classifier classify-consensus-vsearch \
  --i-query rep-seqs.qza \
  --i-reference-reads silva-v1v2-seqs.qza \
  --i-reference-taxonomy /mnt/d/qiime2_classifiers/silva-138-99-tax.qza \
  --p-perc-identity 0.8 \
  --p-query-cov 0.8 \
  --p-strand both \
  --p-maxaccepts 5 \
  --o-classification taxonomy-V1V2.qza \
  --o-search-results search-results.qza \
  --verbose

 qiime taxa barplot --i-table table.qza --i-taxonomy taxonomy-V1V2.qza --o-visualization barplot-V1V2.qzv

#  Filtering Contaminant Features
qiime taxa filter-table \
  --i-table table.qza \
  --i-taxonomy taxonomy-V1V2.qza \
  --p-exclude mitochondria,chloroplast \
  --o-filtered-table table-filtered.qza

#Diversity Analysis after Filtering
qiime diversity core-metrics-phylogenetic \
  --i-phylogeny rooted-tree.qza \
  --i-table table-filtered.qza \
  --p-sampling-depth 10000 \
  --m-metadata-file metadata.tsv \
  --output-dir core-metrics-filtered

# Exporting Data for Python usage
 mkdir python_data
qiime tools export --input-path table-filtered.qza --output-path python_data
qiime tools export --input-path taxonomy-V1V2.qza --output-path python_data
qiime tools export --input-path core-metrics-filtered/bray_curtis_pcoa_results.qza --output-path python_data