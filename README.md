# 🧬 RNA-Seq Signature Miner

This pipeline selects gene expression signatures from RNA-Seq data using LASSO regression. It then evaluates multiple classification models using performance metrics and visualizations such as ROC curves and LDA plots.

---
##  PREQUESTIES 
Before proceeding, please ensure the following prerequisites are met:

  1)  Normalized Expression Data (normalised_values.xlsx)

        i)Confirm that your gene expression dataset has been normalized (e.g., TPM, RPKM, CPM, or log-transformed counts).

        ii) If normalization has not yet been performed, please preprocess your raw count data to generate a normalized expression matrix.

   2) Sample Metadata (Sampleinfo.xlsx)

        A sample information file (e.g., sample_metadata.csv) must be prepared, containing relevant metadata such as:
            i) Sample IDs

       ii) Condition/Group


## 📦 Usage Instructions (via Docker)

### 1. Clone the repository
```
git clone https://github.com/bioinfokushwaha/RnaSeqSignatureMiner.git
```

### 2. Move into the scripts folder
```
cd RnaSeqSignatureMiner/scripts
```
### 3. create doaker container
```
docker build -t rnaseq_signature_miner .
```
### 4. Copy the xlsx files to  scripts folder
```
cp /path/to/folder/normalised_values.xlsx ../
cp /path/to/folder/Sampleinfo.xlsx ../
```
### 5. Run the Docker build and container
```
docker run --rm -v "$PWD":/app rnaseq_signature_miner
````

## 📁 Project Directory Structure
```
scripts/
│
├── Dockerfile
├── main.py
├── main.sh
├── normalised_values.xlsx       # input
├── Sampleinfo.xlsx              # input
```
