# 🧬 RNA-Seq Signature Miner

This pipeline selects gene expression signatures from RNA-Seq data using LASSO regression. It then evaluates multiple classification models using performance metrics and visualizations such as ROC curves and LDA plots.

---
## ✅ PREQUESTIES 
Before proceeding, please ensure the following prerequisites are met:

### 1) 📄 Normalized Expression Data

  You must have a normalized gene expression dataset (normalised_values.xlsx).

  ✅ Confirm that your gene expression data has been normalized, using one of the following methods:  TPM (Transcripts Per Million),  RPKM (Reads Per Kilobase Million),  CPM (Counts Per Million), Log-transformed counts

  ⚠️ If your data is not normalized, please preprocess your raw count data to generate a normalized expression matrix before continuing.
  
### 2) 📋 Sample Metadata

You must provide a metadata file (e.g., Sampleinfo.xlsx) that includes relevant sample information:
    🆔 Sample IDs,     🧪 Condition or Experimental Group (e.g., Control, Treated)

 ### 3) 🛠️ Git

Git is required to clone the project repository.  📥 Download and install Git: https://git-scm.com/downloads
 
 ### 4) 🐳 Docker

Docker is required to build and run the containerized environment. 📥 Install Docker: https://docs.docker.com/engine/install/



## 📦 Usage Instructions (via Docker)

### 1. Clone the repository
```
git clone https://github.com/bioinfokushwaha/RnaSeqSignatureMiner.git
```

### 2. Move into the scripts folder
```
cd RnaSeqSignatureMiner/scripts
```
### 3. Create docker container
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
