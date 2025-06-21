# 🧬 RNA-Seq Signature Miner

This pipeline selects gene expression signatures from RNA-Seq data using LASSO regression. It then evaluates multiple classification models using performance metrics and visualizations such as ROC curves and LDA plots.

---

## 📦 Usage Instructions (via Docker)

# 1. Clone the repository
```
git clone https://github.com/bioinfokushwaha/RnaSeqSignatureMiner.git
```

# 2. Move into the scripts folder
```
cd RnaSeqSignatureMiner/scripts
```
# 3. create doaker container
```
docker build -t rnaseq_signature_miner
```
# 4. Run the Docker build and container
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
