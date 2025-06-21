# 🧬 RNA-Seq Signature Miner

This pipeline selects gene expression signatures using LASSO, and evaluates classification models using multiple metrics and visualizations like ROC and LDA plots.

---

## 📦 Step-by-Step Usage (via Docker)

### 1. Clone the Repository

```bash
git clone https://github.com/bioinfokushwaha/RnaSeqSignatureMiner.git
cd RnaSeqSignatureMiner/scripts 




docker build -t rnaseq_signature_miner .
docker run --rm -v "$PWD":/app rnaseq_signature_miner


scripts/
│
├── Dockerfile
├── main.py
├── main.sh
├── normalised_values.xlsx       # input
├── Sampleinfo.xlsx              # input

