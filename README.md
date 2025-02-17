# 📊 CBIO Hackathon - Predicting Antibiotic Resistance

## 🧑‍🤝‍🧑 Group Members
- **Fahim Francis:** 209177575, fahim.francis@mail.huji.ac.il
- **Asmaa Ghrayeb:** 212017719, asmaa.ghrayeb@mail.huji.ac.il
- **Adam Fattum:** 325156842, adam.fattum@mail.huji.ac.il
- **Maryan Kiwan:** 212514681, maryan.kiwan@mail.huji.ac.il
- **Mustafa Shouman:** 212092613, mustafa.shouman@mail.huji.ac.il

## 📝 Overview
Using machine learning to predict antibiotic resistance from raw genomic sequences with k-mer analysis on MEGARes data for Beta-lactamase (BLUE) and Aminoglycoside (RED) resistance.

## 💾 Data and Preprocessing
- **Source:** MEGARes Database  
- **Classes:** Beta-lactamase (BLUE), Aminoglycoside (RED)  
- **Feature Extraction:** k-mer transformation (k=2,3,4,5)  
- **Preprocessing:** Stratified split, class weighting, SMOTE oversampling

## 🤖 Models and Training
1. **Logistic Regression:** Baseline model with regularization tuning  
2. **SVM (RBF Kernel):** Non-linear pattern detection (Final Model: k=4)  
3. **Random Forest:** Feature importance analysis

## 📊 Performance Summary
| k-mer | Model           | Accuracy |
|------|-----------------|----------|
| 2    | Random Forest   | 0.873    |
| 3    | Random Forest   | 0.878    |
| 4    | SVM (Final)    | 0.906    |
| 5    | SVM            | 0.891    |

**Final Model (SVM, k=4) Metrics:**  
- Balanced Accuracy: 0.919  
- Precision: 0.992  
- Recall: 0.839  
- ROC AUC: 0.988

## 📈 Key Insights
- **Top k-mers (BLUE):** CAGC, GAGG, AGGT  
- **Top k-mers (RED):** CGAG, GAGG, CTTG  
- **Largest Differences:** GCAA, TGTC, CAGC  
- **Most Significant:** CAGC (BLUE), CGAG (RED)

## 📂 Repository Structure
```bash
📂 data/
   ├── megares_resistant.fasta
   └── megares_non_resistant.fasta
📂 src/
   ├── preprocess.py
   ├── train_models.py
   ├── evaluate.py
   └── visualize.py
📂 results/
   └── model_performance.csv
📂 images/
   ├── roc_curves.png
   └── top_kmers.png
📄 README.md
📄 requirements.txt
