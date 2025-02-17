## 📊 **CBIO Hackathon: Predicting Antibiotic Resistance** 
### 📌 **Project Overview**  
This repository contains the code and models developed during the CBIO Hackathon for predicting antibiotic resistance from raw genomic sequences using machine learning techniques. We applied k-mer transformations and compared the performance of Logistic Regression, SVM, and Random Forest models to classify resistant and non-resistant bacterial strains.  

### 👥 **Team Members**  
- **Faheem Francis:** 209177575, fahim.francis, [fahim.francis@mail.huji.ac.il](mailto:fahim.francis@mail.huji.ac.il)  
- Asmaa Ghrayeb: 212017719, asmaa.ghrayeb, [asmaa.ghrayeb@mail.huji.ac.il](mailto:asmaa.ghrayeb@mail.huji.ac.il)  
- Adam Fattum: 325156842, adam_307, [adam.fattum@mail.huji.ac.il](mailto:adam.fattum@mail.huji.ac.il)  
- Maryan Kiwan: 212514681, maryan.kiwan, [maryan.kiwan@mail.huji.ac.il](mailto:maryan.kiwan@mail.huji.ac.il)  
- Mustafa Shouman: 212092613, mustafashouman, [mustafa.shouman@mail.huji.ac.il](mailto:mustafa.shouman@mail.huji.ac.il)  

---

### 🛠️ **Project Structure**  



### 📈 **Machine Learning Models Used**  
- 🧪 **Logistic Regression:** Baseline model for binary classification.  
- 🧠 **Support Vector Machine (SVM):** Used RBF kernel for non-linear pattern detection.  
- 🌲 **Random Forest:** Provided feature importance and handled large feature spaces effectively.  

---

### 📊 **Model Performance**  
| **Model**           | **k-mer Size** | **Balanced Accuracy** | **F1 Score** | **ROC AUC** |
|---------------------|---------------|----------------------|-------------|------------|
| Logistic Regression | 4             | 0.505              | 0.412       | 0.601      |
| SVM (RBF)          | 4             | **0.919**          | **0.909**   | **0.988**  |
| Random Forest       | 3             | 0.878              | 0.896       | 0.965      |

---

### 🧬 **K-mer Analysis Highlights**  
- The **SVM model with k=4** achieved the best performance with a Balanced Accuracy of **0.919**.  
- **Random Forest** revealed important k-mers, such as `CAGC`, `GAGG`, and `AGGT`, strongly associated with resistance.  
- Key statistical insights: `CAGC` had the lowest p-value and a fold increase of 1.84x in resistant strains.  

---

### ⚙️ **How to Run the Project**  

#### ✅ **Step 1: Clone the Repository**  
```bash
git clone https://github.com/your-github/cbio-antibiotic-resistance.git
cd cbio-antibiotic-resistance
⚙️ How to Run the Project
✅ Step 1: Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-github/cbio-antibiotic-resistance.git
cd cbio-antibiotic-resistance
✅ Step 2: Create a Virtual Environment
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
✅ Step 3: Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
✅ Step 4: Run the Models
bash
Copy
Edit
python src/model.py
✅ Step 5: View Results
Results will be saved in the /results directory, including performance metrics and confusion matrices.
