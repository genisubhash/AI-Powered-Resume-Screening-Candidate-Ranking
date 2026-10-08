[README.md](https://github.com/user-attachments/files/24094733/README.md)
# 🤖 AI Powered Resume Screening & Candidate Ranking System

An intelligent machine-learning based application that automatically analyzes resumes, predicts the best-fit job role, extracts technical skills, and produces a **candidate ranking score** to help recruiters shortlist applicants faster and more accurately.

---

## 🚀 Features

### ✅ 1. Resume Classification  
Predicts the most suitable job category using:
- TF-IDF Vectorization  
- Logistic Regression Model  

### ✅ 2. Candidate Ranking Score  
A confidence score that represents how well the resume matches the predicted job category.

### ✅ 3. Automated Skill Extraction  
Identifies technical skills from the resume using keyword-based skill detection.

### ✅ 4. Required Skill Analysis  
Shows:
- ✔ Matched Skills  
- ❌ Missing Skills  
Based on predefined job-role skill requirements.

### ✅ 5. Optional: Job Description (JD) Matching  
If a JD is uploaded, the system provides:
- Skill Match %  
- Textual Similarity %  
- **ATS Compatibility Score** (70% skills + 30% similarity)

### ✅ 6. Fast & User-Friendly Interface  
Built with **Streamlit** for real-time predictions and a modern UI.

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|------------|
| Programming | Python |
| ML Model | Logistic Regression |
| Vectorization | TF-IDF |
| Frontend / UI | Streamlit |
| File Parsing | PyPDF2 |
| Similarity Engine | Cosine Similarity |
| Optional Embeddings | Sentence Transformers |
| Data Processing | Pandas, NumPy |

---

## 📂 Project Structure

```
AI-Resume-Ranking/
│
├── app.py                 # Main Streamlit application
├── models/
│   ├── clf.pkl            # Trained ML model
│   ├── tfidf.pkl          # TF-IDF vectorizer
│
├── dataset/
│   └── resumes.csv        # Training dataset (Resume + Category)
│
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/ai-resume-ranking.git
cd ai-resume-ranking
```

### 2️⃣ Install required libraries
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📝 How the Model Works

1. Extracts text from PDF/TXT resume  
2. Cleans text (removes special characters, lowercases, etc.)  
3. Converts resume into a TF-IDF vector  
4. Logistic Regression predicts job category  
5. Highest probability → Candidate Ranking Score  
6. Extracted skills are matched with expected skills for that category  
7. If JD provided → ATS Score (skill match + similarity)

---

## 📊 Output Provided by the System

- 🏷 **Predicted Job Category**  
- ⭐ **Candidate Ranking Score (%)**  
- 🧠 **Extracted Skills**  
- ❗ **Missing Skills**  
- 📊 **JD Match Score (optional)**  
  - Skill Match %  
  - Text Similarity %  
  - ATS Score  
- ⏱ **Prediction Time**

---

## 🏋️ Training the Model

You can train your model from the app using:

### Dataset Format:
| Column | Description |
|--------|-------------|
| Resume | Resume text |
| Category | Job role label |

### Steps:
1. Go to **Train Model** tab  
2. Upload your `.csv` file  
3. Click **Train Model**  
4. The trained model gets saved automatically (clf.pkl & tfidf.pkl)

---

## 🔮 Future Enhancements

- BERT-based classification for deeper contextual understanding  
- Resume parsing (education, experience timeline)  
- OCR for scanned resumes  
- Multi-resume batch uploading  
- Automated shortlisting with threshold filters  
- Cloud deployment (AWS / GCP / Azure)

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 👤 Author
**Subhash Babu Geni**  
Python , AI & Machine Learning Developer

If you like this project, consider giving it a ⭐ on GitHub!
