## End to end machine learning project

# 🎓 Student Exam Performance Prediction

A Machine Learning web application that predicts a student’s **Math score** based on academic and demographic inputs using regression models and a Flask web interface.

---

## 📌 Project Description

This project analyzes how factors such as **gender, race/ethnicity, parental education, lunch type, test preparation course, reading score, and writing score** affect a student’s math performance.

It follows a complete **end-to-end Machine Learning pipeline**:
- Data ingestion
- Data transformation
- Model training & evaluation
- Model selection
- Web deployment using Flask

---

## 🧠 Machine Learning Pipeline

1. **Data Ingestion**
   - Load dataset
   - Train–test split

2. **Data Transformation**
   - Missing value handling
   - Feature scaling
   - Categorical encoding
   - Save preprocessing object

3. **Model Training**
   - Multiple regression models
   - Hyperparameter tuning (GridSearchCV)
   - Best model selection using R² score

4. **Prediction**
   - User inputs via web form
   - Real-time Math score prediction

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- CatBoost  
- Flask  
- HTML & CSS  

---

## 📂 Project Structure

```
mlproject/
│
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   └── prediction.html
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Training Pipeline
```bash
python src/components/data_ingestion.py
```

### 4️⃣ Start Flask App
```bash
python app.py
```

### 5️⃣ Open Browser
```
http://127.0.0.1:5000/
```

---

## 📊 Models Used

- Linear Regression  
- Ridge & Lasso Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- CatBoost  
- AdaBoost  
- K-Nearest Neighbors  

---

## 📈 Evaluation Metric

- **R² Score**

---

## 🖥️ Web Features

- User-friendly interface  
- Prediction form  
- Real-time output  
- Clean UI with CSS  

---

## 👨‍💻 Author

**Bhargav**  
Aspiring Data Scientist  

---

## ⭐ Future Enhancements

- Cloud deployment  
- Model performance visualization  
- Docker integration  
- Improved UI  
