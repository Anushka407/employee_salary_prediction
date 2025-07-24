# 💼 Employee Salary Prediction App

This project is a **Deep Learning-based web application** that predicts whether an individual's annual income is **greater than or less than $50K** based on various demographic and employment features. The app is built with **Streamlit** and uses a trained **TensorFlow Keras model** under the hood.
  
## **Developer:** [Anushka Shree](https://github.com/Anushka407)


## 🧠 Model Overview

- **Model Type:** Deep Neural Network (Keras Sequential)
- **Framework:** TensorFlow / Keras
- **Input Features:** 13 attributes including `age`, `education`, `workclass`, etc.
- **Output:** Binary classification (`>50K`, `<=50K`)
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Scaler Used:** StandardScaler


## 📊 Input Features

| Feature           | Type         | Example                        |
|------------------|--------------|--------------------------------|
| Age              | Numeric      | 30                             |
| Workclass        | Categorical  | Private, Govt, Self-employed   |
| fnlwgt           | Numeric      | 125000                         |
| Education        | Categorical  | Bachelors, Masters, etc.       |
| Educational Num  | Numeric      | 13                             |
| Marital Status   | Categorical  | Married, Never-married         |
| Occupation       | Categorical  | Tech-support, Exec-manager     |
| Relationship     | Categorical  | Husband, Not-in-family         |
| Race             | Categorical  | White, Black, Asian            |
| Gender           | Categorical  | Male, Female                   |
| Capital Gain     | Numeric      | 0, 10000                       |
| Capital Loss     | Numeric      | 0, 2000                        |
| Hours per Week   | Numeric      | 40                             |
| Native Country   | Categorical  | United-States, India, Others   |

---

## 🖥️ Web App Features

✅ Clean user interface with form-based input  
✅ Real-time prediction of salary group  
✅ Insights tab for interpretation and tips  
✅ Dark-themed custom layout  
✅ Responsive and professional design  
✅ Footer with developer credit


## 🧪 Requirements.txt
streamlit==1.32.2
pandas==2.2.2
scikit-learn==1.5.1
tensorflow==2.19.0
joblib==1.3.2
numpy==1.26.4

## 📈 Model Training Summary
Epochs: 20
Accuracy: ~85% (Validation)
Loss Curve: Converges smoothly after 10 epochs
Confusion Matrix: Balanced prediction with decent precision

## ☁️ Deployment Options
Streamlit Community Cloud
  

##
👩‍💻 Anushka Shree
🎓 MCA Final Year | Frontend & AI Enthusiast
📍 Bihar, India


