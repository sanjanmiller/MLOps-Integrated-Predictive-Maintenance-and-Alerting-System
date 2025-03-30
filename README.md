# 🚀 MLOps-Integrated Predictive Maintenance and Alerting System  

This project implements an **MLOps pipeline** for **predictive maintenance in vehicle engines**. It integrates **anomaly detection using One-Class SVM**, **SHAP analysis for explainability**, and an **automated email alert system** to notify vehicle owners about potential engine issues. **MLflow** is used for experiment tracking and logging key metrics.  

## 📌 Features  
✔️ **Anomaly Detection**: Identifies abnormal engine conditions using **One-Class SVM**.  
✔️ **SHAP Explainability**: Provides insights into which sensor readings contribute most to anomalies.  
✔️ **Automated Email Alerts**: Notifies customers via email if anomalies persist.  
✔️ **MLflow Integration**: Tracks parameters, metrics, and SHAP importance scores for MLOps compliance.  

---

## 🖥️ Demo Output  
### **1️⃣ MLflow Tracking of Anomaly Metrics**  
<img src="https://raw.githubusercontent.com/sanjanmiller/MLOps-Integrated-Predictive-Maintenance-and-Alerting-System/refs/heads/main/mlflow.JPG" width="600">  

### **2️⃣ Sample Anomaly Email Alert**  
<img src="https://raw.githubusercontent.com/sanjanmiller/MLOps-Integrated-Predictive-Maintenance-and-Alerting-System/refs/heads/main/email_alert.JPG" width="600">  

---

## 🔧 **Installation & Setup**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your_username/MLOps-Integrated-Predictive-Maintenance.git  
cd MLOps-Integrated-Predictive-Maintenance

### **2️⃣ Install Dependencies**  
pip install -r requirements.txt

### **3️⃣ Start MLflow UI (to track experiment logs)**
mlflow ui
**📌 Navigate to http://127.0.0.1:5000 in your browser.**

### **4️⃣ Run the Main Script**
python email_alerts_with_MLOps.py  

