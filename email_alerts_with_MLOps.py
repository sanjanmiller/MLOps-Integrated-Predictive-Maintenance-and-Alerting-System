import pandas as pd
import numpy as np
import joblib
import shap
import smtplib
import ssl
import mlflow
import mlflow.sklearn
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Start MLflow Tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Engine_Anomaly_Alerts")

# Load Model & Scaler
model = joblib.load(r"C:\Users\HP\Downloads\MLOps-Integrated Predictive Maintenance and Alerting System\one_class_svm.pkl")
scaler = joblib.load(r"C:\Users\HP\Downloads\MLOps-Integrated Predictive Maintenance and Alerting System\scaler.pkl")

# Load Reference Dataset to Compute Normal Ranges
reference_df = pd.read_csv(r"C:\Users\HP\Downloads\MLOps-Integrated Predictive Maintenance and Alerting System\engine_data.csv")

# Define Feature Columns
features = ["Engine rpm", "Lub oil pressure", "Fuel pressure", 
            "Coolant pressure", "lub oil temp", "Coolant temp"]

# Compute Normal Ranges from "Engine Condition" = 0
normal_data = reference_df[reference_df["Engine Condition"] == 0]
normal_ranges = {feature: (round(normal_data[feature].mean() - 2 * normal_data[feature].std(), 2),
                           round(normal_data[feature].mean() + 2 * normal_data[feature].std(), 2))
                 for feature in features}

print("\nCalculated Normal Ranges:")
print(normal_ranges)

# Load New Dataset (Live Data for Testing)
df = pd.read_csv(r"C:\Users\HP\Downloads\MLOps-Integrated Predictive Maintenance and Alerting System\engine_data_unlabeled_with_email.csv")

# Scale the new data
X_scaled = scaler.transform(df[features])

# Run Anomaly Detection
df["Anomaly_SVM"] = model.predict(X_scaled)
df["Anomaly_SVM"] = df["Anomaly_SVM"].map({1: 0, -1: 1})

# Count anomalies
anomaly_count = df["Anomaly_SVM"].sum()

# Run SHAP Analysis on Anomalies Only
top_feature, top_feature_importance = None, None
shap_importance_dict = {}

if anomaly_count > 0:
    anomalies = X_scaled[df["Anomaly_SVM"] == 1]
    background = shap.sample(X_scaled, min(50, len(X_scaled)))  # Reduce background data for speed
    explainer = shap.Explainer(model.decision_function, background)
    shap_values = explainer(anomalies)

    # Compute feature importance for all features
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    sorted_features = sorted(zip(features, feature_importance), key=lambda x: x[1], reverse=True)

    # Store feature importance in a dictionary
    shap_importance_dict = {feature: round(importance, 4) for feature, importance in sorted_features}

    if sorted_features:
        top_feature, top_feature_importance = sorted_features[0]

print("\nMost Important Feature Contributing to Anomaly:")
print(f"{top_feature}: {round(top_feature_importance, 4)}" if top_feature else "No feature identified.")

# Convert Technical SHAP Scores into Customer-Friendly Messages
impact_messages = {
    "Fuel pressure": "Unusual fluctuation detected in fuel pressure, which may indicate a fuel system issue.",
    "Coolant pressure": "Coolant system pressure is irregular, possibly due to leaks or overheating.",
    "Lub oil pressure": "Lubrication system pressure is outside the expected range, which may impact engine efficiency.",
    "lub oil temp": "Oil temperature is deviating from normal levels, which can lead to increased wear on components.",
    "Engine rpm": "Engine speed is fluctuating, possibly indicating irregular combustion or load variations.",
    "Coolant temp": "Cooling system temperature is abnormal, which could suggest overheating or a faulty thermostat."
}

# Start MLflow Run
with mlflow.start_run():
    mlflow.log_param("Anomaly_Threshold", 5)
    mlflow.log_metric("Total_Anomalies", anomaly_count)

    if top_feature:
        mlflow.log_param("Top_Feature", top_feature)
        mlflow.log_metric("SHAP_Importance", round(top_feature_importance, 4))

    # Log SHAP Importance for all features
    for feature, importance in shap_importance_dict.items():
        mlflow.log_metric(f"SHAP_{feature.replace(' ', '_')}", importance)

    email_count = 0  # Track emails sent

    # Trigger Email Alert Only if Anomalies Persist
    ANOMALY_THRESHOLD = 5

    if anomaly_count >= ANOMALY_THRESHOLD:
        sender_email = "enter your mail here"
        smtp_password = "your_email_password"

        # Loop through each anomaly and send email to the respective customer
        for index, row in df[df["Anomaly_SVM"] == 1].iterrows():
            receiver_email = row["Customer Email"]
            engine_id = row["Engine ID"]

            subject = f"🚨 Engine {engine_id} - Anomaly Alert!"

            # Get description for the most important feature
            key_factor_description = impact_messages.get(top_feature, "Anomaly detected, but specific details are unavailable.")

            # Detected Values vs. Normal Range Table
            detected_values_table = "".join(
                f"<tr><td>{feature}</td><td><b>{row[feature]}</b></td><td>{normal_ranges[feature][0]} - {normal_ranges[feature][1]}</td></tr>"
                for feature in features
            )

            email_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; color: #333;">
                    <h2 style="color: red;">🚨 Alert: Potential Issue with Your Engine</h2>
                    <p><b>Engine ID:</b> {engine_id}</p>

                    <h3>🔍 Key Factor Contributing to This Issue:</h3>
                    <p style="background: #f8d7da; padding: 10px; border-radius: 5px;">
                        ✔ {key_factor_description}
                    </p>

                    <h3>⚠ Detected Values vs. Normal Ranges:</h3>
                    <table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse;">
                        <tr style="background: #ddd;">
                            <th>Feature</th>
                            <th>Detected Value</th>
                            <th>Normal Range</th>
                        </tr>
                        {detected_values_table}
                    </table>

                    <h3>🔧 Recommended Actions:</h3>
                    <ul style="background: #d4edda; padding: 10px; border-radius: 5px;">
                        <li>✔ <b>Check the fuel and lubrication systems</b> – Ensure proper flow and no blockages.</li>
                        <li>✔ <b>Schedule a service check</b> – Contact an authorized service provider.</li>
                    </ul>

                    <p>📞 <b>Need Assistance?</b> Our support team is available 24/7. Reply to this email for expert guidance.</p>
                </body>
            </html>
            """

            # Send Email
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = receiver_email
            msg["Subject"] = subject
            msg.attach(MIMEText(email_body, "html"))

            try:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                    server.login(sender_email, smtp_password)
                    server.sendmail(sender_email, receiver_email, msg.as_string())
                email_count += 1
                print(f"Email Sent to {receiver_email} for Engine {engine_id}!")
            except Exception as e:
                print(f"Failed to send email to {receiver_email}: {e}")

    mlflow.log_metric("Emails_Sent", email_count)
