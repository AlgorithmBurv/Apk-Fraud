from flask import Flask, render_template, request, send_file, make_response

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from joblib import load

import requests


import base64
from io import BytesIO, StringIO

import matplotlib.pyplot as plt  
import matplotlib
matplotlib.use('Agg')
import seaborn as sns 

import os
import csv



# Load model
model = load('model_rfc_done.joblib')
history_data = []

# Menyimpan hasil prediksi global untuk download batch
batch_prediction_df = None

app = Flask(__name__)

# Page: Home
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global batch_prediction_df  # Agar bisa diakses pada route download
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            df = pd.read_csv(file)
            expected_columns = ['step', 'type', 'type2', 'amount', 'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            if not all(col in df.columns for col in expected_columns):
                return render_template('index.html', error_text="CSV tidak sesuai format. Harap pastikan kolom sesuai.")

            # Prediksi batch
            predictions = model.predict(df[expected_columns])
            df['isFraud'] = predictions

            # Simpan hasil prediksi untuk diunduh
            batch_prediction_df = df

            # Hanya tampilkan 10 baris pertama di tabel
            result_data = df.head(20000).to_dict(orient='records')
            return render_template('index.html', result_table=result_data)

        # Input manual
        step = int(request.form['step'])
        type_ = int(request.form['type'])
        type2 = int(request.form['type2'])
        amount = float(request.form['amount'])
        oldbalanceOrig = float(request.form['oldbalanceOrig'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        input_data = np.array([[step, type_, type2, amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest, newbalanceDest]])
        prediction = model.predict(input_data)
        label_map = {0: 'Non-Fraud', 1: 'Fraud'}
        result = label_map[prediction[0]]


        # Menyimpan hasil prediksi manual ke list
        history_data.append({
            'Step': step,
            'Type': type_,
            'Type2': type2,
            'Amount': amount,
            'Old Balance Orig': oldbalanceOrig,
            'New Balance Orig': newbalanceOrig,
            'Old Balance Dest': oldbalanceDest,
            'New Balance Dest': newbalanceDest,
            'Prediction': result
        })


        return render_template('index.html', prediction_text=f'Hasil Prediksi: {result}')
# Route untuk download CSV batch
@app.route('/download_csv')
def download_csv():
    global batch_prediction_df
    if batch_prediction_df is not None:
        csv_buffer = StringIO()
        batch_prediction_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        response = make_response(csv_buffer.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=batch_prediction.csv"
        response.headers["Content-Type"] = "text/csv"
        return response
    else:
        return "No batch prediction data available to download", 400

# Route untuk download Excel batch
@app.route('/download_excel')
def download_excel():
    global batch_prediction_df
    if batch_prediction_df is not None:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            batch_prediction_df.to_excel(writer, index=False, sheet_name='Predictions')
        excel_buffer.seek(0)
        response = make_response(excel_buffer.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=batch_prediction.xlsx"
        response.headers["Content-Type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        return response
    else:
        return "No batch prediction data available to download", 400

# Page: Training
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Baca CSV dan tampilkan preview
            df = pd.read_csv(file)
            X = df.drop(columns=['isFraud'], axis=1)
            y = df.isFraud

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Pilih classifier
            classifier_option = request.form['classifier']
            if classifier_option == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            else:
                model = xgb.XGBClassifier(random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluasi Model
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

            # K-Fold Cross Validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

            evaluation_results = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "Specificity": specificity,
                "F1 Score": f1,
                "AUC": auc,
                "Cross-Validation Mean": cv_scores.mean()
            }

            # Visualisasi confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')


            # Konversi plot confusion matrix ke base64
            buf1 = BytesIO()
            plt.savefig(buf1, format='png')
            buf1.seek(0)
            image_base64_confusion = base64.b64encode(buf1.getvalue()).decode('utf-8')
            buf1.close()
            plt.close()  # Tutup plot agar tidak bentrok

            # Membuat visualisasi distribusi transaksi berdasarkan type dan isFraud
            plt.figure(figsize=(10, 5))
            ax = sns.countplot(data=df, x="type", hue="isFraud", palette=['green', 'red'], order=[0, 1, 2, 3, 4])
            ax.set_xticklabels(['Cash-In', 'Cash-Out', 'Debit', 'Payment', 'Transfer'])  # Mengatur label sumbu x




            # Menambahkan label jumlah transaksi di atas batang
            for p in ax.patches:
                if p.get_height() > 0:  # Menampilkan hanya untuk batang yang jumlahnya signifikan
                    ax.annotate(
                        '{:.0f}'.format(p.get_height()),
                        (p.get_x() + p.get_width() / 2., p.get_height() + 25),
                        ha='center', va='bottom'
                    )

            # Konversi plot distribusi transaksi ke base64
            buf2 = BytesIO()
            plt.savefig(buf2, format='png')
            buf2.seek(0)
            image_base64_barplot = base64.b64encode(buf2.getvalue()).decode('utf-8')
            buf2.close()
            plt.close()  # Tutup plot agar tidak bentrok


            return render_template(
                'upload.html',
                evaluation_results=evaluation_results,
                preview=df.head(20).to_html(classes='table table-striped'),
                confusion_matrix_plot= image_base64_confusion,
                fraud_distribution_plot=image_base64_barplot
            )
    return render_template('upload.html')

#  Page: History
@app.route('/history')
def history():
    # if not history_data:
    #     return "Belum ada riwayat prediksi."

    return render_template('history.html', history_data=history_data)

# Download history as CSV
@app.route('/download_history_csv')
def download_history_csv():
    if not history_data:
        return "No history available for download", 400

    # Konversi list of dict ke DataFrame
    df = pd.DataFrame(history_data)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Mengirimkan file sebagai response
    response = Response(csv_buffer.getvalue(), mimetype='text/csv')
    response.headers["Content-Disposition"] = "attachment; filename=transaction_history.csv"
    return response

# Download history as Excel
@app.route('/download_history_excel')
def download_history_excel():
    if not history_data:
        return "No history available for download", 400

    # Konversi list of dict ke DataFrame
    df = pd.DataFrame(history_data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='History')
    output.seek(0)

    # Mengirimkan file sebagai response
    return send_file(output,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     download_name='transaction_history.xlsx',
                     as_attachment=True)



# Page: Currency
API_URL = "https://api.exchangerate-api.com/v4/latest/USD"

@app.route("/currency", methods=["GET", "POST"])
def currency():
    response = requests.get(API_URL)
    data = response.json()
    exchange_rates = {
        "IDR": data["rates"]["IDR"],
        "SGD": data["rates"]["SGD"],
        "TZS": data["rates"]["TZS"],
        "CNY": data["rates"]["CNY"],
        "EUR": data["rates"]["EUR"],
        "INR": data["rates"]["INR"]
    }
    
    converted_amount = None
    exchange_rate = None
    if request.method == "POST":
        amount = float(request.form["amount"])
        currency = request.form["currency"]
        exchange_rate = data["rates"][currency]
        converted_amount = amount * exchange_rate

    return render_template("currency.html", exchange_rates=exchange_rates, converted_amount=converted_amount, exchange_rate=exchange_rate)


if __name__ == '__main__':
    app.run(debug=True)
