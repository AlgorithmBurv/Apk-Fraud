# ======================================================
# =          Import Libraries and Dependencies         =
# ======================================================
from flask import Flask, render_template, request, send_file, make_response, Response

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
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
import time

# ======================================================
# =                Model and Global Data               =
# ======================================================

model = load('model_rfc.joblib')

history_data = [] #page: history

batch_prediction_df = None #page: prediction

# ======================================================
# =           Flask Application Initialization         =
# ======================================================
app = Flask(__name__)

# ======================================================
# =                    Prediction Page                       =
# ======================================================
@app.route('/')
def home():
    return render_template('index.html')

# ======================================================
# =                   Predict Route                    =
# ======================================================
@app.route('/predict', methods=['POST'])
def predict():
    global batch_prediction_df  # Menggunakan variabel global agar dapat diakses di fungsi lain (download)
    if request.method == 'POST':
        # Cek apakah user mengupload file (untuk batch prediction)
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            df = pd.read_csv(file)
            # Pastikan CSV sesuai dengan kolom yang diharapkan
            expected_columns = ['step', 'type', 'type2', 'amount','nameOrig', 'oldbalanceOrig', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
            if not all(col in df.columns for col in expected_columns):
                # Jika kolom tidak sesuai, render 'index.html' dengan pesan error
                return render_template('index.html', error_text="CSV tidak sesuai format. Harap pastikan kolom sesuai.")

            # Melakukan prediksi secara batch
            predictions = model.predict(df[expected_columns])
            df['isFraud'] = predictions

            # Simpan DataFrame prediksi ke variabel global
            batch_prediction_df = df

            # Tampilkan 20.000 baris pertama sebagai contoh
            result_data = df.head(20000).to_dict(orient='records')
            return render_template('index.html', result_table=result_data)

        # Jika tidak ada file, berarti melakukan single prediction dari form
        step = int(request.form['step'])
        type_ = int(request.form['type'])
        type2 = int(request.form['type2'])
        amount = float(request.form['amount'])
        nameOrig = int(request.form['nameOrig'])
        oldbalanceOrig = float(request.form['oldbalanceOrig'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        nameDest = int(request.form['nameDest'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        # Bentuk input menjadi array 2D karena model memerlukan shape (n_samples, n_features)
        input_data = np.array([[step, type_, type2, amount, nameOrig, oldbalanceOrig, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest]])
        prediction = model.predict(input_data)
        # Mapping label numerik ke teks
        label_map = {0: 'Non-Fraud', 1: 'Fraud'}
        result = label_map[prediction[0]]

        # Simpan data input + hasil prediksi ke history_data untuk riwayat
        history_data.append({
            'Step': step,
            'Type': type_,
            'Type2': type2,
            'Amount': amount,
            'Name Orig': nameOrig,
            'Old Balance Orig': oldbalanceOrig,
            'New Balance Orig': newbalanceOrig,
            'Name Dest': nameDest,
            'Old Balance Dest': oldbalanceDest,
            'New Balance Dest': newbalanceDest,
            'Prediction': result
        })

        # Render 'index.html' sambil menampilkan hasil prediksi single
        return render_template('index.html', prediction_text=f'Prediction Results: {result}')

# ======================================================
# =           Download Routes (Batch Results)          =
# ======================================================
@app.route('/download_csv')
def download_csv():
    # Mengunduh hasil batch prediction dalam format CSV
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

@app.route('/download_excel')
def download_excel():
    # Mengunduh hasil batch prediction dalam format Excel
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

# ======================================================
# =              Training Page and Logic               =
# ======================================================
@app.route('/training', methods=['GET', 'POST'])
def training():

    # Form untuk upload file CSV dan melatih model baru.
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Baca CSV
            df = pd.read_csv(file)

            # X adalah fitur, y adalah label
            X = df.drop(columns=['isFraud'], axis=1)
            y = df.isFraud

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Pilih classifier berdasarkan pilihan user (Random Forest atau XGBoost)
            classifier_option = request.form['classifier']
            if classifier_option == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
            elif classifier_option == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            else:
                model = xgb.XGBClassifier(random_state=42)

            # mulai menghitung training
            start_time = time.time()

            # Latih model
            model.fit(X_train, y_train)

            # selesai menghitung training
            end_time = time.time()
            training_duration = end_time - start_time  # Durasi training dalam detik            

            # Prediksi model
            y_pred = model.predict(X_test)

            # Menghitung metrik evaluasi
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

            # K-Fold Cross Validation
            kf = KFold(n_splits=16, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

            # Hasil evaluasi dalam dictionary
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
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Non-Fraud', 'Fraud'],
                        yticklabels=['Non-Fraud', 'Fraud'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            # Konversi plot ke base64
            buf1 = BytesIO()
            plt.savefig(buf1, format='png')
            buf1.seek(0)
            image_base64_confusion = base64.b64encode(buf1.getvalue()).decode('utf-8')
            buf1.close()
            plt.close()

            # Membuat visualisasi distribusi transaksi berdasarkan type dan isFraud
            plt.figure(figsize=(10, 5))
            ax = sns.countplot(data=df, x="type", hue="isFraud",
                               palette=['green', 'red'], order=[0, 1, 2, 3, 4])
            ax.set_xticklabels(['Cash-In', 'Cash-Out', 'Debit', 'Payment', 'Transfer'])

            # Menambahkan label jumlah transaksi di atas setiap batang
            for p in ax.patches:
                if p.get_height() > 0:
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
            plt.close()

            # Render template training.html dan tampilkan hasil
            return render_template(
                'training.html',
                evaluation_results=evaluation_results,
                training_duration=round(training_duration, 3),
                preview=df.head(20).to_html(classes='table table-striped'),
                confusion_matrix_plot=image_base64_confusion,
                fraud_distribution_plot=image_base64_barplot
            )

    # Jika GET request, hanya render form training
    return render_template('training.html')

# ======================================================
# =                History Prediction Page             =
# ======================================================
@app.route('/history')
def history():
    # Tampilkan data riwayat prediksi single transaction
    return render_template('history.html', history_data=history_data)

# ======================================================
# =          Download Routes (History Results)         =
# ======================================================
@app.route('/download_history_csv')
def download_history_csv():
    # Download riwayat prediksi single transaction sebagai CSV
    if not history_data:
        return "No history available for download", 400

    df = pd.DataFrame(history_data)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    response = Response(csv_buffer.getvalue(), mimetype='text/csv')
    response.headers["Content-Disposition"] = "attachment; filename=transaction_history.csv"
    return response

@app.route('/download_history_excel')
def download_history_excel():
    # Download riwayat prediksi single transaction sebagai Excel
    if not history_data:
        return "No history available for download", 400

    df = pd.DataFrame(history_data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='History')
    output.seek(0)

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        download_name='transaction_history.xlsx',
        as_attachment=True
    )

# ======================================================
# =                 Currency Exchange Page             =
# ======================================================
API_URL = "https://api.exchangerate-api.com/v4/latest/TZS"
# Base currency: TZS. API akan memberikan rates TZS -> mata uang lain.

@app.route("/currency", methods=["GET", "POST"])
def currency():
    # Ambil data nilai tukar berbasis TZS dari API
    response = requests.get(API_URL)
    data = response.json()

    # Dictionary: mata uang yang ingin ditampilkan
    exchange_rates = {
        "USD": data["rates"]["USD"],
        "IDR": data["rates"]["IDR"],
        "SGD": data["rates"]["SGD"],
        "CNY": data["rates"]["CNY"],
        "EUR": data["rates"]["EUR"],
        "INR": data["rates"]["INR"]
    }

    converted_amount = None
    exchange_rate = None

    # Jika user melakukan POST (konversi mata uang)
    if request.method == "POST":
        amount = float(request.form["amount"])     # Jumlah dalam TZS
        target_currency = request.form["currency"] # Mata uang tujuan

        # Ambil nilai tukar TZS -> target_currency
        exchange_rate = data["rates"][target_currency]

        # Hitung hasil konversi
        converted_amount = amount * exchange_rate

    # Render template currency.html
    return render_template(
        "currency.html",
        exchange_rates=exchange_rates,
        converted_amount=converted_amount,
        exchange_rate=exchange_rate
    )

# ======================================================
# =                   Main Program                     =
# ======================================================
if __name__ == '__main__':
   
    app.run(debug=True)
