# ✈️ Flight Analytics Dashboard

Interactive **Flight Analytics Dashboard** built with **Python, Streamlit, Pandas, Matplotlib, Seaborn, and Scikit-learn**.

Project ini digunakan untuk menganalisis performa penerbangan, mengidentifikasi pola keterlambatan dan pembatalan penerbangan, serta melakukan prediksi keterlambatan menggunakan algoritma **Random Forest Classifier**.

---

## 📊 Dashboard Preview

### Flight Analytics Dashboard

![Flight Analytics Dashboard](screenshots/dashboard-overview.png)

### Machine Learning

![Machine Learning Dashboard](screenshots/machine-learning.png)

> **Note:** Screenshot dapat diganti dengan hasil screenshot dashboard kamu sendiri.

---

## 🚀 Features

Dashboard ini memiliki beberapa fitur utama:

### 📊 1. Flight Performance Overview

Menampilkan informasi performa penerbangan secara interaktif, termasuk:

* Total flights
* Delayed flights
* Cancelled flights
* Average departure delay
* Monthly delay & cancellation trend
* Delay causes
* Airline performance

Dashboard juga menyediakan filter berdasarkan:

* ✈️ Airline
* 📅 Flight Date

---

### 🤖 2. Machine Learning

Dashboard dilengkapi model machine learning untuk memprediksi apakah sebuah penerbangan akan mengalami keterlambatan lebih dari **15 menit**.

Model yang digunakan:

**Random Forest Classifier**

Features yang digunakan:

* `AIRLINE`
* `ORIGIN`
* `DEST`
* `DAY_OF_WEEK`
* `MONTH`

Target:

```text
DEP_DELAY > 15 minutes
```

Model menggunakan pipeline preprocessing yang terdiri dari:

* `OneHotEncoder` untuk fitur kategorikal
* `SimpleImputer` untuk fitur numerik
* `RandomForestClassifier` sebagai model klasifikasi

Evaluasi model meliputi:

* Accuracy
* Confusion Matrix
* Prediction Distribution
* Feature Importance
* Classification Report

---

### 📄 3. Flight Dataset

Tab Data menampilkan dataset penerbangan berdasarkan filter yang dipilih.

Data ditampilkan menggunakan interactive dataframe dari Streamlit sehingga pengguna dapat melakukan eksplorasi data secara langsung.

---

## 🛠️ Tech Stack

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming language      |
| Streamlit    | Interactive web dashboard |
| Pandas       | Data manipulation         |
| NumPy        | Numerical computation     |
| Matplotlib   | Data visualization        |
| Seaborn      | Statistical visualization |
| Scikit-learn | Machine learning          |

---

## 📁 Project Structure

Struktur project yang direkomendasikan:

```text
flight-analytics/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── flights_sample_10000.csv
│
└── screenshots/
    ├── dashboard-overview.png
    └── machine-learning.png
```

---

## 💻 Installation

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME/flight-analytics.git
```

Masuk ke folder project:

```bash
cd flight-analytics
```

---

### 2. Create Virtual Environment

Windows:

```bash
python -m venv venv
```

Aktifkan virtual environment:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

Install semua library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

Jika belum memiliki `requirements.txt`, buat file tersebut dengan isi:

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Kemudian jalankan:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Jalankan Streamlit:

```bash
streamlit run app.py
```

Setelah itu buka URL yang diberikan oleh Streamlit, biasanya:

```text
http://localhost:8501
```

---

## 📂 Dataset

Project ini menggunakan dataset penerbangan dalam format CSV.

Pastikan dataset berada di:

```text
data/flights_sample_10000.csv
```

Kemudian ubah bagian `LOAD DATA` pada `app.py` menjadi:

```python
@st.cache_data
def load_data():
    file_path = "data/flights_sample_10000.csv"

    df = pd.read_csv(file_path)
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

    return df
```

### Dataset Columns

Beberapa kolom utama yang digunakan dalam project:

| Column                    | Description                    |
| ------------------------- | ------------------------------ |
| `FL_DATE`                 | Flight date                    |
| `AIRLINE`                 | Airline code                   |
| `ORIGIN`                  | Origin airport                 |
| `DEST`                    | Destination airport            |
| `DEP_DELAY`               | Departure delay                |
| `CANCELLED`               | Cancellation status            |
| `DELAY_DUE_CARRIER`       | Carrier-related delay          |
| `DELAY_DUE_WEATHER`       | Weather-related delay          |
| `DELAY_DUE_NAS`           | National Airspace System delay |
| `DELAY_DUE_SECURITY`      | Security-related delay         |
| `DELAY_DUE_LATE_AIRCRAFT` | Late aircraft delay            |

---

## 🧠 Machine Learning Workflow

Proses machine learning dalam dashboard:

```text
Flight Dataset
      │
      ▼
Data Filtering
      │
      ▼
Feature Engineering
      │
      ├── DAY_OF_WEEK
      └── MONTH
      │
      ▼
Train / Test Split
      │
      ▼
Preprocessing
      │
      ├── OneHotEncoder
      └── SimpleImputer
      │
      ▼
Random Forest Classifier
      │
      ▼
Prediction
      │
      ▼
Model Evaluation
      │
      ├── Accuracy
      ├── Confusion Matrix
      ├── Feature Importance
      └── Classification Report
```

---

## 📈 Model Evaluation

Dataset dibagi menjadi:

```text
70% Training Data
30% Testing Data
```

Pembagian dataset menggunakan:

```python
train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

Random Forest menggunakan:

```python
RandomForestClassifier(
    random_state=42,
    n_estimators=100
)
```

---

## 📸 Adding Screenshots

Untuk menampilkan screenshot dashboard di README GitHub, buat folder:

```text
screenshots/
```

Contoh:

```text
screenshots/
├── dashboard-overview.png
└── machine-learning.png
```

Kemudian gunakan Markdown berikut di `README.md`:

```markdown
## 📊 Dashboard Preview

### Flight Analytics Dashboard

![Flight Analytics Dashboard](screenshots/dashboard-overview.png)

### Machine Learning

![Machine Learning Dashboard](screenshots/machine-learning.png)
```

GitHub akan otomatis menampilkan gambar tersebut ketika file sudah di-push ke repository.

---

## 📤 Upload Screenshot to GitHub

### Cara 1 — Menggunakan Git

Setelah mengambil screenshot dan menyimpannya di folder `screenshots`, jalankan:

```bash
git add screenshots/
```

Kemudian commit:

```bash
git commit -m "Add dashboard screenshots"
```

Push ke GitHub:

```bash
git push origin main
```

Setelah berhasil, struktur repository akan menjadi:

```text
flight-analytics/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── flights_sample_10000.csv
│
└── screenshots/
    ├── dashboard-overview.png
    └── machine-learning.png
```

---

### Cara 2 — Upload melalui GitHub

1. Buka repository GitHub.
2. Klik **Add file**.
3. Pilih **Upload files**.
4. Upload screenshot.
5. Masukkan file ke folder `screenshots`.
6. Klik **Commit changes**.
7. Pastikan nama file sesuai dengan yang digunakan di README.

Contoh:

```text
screenshots/dashboard-overview.png
```

---

## 🎯 Dashboard KPI

Dashboard menampilkan empat KPI utama:

```text
┌────────────────┬────────────────┬────────────────┬────────────────┐
│ TOTAL FLIGHTS  │ DELAYED        │ CANCELLED      │ AVG. DELAY     │
│                │                │                │                │
│ Total flights  │ >15 minutes    │ Cancellation   │ Avg departure  │
│                │                │ rate           │ delay          │
└────────────────┴────────────────┴────────────────┴────────────────┘
```

---

## 📊 Visualizations

Dashboard menyediakan beberapa visualisasi:

### Monthly Flight Trend

Menampilkan tren:

* Delayed flights
* Cancelled flights

berdasarkan bulan.

### Delay Causes

Menganalisis total menit keterlambatan berdasarkan penyebab:

* Carrier
* Weather
* NAS
* Security
* Late Aircraft

### Airline Performance

Membandingkan rata-rata departure delay antar maskapai.

### Confusion Matrix

Menampilkan performa prediksi model:

```text
             Predicted
             On Time   Delayed
Actual
On Time
Delayed
```

### Feature Importance

Menampilkan 15 fitur yang paling berpengaruh terhadap prediksi keterlambatan penerbangan.

---

## ⚠️ Important Note

Model machine learning pada project ini digunakan sebagai bagian dari analisis dan demonstrasi data science.

Hasil prediksi tidak dimaksudkan sebagai sistem operasional penerbangan atau keputusan real-world tanpa validasi lebih lanjut.

Performa model dapat berubah tergantung:

* Dataset
* Data preprocessing
* Feature engineering
* Train/test split
* Model parameters

---

## 🔮 Future Improvements

Beberapa pengembangan yang dapat dilakukan:

* [ ] Menambahkan interactive Plotly charts
* [ ] Menambahkan airport analysis
* [ ] Menambahkan route performance analysis
* [ ] Menambahkan model comparison
* [ ] Menambahkan hyperparameter tuning
* [ ] Menambahkan prediction input form
* [ ] Menyimpan trained model menggunakan `joblib`
* [ ] Deploy ke Streamlit Community Cloud
* [ ] Menambahkan automated data preprocessing
* [ ] Menambahkan model performance monitoring

---

## 👨‍💻 Author

**Your Name**

Data Visualization & Machine Learning Project

---

## 📄 License

This project is intended for educational and portfolio purposes.
