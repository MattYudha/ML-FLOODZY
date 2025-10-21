Laporan Analisis Feature Importance Model Floodzy

1. Pendahuluan

Memahami "kotak hitam" atau black box dari model machine learning adalah salah satu tantangan utama dalam Explainable AI (XAI). Analisis feature importance bertujuan untuk mengidentifikasi dan mengkuantifikasi kontribusi setiap fitur input terhadap hasil prediksi model. Dalam konteks Floodzy, ini berarti kita ingin mengetahui faktor-faktor apa saja (misalnya curah hujan, ketinggian air, dll.) yang paling dominan digunakan oleh model untuk memprediksi kejadian banjir.

Analisis ini menggunakan dua metode:

XGBoost Built-in Importance: Metode cepat untuk mendapatkan peringkat fitur berdasarkan kontribusinya dalam proses training (gain).

SHAP (SHapley Additive exPlanations): Pendekatan berbasis teori permainan yang memberikan penjelasan yang konsisten dan akurat baik secara global maupun untuk prediksi individual.

2. Metodologi

Model: xgb_floodzy_national_v2_cuda.json.

Dataset Referensi: data/processed/floodzy_new_train.csv digunakan sebagai data dasar untuk menghitung nilai kepentingan fitur.

Visualisasi:

Bar Chart (XGBoost): Menampilkan 10 fitur teratas berdasarkan nilai gain. Semakin tinggi gain, semakin besar kontribusi fitur tersebut dalam mengurangi loss (kesalahan) model.

Summary Plot (SHAP): Memvisualisasikan distribusi SHAP value untuk setiap fitur. Plot ini menunjukkan:

Peringkat Fitur: Fitur di bagian atas adalah yang paling penting.

Besaran Pengaruh: Jarak titik dari garis tengah (sumbu x=0) menunjukkan seberapa besar pengaruhnya pada prediksi.

Arah Pengaruh: Warna titik (merah untuk nilai fitur tinggi, biru untuk rendah) menunjukkan apakah nilai fitur tersebut menaikkan atau menurunkan probabilitas banjir.

3. Hasil

(Di bagian ini, Anda akan menyisipkan gambar-gambar yang dihasilkan oleh skrip)

3.1. XGBoost Feature Importance (Top 10)

Naratif:
Grafik di atas menunjukkan 10 fitur teratas yang memberikan kontribusi terbesar selama proses training. Fitur yang berada di puncak memiliki dampak paling signifikan dalam meningkatkan akurasi model secara keseluruhan.

3.2. SHAP Summary Plot

Naratif:
SHAP Summary Plot memberikan gambaran yang lebih detail. Sebagai contoh, jika kita melihat fitur rain_mm:

Titik-titik yang tersebar di sumbu X menunjukkan besaran pengaruh setiap data point.

Warna titik-titik tersebut (merah untuk curah hujan tinggi, biru untuk rendah) menunjukkan bahwa nilai rain_mm yang tinggi (merah) cenderung memiliki SHAP value positif, yang berarti mendorong prediksi ke arah "banjir". Sebaliknya, nilai rain_mm yang rendah (biru) memiliki SHAP value negatif, mendorong prediksi ke arah "tidak banjir".

4. Kesimpulan

(Skrip akan mencetak kesimpulan utama secara otomatis di terminal, yang bisa Anda salin dan kembangkan di sini)

Analisis dari kedua metode secara konsisten menunjukkan bahwa fitur [Nama Fitur Teratas] adalah prediktor paling kuat dalam model. Hal ini sejalan dengan pemahaman domain bahwa [jelaskan mengapa fitur tersebut masuk akal secara ilmiah/logis]. Pemahaman ini krusial untuk validasi model, komunikasi dengan pemangku kepentingan, dan pengembangan fitur di masa depan.