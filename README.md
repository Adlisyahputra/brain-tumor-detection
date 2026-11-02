# Klasifikasi MRI Otak dengan VGG16

Project belajar deep learning: mengklasifikasikan gambar MRI otak ke dalam empat kelas (glioma, meningioma, pituitary, tanpa tumor) memakai transfer learning VGG16, lalu membungkus model dalam aplikasi web Flask.

> **Ini demo untuk belajar, bukan alat diagnosis.** Model tidak boleh dipakai untuk keputusan medis.

## Isi repo

| Path | Fungsi |
|---|---|
| `notebooks/brain_tumor_detection.ipynb` | Notebook training dan evaluasi (dijalankan di Google Colab) |
| `models/brain_tumor_vgg16_small.keras` | Model terlatih (tanpa status optimizer, sekitar 60 MB) |
| `app.py` | Aplikasi Flask: halaman web dan endpoint `/api/predict` |
| `templates/index.html` | Halaman unggah gambar dan tampilan hasil |

## Sumber dan kredit

Project ini dikerjakan sambil belajar, dengan mengikuti tutorial dan kode terbuka berikut:

- Video tutorial: https://www.youtube.com/watch?v=MTQQnziWGQY&t=2465s
- Repo asal kode: https://github.com/611noorsaeed/Brain-Tumor-Detection-Using-Deep-Learning-MRI-Images-Detection-Using-Computer-Vision/tree/main


Alur dan sebagian besar kode di notebook training mengikuti sumber di atas. Yang saya kerjakan atau ubah sendiri:

- Menjalankan ulang training dan evaluasi di Google Colab, lalu menganalisis kesalahan model (glioma yang terdeteksi sebagai normal, glioma dan meningioma yang tertukar).
- Menyesuaikan urutan label supaya cocok dengan model saya (urutan di kode asal berbeda dan akan menghasilkan label yang salah).
- Memperkecil model dari sekitar 122 MB menjadi sekitar 60 MB dengan membuang status optimizer.
- Membuat aplikasi web Flask lokal dengan endpoint JSON dan tampilan probabilitas tiap kelas, dibuat dengan bantuan asisten AI (Claude).
- Menulis catatan keterbatasan dan ide pengembangan di README ini.

## Dataset

Brain Tumor MRI Dataset dari Kaggle, berisi gambar MRI otak dalam 4 kelas. Dataset tidak disertakan di repo, silakan unduh dari Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Dataset ini gabungan dari beberapa sumber publik, jadi baca deskripsinya di Kaggle untuk asal dan keterbatasannya.

## Model dan training

- Basis: VGG16 dengan bobot ImageNet, input 128x128 RGB.
- Layer VGG16 dibekukan, kecuali tiga layer konvolusi terakhir yang ikut dilatih.
- Kepala klasifikasi: Flatten, Dropout 0,3, Dense 128 (ReLU), Dropout 0,2, Dense 4 (softmax).
- Optimizer Adam (learning rate 0,0001), batch 20, 5 epoch.
- Augmentasi: kecerahan dan kontras acak (0,8 sampai 1,2), piksel dinormalisasi ke 0-1.

## Hasil

Dievaluasi pada 1600 gambar data uji (400 per kelas). Akurasi sekitar **91%**.

| Kelas | Recall | AUC |
|---|---|---|
| Meningioma | 0,87 | 0,98 |
| Pituitary | 0,99 | 1,00 |
| Glioma | 0,80 | 0,94 |
| Tanpa tumor | 1,00 | 1,00 |

Temuan dari confusion matrix:

- Kesalahan terbanyak adalah glioma dan meningioma yang saling tertukar.
- Sekitar 30 gambar bertumor (28 glioma, 2 meningioma) diprediksi "tanpa tumor". Pada konteks medis, ini jenis kesalahan yang paling berbahaya.
- Hanya 1 gambar normal yang diprediksi sebagai tumor.

## Keterbatasan

- Tidak ada validation set terpisah, sehingga overfitting tidak dipantau selama training.
- Gambar data uji ikut diaugmentasi acak saat evaluasi, jadi angka bisa sedikit berubah antar eksekusi.
- Dataset berisi potongan 2D berformat JPG dari beberapa sumber, dan satu pasien bisa punya banyak gambar. Hasil pada data dari rumah sakit atau mesin lain kemungkinan lebih rendah dan belum diuji.
- Model selalu memilih salah satu dari empat kelas, bahkan untuk gambar yang bukan MRI otak.
- Confidence tinggi tidak menjamin prediksi benar.

## Cara menjalankan (lokal)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Lalu buka `http://127.0.0.1:5000`. Untuk memakai endpoint JSON:

```bat
curl -F "file=@gambar.jpg" http://127.0.0.1:5000/api/predict
```

Samakan versi TensorFlow dengan yang dipakai saat model disimpan kalau `load_model` error.

## Ide pengembangan

- Tambah validation set dan evaluasi tanpa augmentasi pada data uji.
- Latih dengan ukuran gambar 224 dan lebih banyak epoch.
- Pisahkan data per pasien untuk menghindari kebocoran data.
- Uji pada dataset MRI dari sumber lain.
