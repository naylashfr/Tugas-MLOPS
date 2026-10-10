# LAPORAN TUGAS MLOps
## Model Experimentation & Tracking dengan MLflow

**Mata Kuliah:** Machine Learning Operations (MLOps)

**Nama:** Nayla Shafira Roza

**NIM:** 123450017

**Topik:** Eksperimen CNN untuk Klasifikasi Fashion-MNIST dengan MLflow, Optuna, dan DVC

---

## 1. Pendahuluan

Eksperimen *machine learning* biasanya melibatkan beberapa variasi konfigurasi, parameter, dan proses evaluasi. Tanpa pencatatan yang terstruktur, hasil eksperimen akan sulit dibandingkan dan diulang kembali. Oleh karena itu, *experiment tracking* menjadi bagian penting dalam praktik MLOps karena membantu mencatat parameter, metrik, artefak, versi data, serta model yang dihasilkan.

Pada tugas ini dibangun alur eksperimen klasifikasi citra menggunakan *Convolutional Neural Network* (CNN) pada dataset Fashion-MNIST. MLflow digunakan untuk mencatat eksperimen dan mengelola model, Optuna digunakan untuk melakukan *hyperparameter tuning*, sedangkan DVC digunakan untuk membantu versioning snapshot dataset.

## 2. Tujuan

Tujuan tugas ini adalah:

1. Menyiapkan MLflow tracking dengan backend SQLite dan direktori artefak.
2. Membuat model baseline CNN dan mencatat parameter serta metrik pada setiap epoch.
3. Melakukan minimal 20 trial *hyperparameter tuning* menggunakan Optuna.
4. Menjalankan konfigurasi terpilih dengan tiga random seed dan melaporkan rata-rata serta standar deviasi.
5. Melakukan *ablation study* pada komponen augmentasi data dan dropout.
6. Mendaftarkan model ke MLflow Model Registry menggunakan alias `@champion`, kemudian memuat model untuk prediksi.
7. Membuat snapshot dataset yang diberi metadata DVC dan mencatat hash dataset dalam eksperimen.

## 3. Dataset dan Persiapan Data

### 3.1 Dataset

Dataset yang digunakan adalah Fashion-MNIST, yaitu dataset citra grayscale pakaian dan aksesori dengan ukuran citra 28 × 28 piksel serta 10 kelas. Dataset ini digunakan untuk eksperimen klasifikasi citra dengan CNN.

Agar proses eksperimen lebih ringan, tugas memakai subset data yang dipilih secara tetap. Pembagian data yang digunakan adalah:

| Bagian data | Jumlah sampel |
|---|---:|
| Training | 12.000 |
| Validation | 2.000 |
| Test | 2.000 |
| **Total subset** | **16.000** |

Data validation digunakan untuk membandingkan konfigurasi model. Data test digunakan pada evaluasi akhir setelah konfigurasi dipilih.

### 3.2 Versioning dan Hash Dataset

Snapshot subset disimpan sebagai `data/fashion_mnist_subset.npz`. File tersebut ditambahkan ke DVC sehingga menghasilkan metadata `.dvc`. Hash digunakan untuk membantu menelusuri dataset yang digunakan pada eksperimen.

| Informasi | Nilai |
|---|---|
| Hash konten dataset (SHA-256) | `dfd40e17974b94d009421d59c178de40afd9a0b27e0ba60072a3485a510000d9` |
| Hash file snapshot (SHA-256) | `3f3cbca389b3495638a768ee1688e9657018f6b15c1bdff10401f3a3739d20f5` |
| Metadata DVC | `data/fashion_mnist_subset.npz.dvc` |
| Ukuran training / validation / test | 12.000 / 2.000 / 2.000 |

Hash di atas merupakan nilai yang dicetak saat notebook dijalankan. Untuk versioning yang bisa direproduksi dari komputer lain, file `.dvc` perlu disimpan dalam repository Git. Remote storage DVC belum dikonfigurasi pada eksperimen ini.

## 4. Metode dan Rancangan Eksperimen

### 4.1 Arsitektur CNN

Model baseline menggunakan CNN sederhana. Model memiliki dua lapisan konvolusi yang diikuti aktivasi ReLU dan *max pooling*. Hasil ekstraksi fitur kemudian diratakan dan diteruskan ke lapisan *fully connected* untuk menghasilkan prediksi 10 kelas. Dropout digunakan sebagai salah satu komponen regularisasi yang diuji.

Model dilatih menggunakan optimizer Adam dan *cross-entropy loss*. Metrik yang diamati antara lain `train_loss`, `val_loss`, `val_accuracy`, dan `val_f1_macro`.

### 4.2 Experiment Tracking dengan MLflow

MLflow digunakan untuk mencatat parameter, metrik, tag, riwayat pelatihan, serta artefak model. Eksperimen disimpan dalam backend SQLite dengan folder artefak lokal. Informasi seed dan hash dataset ikut dicatat agar run dapat ditelusuri.

### 4.3 Hyperparameter Tuning

Optuna digunakan untuk melakukan 20 trial pencarian parameter. Parameter yang dicari meliputi:

- Learning rate (`lr`)
- Batch size
- Dropout
- Jumlah kanal konvolusi (`channels`)
- Weight decay

Metrik optimasi adalah *validation macro-F1*. Setiap trial dicatat sebagai run di MLflow dan konfigurasi dengan skor validation macro-F1 tertinggi dipilih untuk eksperimen lanjutan.

### 4.4 Reproducibility dengan Tiga Seed

Konfigurasi hasil tuning diuji menggunakan tiga seed, yaitu 17, 42, dan 101. Nilai validation macro-F1 dirangkum sebagai mean ± standar deviasi sampel untuk menunjukkan variasi hasil akibat proses stokastik.

### 4.5 Ablation Study

Ablation dilakukan untuk mengukur perubahan performa ketika komponen model diubah. Tiga skenario yang diuji adalah konfigurasi penuh, model tanpa augmentasi, dan model tanpa dropout. Perbandingan dilakukan dengan metrik validation macro-F1.

### 4.6 Model Registry dan Prediksi

Model final didaftarkan pada MLflow Model Registry dengan nama `fashion-mnist-cnn`, versi 1, dan alias `@champion`. Setelah registrasi, model dimuat menggunakan alias tersebut dan digunakan untuk memprediksi sampel dari test subset.

## 5. Hasil Eksperimen

### 5.1 Hasil Baseline CNN

Hasil pelatihan baseline untuk tiga epoch adalah:

| Epoch | Train loss | Validation loss | Validation macro-F1 |
|---:|---:|---:|---:|
| 1 | 1,179133 | 0,644334 | 0,737329 |
| 2 | 0,656629 | 0,538091 | 0,782270 |
| 3 | 0,563183 | 0,468596 | 0,824796 |

Train loss dan validation loss turun pada setiap epoch, sedangkan validation macro-F1 meningkat dari 0,737329 menjadi 0,824796. Pola ini menunjukkan bahwa baseline belajar dari data training dan performa pada validation set meningkat selama tiga epoch tersebut.

### 5.2 Hasil Hyperparameter Tuning

Sebanyak 20 trial Optuna selesai dijalankan. Trial terbaik adalah trial ke-16 (penomoran Optuna dimulai dari 0), dengan validation macro-F1 sebesar **0,850015**.

Parameter trial terbaik:

| Parameter | Nilai |
|---|---:|
| Learning rate | 0,0026983168 |
| Batch size | 64 |
| Dropout | 0,303709 |
| Channels | 16 |
| Weight decay | 0,00000151424 |
| Best validation macro-F1 | **0,850015** |

Skor terbaik dari tuning lebih tinggi daripada validation macro-F1 baseline pada epoch ketiga (0,824796). Perbandingan ini menjadi indikasi bahwa pencarian hyperparameter membantu menemukan konfigurasi yang lebih baik pada validation set. Namun, jumlah epoch pada trial tuning lebih sedikit daripada pelatihan baseline sehingga hasilnya perlu dipahami sebagai hasil seleksi konfigurasi, bukan perbandingan pelatihan yang sepenuhnya setara.

### 5.3 Hasil Reproducibility: Tiga Seed

Konfigurasi hasil tuning dijalankan dengan seed 17, 42, dan 101.

| Seed | Validation macro-F1 |
|---:|---:|
| 17 | 0,872842 |
| 42 | 0,865567 |
| 101 | 0,870123 |
| **Mean** | **0,869511** |
| **Standar deviasi** | **0,003676** |

Dengan demikian, validation macro-F1 untuk tiga seed adalah **0,8695 ± 0,0037**. Standar deviasi yang relatif kecil menunjukkan bahwa performa validasi pada tiga seed tersebut cukup konsisten, meskipun hasil ini hanya berdasarkan tiga pengulangan.

### 5.4 Hasil Ablation Study

| Eksperimen | Augmentasi | Dropout | Validation macro-F1 |
|---|---|---:|---:|
| Konfigurasi penuh | Ya | 0,303709 | 0,862268 |
| Tanpa augmentasi | Tidak | 0,303709 | 0,863515 |
| Tanpa dropout | Ya | 0,000000 | **0,873331** |

Pada tiga konfigurasi yang diuji, model tanpa dropout memperoleh validation macro-F1 tertinggi, yaitu 0,873331. Model tanpa augmentasi juga sedikit lebih tinggi daripada konfigurasi penuh. Dalam eksperimen ini, augmentasi dan dropout tidak meningkatkan validation macro-F1 dibandingkan konfigurasi yang diubah. Kesimpulan ini terbatas pada subset data, parameter, seed, dan durasi pelatihan yang digunakan.

### 5.5 Evaluasi Akhir pada Test Set

Model yang tercatat sebagai `final-candidate` dievaluasi pada test subset satu kali. Hasilnya:

| Metrik | Nilai |
|---|---:|
| Test loss | 0,391057 |
| Test accuracy | 0,861500 (86,15%) |
| Test macro-F1 | 0,864115 (86,41%) |

Confusion matrix menunjukkan bahwa banyak prediksi berada pada diagonal utama, yang menandakan prediksi benar untuk masing-masing kelas. Kesalahan klasifikasi masih terlihat pada beberapa kelas pakaian yang secara visual mirip.

**Catatan interpretasi:** model `final-candidate` yang dievaluasi dan didaftarkan menggunakan konfigurasi hasil tuning dengan dropout 0,303709 dan augmentasi aktif. Walaupun ablation tanpa dropout menghasilkan validation macro-F1 tertinggi, konfigurasi ablation tersebut tidak sama dengan konfigurasi `final-candidate` yang menghasilkan metrik test di atas. Oleh karena itu, hasil test di atas tidak boleh diklaim sebagai hasil model tanpa dropout.

### 5.6 Registrasi dan Prediksi Model

Model berhasil diregistrasikan dengan informasi berikut:

| Informasi | Nilai |
|---|---|
| Registered model | `fashion-mnist-cnn` |
| Versi model | 1 |
| Alias | `@champion` |
| Nama run sumber | `final-candidate` |
| Run ID | `f36b5ca504d846c8b0b03267542e1386` |

Notebook berhasil memuat model menggunakan alias `models:/fashion-mnist-cnn@champion` dan menjalankan prediksi pada sampel test. Keberhasilan ini menunjukkan bahwa model dapat diakses dari Model Registry tanpa perlu membuat ulang model pada tahap prediksi.

## 6. Pembahasan

Tugas ini memperlihatkan alur experiment tracking mulai dari baseline, pencarian hyperparameter, pengujian beberapa seed, ablation, evaluasi test, sampai registrasi dan prediksi model. MLflow menyimpan hasil eksperimen sehingga perbedaan parameter dan metrik dapat diperiksa kembali. Optuna membantu mengeksplorasi konfigurasi dengan cara yang terstruktur, sementara DVC menghasilkan metadata untuk snapshot dataset.

Baseline mencapai validation macro-F1 0,824796 pada epoch ketiga. Pencarian 20 trial menghasilkan validation macro-F1 terbaik sebesar 0,850015. Ketika konfigurasi terpilih dijalankan pada tiga seed, mean validation macro-F1 adalah 0,869511 dengan standar deviasi 0,003676. Pada ablation yang dilakukan, penghilangan dropout menghasilkan skor validasi tertinggi, tetapi model yang didaftarkan sebagai `@champion` masih menggunakan konfigurasi tuning dengan dropout. Perbedaan ini dicatat agar interpretasi metrik tidak mencampuradukkan hasil dari konfigurasi berbeda.

Pada test subset, model terdaftar memperoleh accuracy 86,15% dan macro-F1 86,41%. Hasil tersebut menunjukkan kemampuan klasifikasi yang cukup baik untuk eksperimen tugas, tetapi masih terdapat kelas-kelas yang tertukar. Hasil ini tidak dapat langsung digeneralisasi ke seluruh penggunaan klasifikasi citra di luar dataset Fashion-MNIST.

## 7. Keterbatasan

1. Eksperimen menggunakan subset tetap Fashion-MNIST, bukan seluruh data training dan test asli.
2. Setiap trial tuning dijalankan selama dua epoch demi efisiensi waktu komputasi; jumlah epoch ini relatif singkat untuk menyimpulkan performa maksimum model.
3. Pengujian reproducibility hanya memakai tiga seed.
4. Ablation tanpa dropout menghasilkan skor validasi tertinggi, tetapi model final yang diuji pada test set tidak dilatih ulang dengan dropout nol.
5. File `.dvc` dibuat di lingkungan Colab, tetapi remote storage DVC belum dikonfigurasi. Karena itu, file metadata `.dvc` perlu disertakan dalam repository dan remote DVC ditambahkan bila replikasi lintas mesin diperlukan.
6. Pada output audit L9 yang dibagikan, tag hash pada versi model registry masih terbaca `null`. Hash dataset tercatat pada tahap persiapan data, tetapi metadata registry perlu diperbarui dan diverifikasi agar lineage model terhubung langsung dengan hash tersebut.
7. URL MLflow UI yang tercetak di Colab adalah URL runtime. Screenshot perlu disimpan selama runtime masih aktif; URL tersebut belum tentu tersedia setelah sesi Colab berakhir.

## 8. Kesimpulan

Berhasil membangun alur eksperimen CNN Fashion-MNIST dengan MLflow, Optuna, dan DVC. Baseline mencapai validation macro-F1 0,824796 pada epoch ketiga, sedangkan tuning 20 trial menghasilkan skor validation macro-F1 terbaik sebesar 0,850015. Evaluasi tiga seed menghasilkan mean validation macro-F1 0,8695 dengan standar deviasi 0,0037. Dalam ablation, model tanpa dropout memperoleh validation macro-F1 tertinggi sebesar 0,873331. Model final yang terdaftar memperoleh test accuracy 86,15% dan test macro-F1 86,41%, serta berhasil dimuat melalui alias `@champion` untuk prediksi.