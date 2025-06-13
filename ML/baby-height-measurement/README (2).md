
# 🍼 Baby Height Estimation API

Proyek ini adalah sebuah REST API berbasis FastAPI yang berfungsi untuk memprediksi panjang/tinggi bayi dari sebuah foto menggunakan model YOLOv8 dan keypoint detection.

## 📦 Fitur

- Deteksi koin sebagai acuan skala menggunakan YOLOv8
- Deteksi pose bayi (keypoints) dengan YOLOv11 pose
- Hitung panjang bayi dalam satuan sentimeter (cm)
- API siap konsumsi oleh tim frontend/web

---

## 🚀 Cara Menjalankan API

### 1. Clone repository ini

```bash
git clone https://github.com/username/baby-height-api.git
cd baby-height-api
```

### 2. Buat dan aktifkan environment (opsional tapi disarankan)

```bash
conda create -n babyheight python=3.10
conda activate babyheight
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Pastikan kamu sudah install `torch`, `ultralytics`, `opencv-python`, `fastapi`, dan `uvicorn`.

### 4. Jalankan server FastAPI

```bash
uvicorn api:app --reload
```

Server akan berjalan di:  
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🧪 Endpoint API

### `POST /predict-babyheight`

**Deskripsi**: Prediksi panjang bayi dari foto.  
**Request**: Form-data, dengan field:
- `image`: file gambar (`.jpg`, `.jpeg`, `.png`)

**Contoh request pakai `curl`:**

```bash
curl -X POST http://127.0.0.1:8000/predict-babyheight \
  -F "image=@baby_1.jpeg"
```

**Response JSON:**

```json
{
  "status": "success",
  "predicted_height_cm": 63.72
}
```

---

## 📁 Struktur Folder

```
.
├── api.py                  # Endpoint FastAPI
├── main.py                 # Pipeline utama (fungsi measure_all)
├── yolo.py (optional)
├── coin/                   # Model YOLOv8 untuk deteksi koin
│   └── yolo11s.pt
├── keypoints/              # Model YOLOv8 pose untuk deteksi keypoint bayi
│   └── yolo11s-pose.pt
├── uploads/                # Temp folder untuk simpan gambar sementara
├── output.jpg              # Gambar hasil anotasi (opsional)
├── requirements.txt
└── README.md
```

---

## ✅ Catatan

- Model YOLO harus sudah ditrain dan file `.pt` diletakkan di folder `coin/` dan `keypoints/`
- Jika tidak ada koin terdeteksi, sistem tidak bisa mengukur skala → hasil akan gagal
- Output akhir disimpan juga sebagai `output.jpg` untuk keperluan debug/visualisasi

---