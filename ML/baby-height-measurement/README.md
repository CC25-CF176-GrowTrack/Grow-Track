---
---
title: Baby Height Measurement
emoji: 🐨
colorFrom: blue
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference



# 📏 Estimasi Tinggi Badan Bayi dari Gambar

Aplikasi ini menggunakan model YOLOv8 untuk mendeteksi koin dan pose bayi dari gambar, kemudian menghitung estimasi tinggi badan bayi berdasarkan ukuran koin yang terdeteksi sebagai referensi skala. API ini dibangun dengan FastAPI dan dideploy di Hugging Face Spaces.

---

## 🚀 Endpoint API

### `POST /predict-babyheight`

Mengestimasi tinggi badan bayi berdasarkan gambar yang diunggah.

- **Method**: `POST`
- **URL**: [`https://desssti006-baby-height-api.hf.space/predict-babyheight`](https://desssti006-baby-height-api.hf.space/predict-babyheight)
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `image`: file gambar bayi (JPEG/PNG) yang berisi bayi dan koin sebagai referensi ukuran.

#### Contoh Penggunaan (Postman):
- **Method**: `POST`
- **URL**: `https://desssti006-baby-height-api.hf.space/predict-babyheight`
- **Body (form-data)**:
  | Key   | Type | Value        |
  |-------|------|--------------|
  | image | File | `baby_1.jpeg` |

#### Contoh Respons:
```json
{
  "status": "success",
  "predicted_height_cm": 31.6,
  "annotated_image_url": "/result-image/539e81c8-e0bc-4cd8-a2a9-663bb0f88bac_out.jpg"
}
```

---

### `GET /result-image/{filename}`

Mengambil gambar hasil anotasi dengan pose keypoints dan koin yang terdeteksi.

- **URL Contoh**: `https://desssti006-baby-height-api.hf.space/result-image/539e81c8-e0bc-4cd8-a2a9-663bb0f88bac_out.jpg`
- **Response**: Gambar JPEG hasil anotasi atau `404` jika gambar tidak ditemukan.

---

## 🧠 Model yang Digunakan

- 🟡 `coin/yolo11s.pt` — untuk mendeteksi koin di gambar (sebagai acuan skala ukuran nyata).
- 🔴 `keypoints/yolo11s-pose.pt` — untuk mendeteksi pose bayi (keypoints seperti hidung, bahu, pergelangan kaki).

---

## ⚙️ Proses Pengukuran

1. Deteksi koin → hitung skala px ke cm.
2. Deteksi pose bayi (hidung hingga pergelangan kaki).
3. Hitung panjang bayi dalam pixel → dikalikan skala → hasil dalam **centimeter**.
4. Gambar hasil anotasi disimpan dan URL-nya dikembalikan.

---

## 📂 Struktur Proyek

```
.
├── api.py                    # FastAPI app utama
├── main.py                   # Proses pengukuran utama
├── coin/yolo11s.pt           # Model deteksi koin
├── keypoints/yolo11s-pose.pt # Model deteksi keypoints pose bayi
├── /tmp/uploads/             # Direktori gambar input dan output
├── requirements.txt
└── README.md
```

---

## 📦 Cara Menjalankan Secara Lokal

```bash
git clone https://huggingface.co/spaces/desssti006/baby-height-api
cd baby-height-api
pip install -r requirements.txt
uvicorn api:app --reload
```

Kemudian buka Postman dan kirim gambar seperti pada contoh di atas.

---

## 📸 Contoh Gambar Hasil

![Contoh Output](images/sample_output.png)

---

## 🧑‍💻 Kontributor

- `@desssti006` — Developer utama dan deployer Hugging Face Space

---

## 📄 Lisensi

MIT License – bebas digunakan, dimodifikasi, dan disebarluaskan dengan menyertakan atribusi.
