from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn
from xgboost import XGBClassifier

# Buat instance dulu
model = XGBClassifier()

# Load model
model.load_model("model/xgboost_model.json")

# Load menu data
menu_df = pd.read_csv("data/data_menu_mpasi.csv")
menu_df.rename(columns={"Kode Menu": "menu_id"}, inplace=True)
menu_df['menu_id'] = pd.factorize(menu_df['menu_id'])[0].astype(int)

# Definisikan fitur sesuai training
features = ['child_id', 'Umur (bulan)', 'Tinggi Badan (cm)', 'menu_id'] + [
    'calories_kcal', 'fats_g', 'sod_mg', 'carb_g', 'fiber_g', 'sugar_g',
    'protein_g', 'vitA_g', 'calcium_mg', 'thiamin_mg', 'zinc_mg',
    'potassium_mg', 'magnesium_mg', 'vitE_mg', 'vitK_mcg', 'vitC_mg',
    'vitB6_mg', 'copper_mg', 'carotene_mg', 'carotene_mcg',
    'cryptoxanthin_mcg', 'lycopene_mcg', 'cholesterol_mg'
]
nutrisi_cols = features[4:]

# Tambahkan kolom nutrisi yang belum ada
for col in nutrisi_cols:
    if col not in menu_df.columns:
        menu_df[col] = 0

# Ambil hanya kolom yang dibutuhkan
menu_df = menu_df[['menu_id', 'Kombinasi Menu', 'Kode Frekuensi'] + nutrisi_cols]


# FastAPI init
app = FastAPI()

class AnakInput(BaseModel):
    umur: int
    tinggi: float

@app.post("/rekomendasi")
def rekomendasi_endpoint(data: AnakInput):
    anak = pd.Series({
        "child_id": 999,
        "Umur (bulan)": data.umur,
        "Tinggi Badan (cm)": data.tinggi,
        "label_gizi": 2
    })

    # Prediksi label gizi
    fitur_input = pd.DataFrame([{
        "child_id": anak["child_id"],
        "Umur (bulan)": anak["Umur (bulan)"],
        "Tinggi Badan (cm)": anak["Tinggi Badan (cm)"],
        "menu_id": 0
    }], columns=features).fillna(0)

    # Validasi tipe data dan isi
    if fitur_input.select_dtypes(include=[np.number]).shape[1] != fitur_input.shape[1]:
        raise HTTPException(status_code=400, detail="Input mengandung data non-numerik.")
    if fitur_input.isnull().values.any():
        raise HTTPException(status_code=400, detail="Input mengandung nilai kosong.")

    try:
        booster = model.get_booster()
        if booster.feature_names is None:
            # Jika model tidak menyimpan nama fitur
            pred_gizi = int(model.predict(fitur_input.to_numpy()).reshape(-1)[0])
        else:
            pred_gizi = int(model.predict(fitur_input).reshape(-1)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")

    anak["label_gizi"] = pred_gizi

    # Gizi label map
    gizi_map = {
        0: "Sangat Pendek – Risiko tinggi stunting",
        1: "Pendek – Perlu perhatian gizi",
        2: "Normal – Pertumbuhan baik",
        3: "Tinggi – Di atas rata-rata"
    }

    hasil = rekomendasi_makanan(anak, menu_df, model, features, filter_by_label=True)

    return {
        "status_gizi": gizi_map.get(pred_gizi, "Tidak diketahui"),
        "rekomendasi": hasil
    }

def rekomendasi_makanan(child_row, menu_df, model, feature_columns, filter_by_label=False):
    umur = child_row["Umur (bulan)"]

    if umur <= 5:
        return [
            {
                "menu": "ASI Eksklusif",
                "label_prediksi": "Rekomendasi utama",
                "frekuensi": "Sesuai kebutuhan bayi",
                "nutrisi": {"Keterangan": "ASI eksklusif sudah mencukupi gizi bayi <6 bulan"}
            },
            {
                "menu": "Susu Formula",
                "label_prediksi": "Alternatif jika ASI tidak tersedia",
                "frekuensi": "Sesuai dosis kemasan",
                "nutrisi": {"Keterangan": "Pastikan susu formula sesuai anjuran dokter"}
            }
        ]

    frekuensi_map = {
        "F1": "2 kali makanan utama per hari",
        "F2": "3 kali makanan utama per hari",
        "F3": "3x makanan utama + 1x snack per hari",
        "F4": "3x makanan utama + 2x snack per hari"
    }

    gizi_map = {
        0: "Sangat Pendek – Risiko tinggi stunting",
        1: "Pendek – Perlu perhatian gizi",
        2: "Normal – Pertumbuhan baik",
        3: "Tinggi – Di atas rata-rata"
    }

    rekomendasi = []
    target_label = int(child_row.get("label_gizi", 2))

    for _, menu in menu_df.iterrows():
        missing = set(features[4:]) - set(menu_df.columns)
        if missing:
            raise ValueError(f"Kolom nutrisi berikut hilang di menu_df: {missing}")

        row = {
            "child_id": child_row["child_id"],
            "Umur (bulan)": umur,
            "Tinggi Badan (cm)": child_row["Tinggi Badan (cm)"],
            "menu_id": menu["menu_id"]
        }

        for col in feature_columns[4:]:  # hanya kolom nutrisi
            row[col] = menu.get(col, 0)

        row_df = pd.DataFrame([row])[feature_columns].fillna(0)

        try:
            if model.get_booster().feature_names is None:
                pred = int(model.predict(row_df.to_numpy()).reshape(-1)[0])
            else:
                pred = int(model.predict(row_df).reshape(-1)[0])
        except:
            pred = -1

        if filter_by_label and pred != target_label:
            continue

        nutrisi = {
            "kalori (kkal)": round(menu.get("calories_kcal", 0), 2),
            "protein (g)": round(menu.get("protein_g", 0), 2),
            "karbohidrat (g)": round(menu.get("carb_g", 0), 2),
            "lemak (g)": round(menu.get("fats_g", 0), 2),
            "zat besi (mg)": round(menu.get("zinc_mg", 0), 2),
            "vitamin A (g)": round(menu.get("vitA_g", 0), 2)
        }

        rekomendasi.append({
            "menu": menu.get("Kombinasi Menu", f"Menu {menu['menu_id']}"),
            "label_prediksi": gizi_map.get(pred, "Tidak diketahui"),
            "frekuensi": frekuensi_map.get(menu.get("Kode Frekuensi", "F2"), "Frekuensi tidak diketahui"),
            "nutrisi": nutrisi
        })

    if not rekomendasi:
        fallback = menu_df.sample(5)
        for _, menu in fallback.iterrows():
            rekomendasi.append({
                "menu": menu.get("Kombinasi Menu", f"Menu {menu['menu_id']}"),
                "label_prediksi": "Alternatif (fallback)",
                "frekuensi": frekuensi_map.get(menu.get("Kode Frekuensi", "F2"), "Tidak diketahui"),
                "nutrisi": {
                    "kalori (kkal)": round(menu.get("calories_kcal", 0), 2),
                    "protein (g)": round(menu.get("protein_g", 0), 2),
                    "karbohidrat (g)": round(menu.get("carb_g", 0), 2),
                    "lemak (g)": round(menu.get("fats_g", 0), 2),
                    "zat besi (mg)": round(menu.get("zinc_mg", 0), 2),
                    "vitamin A (g)": round(menu.get("vitA_g", 0), 2)
                }
            })

    return rekomendasi[:5]
