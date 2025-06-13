from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import uuid, os

from main import measure_all  # gunakan versi asli kamu

app = FastAPI()

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Halo dari FastAPI di Hugging Face!"}

@app.post("/predict-babyheight")
async def predict_height(image: UploadFile = File(...)):
    try:
        if not image.content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"error": "File harus berupa gambar."})

        file_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")
        output_path = os.path.join(UPLOAD_DIR, f"{file_id}_out.jpg")

        with open(input_path, "wb") as f:
            f.write(await image.read())

        result_cm, result_img_path = measure_all(input_path, output_path)

        os.remove(input_path)

        if result_cm is None:
            return JSONResponse(status_code=422, content={"error": "Gagal mengukur tinggi bayi dari gambar ini."})

        return {
            "status": "success",
            "predicted_height_cm": round(result_cm, 2),
            "annotated_image_url": f"/result-image/{os.path.basename(result_img_path)}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/result-image/{filename}")
def get_result_image(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"error": "Gambar tidak ditemukan."})
