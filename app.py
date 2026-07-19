from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
from model import detector
import shutil
import os
import traceback

app = FastAPI(
    title="YOLOv11 Object Detection API",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "YOLOv11 Object Detection API is Running"
    }


# ---------------- IMAGE DETECTION ----------------

@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    try:

        image = Image.open(file.file).convert("RGB")

        results = detector.predict_image(image)

        return {
            "success": True,
            "objects": results
        }

    except Exception as e:

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# ---------------- VIDEO DETECTION ----------------

@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    try:

        os.makedirs("uploads", exist_ok=True)

        video_path = os.path.join("uploads", file.filename)

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("Uploaded Video:", video_path)

        output_video = detector.predict_video(video_path)

        print("Detected Video:", output_video)

        if not os.path.exists(output_video):
            raise FileNotFoundError(f"Video not found: {output_video}")

        return FileResponse(
            output_video,
            media_type="video/mp4",
            filename=os.path.basename(output_video)
        )

    except Exception as e:

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8006,
        reload=True
    )