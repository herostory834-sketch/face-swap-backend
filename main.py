from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import io
import base64
import traceback
import requests

app = FastAPI(title="Face Swap Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SPACE_URL = "https://ezioruan-roop.hf.space/api/predict"

@app.get("/ping")
def ping():
    return HTMLResponse(content="<h2>✅ Backend is alive!</h2>", status_code=200)

@app.post("/swap_faces")
async def swap_faces(target: UploadFile = File(...), source: UploadFile = File(...)):
    try:
        # Read uploaded files
        target_bytes = await target.read()
        source_bytes = await source.read()

        # Encode images to base64 strings with data URI prefix (required for Gradio image inputs)
        source_base64 = "data:image/png;base64," + base64.b64encode(source_bytes).decode("utf-8")
        target_base64 = "data:image/png;base64," + base64.b64encode(target_bytes).decode("utf-8")

        print(f"Images encoded with prefix: source length={len(source_base64)}, target length={len(target_base64)}")  # Debug

        # Prepare payload for Gradio API (roop expects: source_image, target_image, face_enhancer=False, restore_face=False, etc.)
        # Start with minimal: just the two images; add defaults if needed
        payload = {
            "data": [
                source_base64,      # source_image (face to swap in)
                target_base64,      # target_image (body to swap onto)
                False,              # face_enhancer
                False               # restore_face
            ]
        }

        # Send POST request to HF Space API
        response = requests.post(
            SPACE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=180  # Further increased timeout for potential enhancements
        )

        print(f"HF API status: {response.status_code}, response: {response.text[:500]}...")  # Debug (truncated)

        if response.status_code == 200:
            result = response.json()
            if "data" in result and result["data"] is not None and len(result["data"]) > 0:
                # Extract the output image base64 (remove prefix if present)
                output_base64_full = result["data"][0]
                if output_base64_full.startswith("data:image"):
                    output_base64 = output_base64_full.split(",")[1]
                else:
                    output_base64 = output_base64_full
                print(f"Received base64 length: {len(output_base64)}")  # Debug
                return {"result": output_base64}
            else:
                return {"error": f"No valid data in response: {result}"}
        else:
            return {"error": f"HF API error: {response.status_code} - {response.text}"}

    except Exception as e:
        print(f"Exception details: {type(e).__name__}: {e}")
        print(traceback.format_exc())  # Full traceback
        return {"error": str(e)}
