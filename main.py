from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from gradio_client import Client
from PIL import Image
import io
import base64
import numpy as np
import traceback
import os
import tempfile

app = FastAPI(title="Face Swap Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = None  # Don't initialize yet

def get_client():
    global client
    if client is None:
        try:
            client = Client("felixrosberg/face-swap", verbose=False)
            print("✅ Hugging Face client initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize Hugging Face client: {e}")
            client = None
    return client

@app.get("/ping")
def ping():
    return HTMLResponse(content="<h2>✅ Backend is alive!</h2>", status_code=200)

@app.post("/swap_faces")
async def swap_faces(target: UploadFile = File(...), source: UploadFile = File(...)):
    temp_files = []
    try:
        gr_client = get_client()
        if gr_client is None:
            return {"error": "Face swap service unavailable. Please try again later."}

        # --- Read uploaded files ---
        target_bytes = await target.read()
        source_bytes = await source.read()

        # --- Create temporary files ---
        source_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        with open(source_path, "wb") as f:
            f.write(source_bytes)
        temp_files.append(source_path)

        target_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        with open(target_path, "wb") as f:
            f.write(target_bytes)
        temp_files.append(target_path)

        print(f"Temp files created: source={source_path}, target={target_path}")

        # --- Call Hugging Face Space ---
        result = gr_client.predict(
            source_path,
            target_path,
            False,
            api_name="/run_inference"
        )

        output = result[0] if isinstance(result, (list, tuple)) else result

        # --- Convert result to base64 ---
        if isinstance(output, (bytes, bytearray)):
            base64_image = base64.b64encode(output).decode("utf-8")
        elif isinstance(output, Image.Image):
            buf = io.BytesIO()
            output.save(buf, format="PNG")
            base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
        elif isinstance(output, np.ndarray):
            img = Image.fromarray(output.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
        elif isinstance(output, str) and os.path.isfile(output):
            with open(output, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(output, str):
            base64_image = output
        else:
            return {"error": f"Unexpected output type: {type(output)}"}

        return {"result": base64_image}

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}
    finally:
        for path in temp_files:
            try:
                os.remove(path)
            except Exception:
                pass
