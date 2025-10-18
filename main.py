from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import base64
import numpy as np
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

client = Client("felixrosberg/face-swap")

@app.get("/ping")
def ping():
    return HTMLResponse(content="<h2>✅ Backend is alive!</h2>", status_code=200)

@app.post("/swap_faces")
async def swap_faces(target: UploadFile = File(...), source: UploadFile = File(...)):
    temp_files = []
    try:
        target_bytes = await target.read()
        source_bytes = await source.read()

        # Create temporary files for images (gradio_client expects paths/URLs)
        target_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        target_file.write(target_bytes)
        target_file.close()
        target_path = target_file.name
        temp_files.append(target_path)

        source_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        source_file.write(source_bytes)
        source_file.close()
        source_path = source_file.name
        temp_files.append(source_path)

        print(f"Created temp files: {target_path}, {source_path}")  # Debug

        # Call Gradio Space with file paths
        result = client.predict(
            target_path,
            source_path,
            0,            # Anonymization ratio
            0,            # Adversarial defense ratio
            ["Compare"],  # Mode (may return multiple outputs)
            api_name="/run_inference"
        )

        print(f"Result type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")  # Debug

        # Assume multiple outputs for "Compare" mode; take the last one (swapped image)
        output = result[-1] if isinstance(result, (list, tuple)) else result
        print(f"Output type: {type(output)}")  # Debug

        # Convert to base64 string
        if isinstance(output, (bytes, bytearray)):
            img_bytes = output
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            print("Handled as bytes")  # Debug
        elif isinstance(output, Image.Image):
            buffered = io.BytesIO()
            output.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            print("Handled as PIL.Image")  # Debug
        elif isinstance(output, np.ndarray):
            # Ensure it's uint8 RGB
            if output.dtype != np.uint8:
                output = (np.clip(output, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(output)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            print("Handled as np.ndarray")  # Debug
        elif isinstance(output, str):
            # Could be temp path, URL, or pre-encoded base64
            if os.path.isfile(output):  # Local temp file path from Gradio client
                with open(output, "rb") as f:
                    img_bytes = f.read()
                base64_image = base64.b64encode(img_bytes).decode("utf-8")
                print("Handled as local file path")  # Debug
            else:
                # Assume URL or existing base64 str
                base64_image = output
                print("Handled as str (URL/base64)")  # Debug
        else:
            print(f"Unexpected output type: {type(output)}")  # Debug
            return {"error": f"Unexpected output type: {type(output)}"}

        print(f"Base64 image type: {type(base64_image)}, length: {len(base64_image)}")  # Debug: Should be str
        return {"result": base64_image}

    except Exception as e:
        print(f"Exception details: {type(e).__name__}: {e}")  # More debug
        return {"error": str(e)}
    finally:
        # Clean up temp files
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"Cleaned up: {temp_path}")  # Debug
            except Exception as cleanup_e:
                print(f"Cleanup error: {cleanup_e}")
