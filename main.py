from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client
from fastapi.responses import HTMLResponse
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

# Using felixrosberg/face-swap, reliable with known API
client = Client("felixrosberg/face-swap", verbose=False)

@app.get("/ping")
def ping():
    return HTMLResponse(content="<h2>✅ Backend is alive!</h2>", status_code=200)

@app.post("/swap_faces")
async def swap_faces(target: UploadFile = File(...), source: UploadFile = File(...)):
    temp_files = []
    try:
        # Read uploaded files
        target_bytes = await target.read()
        source_bytes = await source.read()

        # Create temporary files for images (paths are serializable)
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

        print(f"Temp files created: target={target_path}, source={source_path}")  # Debug

        # Call Gradio Space: target_path (body), source_path (face), 0 (no anonym), 0 (no adv), [] (basic swap)
        # api_name="/run_inference"
        result = client.predict(
            target_path,
            source_path,
            0,  # Anonymization ratio (0 = full source identity)
            0,  # Adversarial defense (0 = none)
            [],  # Settings (empty for basic single output)
            api_name="/run_inference"
        )

        print(f"Result type: {type(result)}")  # Debug

        # Handle possible list output; take the single image
        output = result[0] if isinstance(result, (list, tuple)) and len(result) > 0 else result
        print(f"Output type: {type(output)}")  # Debug

        # Convert to base64 string (robust handling)
        if isinstance(output, (bytes, bytearray)):
            img_bytes = output
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            print("Handled as bytes")
        elif isinstance(output, Image.Image):
            buffered = io.BytesIO()
            output.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            print("Handled as PIL.Image")
        elif isinstance(output, np.ndarray):
            # Ensure uint8 RGB
            if output.dtype != np.uint8:
                output = np.clip(output, 0, 1) * 255
                output = np.uint8(output)
            if len(output.shape) == 2:  # Grayscale to RGB
                output = np.stack((output,) * 3, axis=-1)
            img = Image.fromarray(output)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            print("Handled as np.ndarray")
        elif isinstance(output, str):
            # Handle path or base64 str
            if os.path.isfile(output):
                with open(output, "rb") as f:
                    img_bytes = f.read()
                base64_image = base64.b64encode(img_bytes).decode("utf-8")
                print("Handled as local file path")
            else:
                # Assume it's already base64
                base64_image = output
                print("Handled as str (URL/base64)")
        else:
            print(f"Unexpected output type: {type(output)}")
            return {"error": f"Unexpected output type: {type(output)}"}

        # Ensure base64_image is str
        if not isinstance(base64_image, str):
            base64_image = base64_image.decode("utf-8")

        print(f"Base64 image generated: type={type(base64_image)}, length={len(base64_image)}")  # Debug
        return {"result": base64_image}

    except Exception as e:
        print(f"Exception details: {type(e).__name__}: {e}")
        print(traceback.format_exc())  # Full traceback
        return {"error": str(e)}
    finally:
        # Clean up temp files
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"Cleaned up: {temp_path}")
            except Exception as cleanup_e:
                print(f"Cleanup error: {cleanup_e}")
