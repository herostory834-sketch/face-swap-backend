from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import base64
import numpy as np

app = FastAPI(title="Face Swap Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Using felixrosberg/face-swap with verbose logging
client = Client("felixrosberg/face-swap", verbose=True)

@app.get("/ping")
def ping():
    return HTMLResponse(content="<h2>✅ Backend is alive!</h2>", status_code=200)

@app.post("/swap_faces")
async def swap_faces(target: UploadFile = File(...), source: UploadFile = File(...)):
    try:
        target_bytes = await target.read()
        source_bytes = await source.read()

        # Convert bytes to PIL Images
        target_img = Image.open(io.BytesIO(target_bytes))
        source_img = Image.open(io.BytesIO(source_bytes))

        print(f"Input types: target={type(target_img)}, source={type(source_img)}")  # Debug

        # Call Gradio Space with PIL images
        # Parameters: target, source, anonymization_ratio=0 (no anonymization), adversarial_defense=0 (no defense), settings=[] (simple swap, no modes)
        # Use ["Compare"] if you want side-by-side output
        result = client.predict(
            target_img,
            source_img,
            0,  # Anonymization ratio (0 for full identity preservation)
            0,  # Adversarial defense ratio (0 for no defense)
            [],  # Settings: empty for basic swap; use ["Compare"] for side-by-side
            api_name="/run_inference"
        )

        print(f"Result type: {type(result)}")  # Debug

        # Handle possible list output (e.g., if multiple)
        output = result[0] if isinstance(result, (list, tuple)) else result
        print(f"Output type: {type(output)}")  # Debug

        # Convert to base64 string (robust handling for all common types)
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
            # Ensure uint8 RGB/A
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
            import os
            if os.path.isfile(output):
                with open(output, "rb") as f:
                    img_bytes = f.read()
                base64_image = base64.b64encode(img_bytes).decode("utf-8")
                print("Handled as local file path")
            else:
                base64_image = output
                print("Handled as str (URL/base64)")
        else:
            print(f"Unexpected output type: {type(output)}")
            return {"error": f"Unexpected output type: {type(output)}"}

        print(f"Base64 image type: {type(base64_image)}, length: {len(base64_image) if isinstance(base64_image, str) else 'N/A'}")  # Debug: Should be str
        return {"result": base64_image}

    except Exception as e:
        print(f"Exception details: {type(e).__name__}: {e}")
        import traceback
        print(traceback.format_exc())  # Full traceback for debugging
        return {"error": str(e)}
