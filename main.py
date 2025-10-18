from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, file
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import base64
import numpy as np
import os

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
    try:
        target_bytes = await target.read()
        source_bytes = await source.read()

        print(f"Input bytes lengths: target={len(target_bytes)}, source={len(source_bytes)}")  # Debug

        # Prepare inputs using gradio_client.file for binary data
        target_file = file(data=target_bytes)
        source_file = file(data=source_bytes)

        # Call Gradio Space
        result = client.predict(
            target_file,
            source_file,
            0,            # Anonymization ratio
            0,            # Adversarial defense ratio
            ["Compare"],  # Mode
            api_name="/run_inference"
        )

        print(f"Result type: {type(result)}")  # Debug
        if isinstance(result, (list, tuple)):
            output = result[0]  # Take first (or only) output; adjust if multiple
            print(f"List result, first item type: {type(output)}")  # Debug
        else:
            output = result
        print(f"Output type: {type(output)}")  # Debug

        # If None, early error
        if output is None:
            return {"error": "No output generated (possibly no faces detected)"}

        # Convert to base64 string, ensuring str in all cases
        base64_image = None
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
            # Ensure uint8 RGB
            if output.dtype != np.uint8:
                output = (np.clip(output, 0, 1) * 255).astype(np.uint8)
            if len(output.shape) == 3 and output.shape[2] == 3:  # HWC
                pass
            elif len(output.shape) == 3 and output.shape[0] == 3:  # CHW
                output = np.transpose(output, (1, 2, 0))
            img = Image.fromarray(output)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            print("Handled as np.ndarray")  # Debug
        elif isinstance(output, str):
            # Handle path, URL, or base64 str
            if os.path.isfile(output):
                with open(output, "rb") as f:
                    img_bytes = f.read()
                base64_image = base64.b64encode(img_bytes).decode("utf-8")
                print("Handled as local file path")  # Debug
            else:
                # Assume it's already base64 or URL; for URL, you'd fetch, but assume base64 for now
                base64_image = output
                print("Handled as str (assumed base64/URL)")  # Debug
        else:
            print(f"Unexpected output type: {type(output)}")  # Debug
            return {"error": f"Unexpected output type: {type(output)}"}

        # Final safety check: ensure it's str
        if not isinstance(base64_image, str):
            print(f"Warning: base64_image is not str, type: {type(base64_image)}")  # Debug
            base64_image = base64.b64encode(base64_image).decode("utf-8") if isinstance(base64_image, bytes) else str(base64_image)

        print(f"Final base64_image type: {type(base64_image)}, length: {len(base64_image) if base64_image else 0}")  # Debug: Should be str
        return {"result": base64_image}

    except Exception as e:
        print(f"Exception details: {type(e).__name__}: {str(e)}")  # More debug
        import traceback
        print(traceback.format_exc())  # Full traceback for debugging
        return {"error": str(e)}
