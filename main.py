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

# Switched to a more stable space: tuan2308/face-swap
# This space uses InsightFace for simple face swapping with two image inputs
client = Client("tuan2308/face-swap", verbose=True)  # verbose=True for better error logging

@app.get("/ping")
def ping():
    return HTMLResponse(content="<h2>✅ Backend is alive!</h2>", status_code=200)

@app.post("/swap_faces")
async def swap_faces(target: UploadFile = File(...), source: UploadFile = File(...)):
    temp_files = []
    try:
        target_bytes = await target.read()
        source_bytes = await source.read()

        # Note: For tuan2308/face-swap, target is the image to swap INTO (destination),
        # source is the face to swap FROM (source face)
        # Create temporary files for images
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

        print(f"Created temp files: target={target_path}, source={source_path}")  # Debug

        # Call Gradio Space with simplified parameters (no ratios or mode for this space)
        result = client.predict(
            source_path,  # Source face
            target_path,  # Target image
            api_name="/predict"  # Standard for simple Gradio spaces
        )

        print(f"Result type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")  # Debug

        # Assume single output for swapped image; adjust if multiple
        output = result[0] if isinstance(result, (list, tuple)) else result
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
            if output.dtype != np.uint8:
                output = (np.clip(output, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(output)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            print("Handled as np.ndarray")
        elif isinstance(output, str):
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

        print(f"Base64 image type: {type(base64_image)}, length: {len(base64_image)}")  # Should be str
        return {"result": base64_image}

    except Exception as e:
        print(f"Exception details: {type(e).__name__}: {e}")
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
