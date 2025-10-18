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

client = Client("felixrosberg/face-swap")

@app.get("/ping")
def ping():
    return HTMLResponse(content="<h2>✅ Backend is alive!</h2>", status_code=200)

@app.post("/swap_faces")
async def swap_faces(target: UploadFile = File(...), source: UploadFile = File(...)):
    try:
        target_bytes = await target.read()
        source_bytes = await source.read()

        # Call Gradio Space
        result = client.predict(
            target_bytes,
            source_bytes,
            0,            # Anonymization ratio
            0,            # Adversarial defense ratio
            ["Compare"],  # Mode
            api_name="/run_inference"
        )

        output = result[0]

        # Convert to base64
        if isinstance(output, (bytes, bytearray)):
            base64_image = base64.b64encode(output).decode("utf-8")
        elif isinstance(output, Image.Image):
            buffered = io.BytesIO()
            output.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        elif isinstance(output, np.ndarray):
            img = Image.fromarray(output)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        elif isinstance(output, str):
            # Already base64 or URL
            base64_image = output
        else:
            return {"error": f"Unexpected output type: {type(output)}"}

        return {"result": base64_image}

    except Exception as e:
        return {"error": str(e)}
