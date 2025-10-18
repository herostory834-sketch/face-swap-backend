from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client
from fastapi.responses import HTMLResponse
import base64

app = FastAPI(title="Face Swap Backend")

# Allow Flutter app to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Hugging Face Space
client = Client("felixrosberg/face-swap")

@app.get("/ping")
def ping():
    html_content = "<html><body><h2>✅ Backend is alive!</h2></body></html>"
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/swap_faces")
async def swap_faces(target: UploadFile = File(...), source: UploadFile = File(...)):
    try:
        target_bytes = await target.read()
        source_bytes = await source.read()

        # Call Gradio Space
        result = client.predict(
            target_bytes,
            source_bytes,
            0,           # Anonymization ratio
            0,           # Adversarial defense ratio
            ["Compare"], # Mode
            api_name="/run_inference"
        )

        # Convert bytes to base64 string if needed
        output = result[0]
        if isinstance(output, (bytes, bytearray)):
            output = base64.b64encode(output).decode("utf-8")

        # Return JSON with base64 string
        return {"result": output}

    except Exception as e:
        return {"error": str(e)}
