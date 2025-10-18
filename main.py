from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client
from fastapi.responses import HTMLResponse
import base64

app = FastAPI(title="Face Swap Backend")

# Allow Flutter app to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Hugging Face Space
client = Client("felixrosberg/face-swap")

@app.get("/ping")
def ping():
    html_content = """
    <html>
        <head><title>Face Swap Backend</title></head>
        <body><h2>✅ Backend is alive!</h2></body>
    </html>
    """
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

        # If result is bytes, encode to base64 for JSON serialization
        if isinstance(result[0], (bytes, bytearray)):
            base64_image = base64.b64encode(result[0]).decode("utf-8")
            return {"result": base64_image}

        # If result is already a string (base64), return as-is
        return {"result": result[0]}

    except Exception as e:
        return {"error": str(e)}
