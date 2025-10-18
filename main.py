from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client

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
        # Return base64 image string
        return {"result": result[0]}
    except Exception as e:
        return {"error": str(e)}
