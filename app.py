from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.diabetes_pipeline import DiabetesRiskPipeline
from services.llm_extractor import QwenExtractor

from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

# เปิดให้ Frontend เรียก API ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลด Pipeline
llm_service = QwenExtractor(HF_TOKEN)

pipeline = DiabetesRiskPipeline(
    model_path="models/evita_diabetes_risk_model.json",
    llm_extractor_func=llm_service
)

@app.get("/")
def root():
    return {"message": "API Running"}

@app.post("/predict")
async def predict(data: dict):
    return pipeline.process_and_predict(data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)