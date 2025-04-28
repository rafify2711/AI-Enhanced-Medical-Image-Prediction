from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io

# استيراد الموديلات والفانكشنات
from models import BaseModel, Covid19Model, BrainTumorModel, KidneyStoneModel, SkinCancerModel, \
    TuberculosisModel, BoneFractureModel, AlzheimerModel, EyeDiseasesModel, DentalModel
from prescription import predict
from chatbot import MedicalChatbot

app = FastAPI()

# ✅ تحميل جميع الموديلات مرة واحدة عند بداية التشغيل
covid19_model = Covid19Model()
brain_tumor_model = BrainTumorModel()
skin_cancer_model = SkinCancerModel()
kidney_stone_model = KidneyStoneModel()
tuberculosis_model = TuberculosisModel()
bone_fracture_model = BoneFractureModel()
alzheimer_model = AlzheimerModel()
eye_diseases_model = EyeDiseasesModel()

# ✅ هيلبر موحد للتعامل مع الموديلات
async def predict_helper(file: UploadFile, model: BaseModel) -> Dict[str, Any]:
    """Helper function to handle predictions for all models."""
    try:
        contents = await file.read()
        img = io.BytesIO(contents)
        res = model.predict(img)
        return {**res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Endpoints مرتبة
@app.post("/predict/covid19/")
async def predict_covid19(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await predict_helper(file, covid19_model)

@app.post("/predict/brain-tumor/")
async def predict_brain_tumor(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await predict_helper(file, brain_tumor_model)

@app.post("/predict/skin-cancer/")
async def predict_skin_cancer(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await predict_helper(file, skin_cancer_model)

@app.post("/predict/kidney-stone/")
async def predict_kidney_stone(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await predict_helper(file, kidney_stone_model)

@app.post("/predict/tuberculosis/")
async def predict_tuberculosis(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await predict_helper(file, tuberculosis_model)

@app.post("/predict/bone-fracture/")
async def predict_bone_fracture(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await predict_helper(file, bone_fracture_model)

@app.post("/predict/alzheimer/")
async def predict_alzheimer(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await predict_helper(file, alzheimer_model)

@app.post("/predict/eye-diseases/")
async def predict_eye_diseases(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await predict_helper(file, eye_diseases_model)

@app.post("/predict/dental/")
async def predict_dental(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await predict_helper(file, dental_model)

@app.post("/predict/prescription/")
async def predict_prescription(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for prescription predictions."""
    try:
        contents = await file.read()
        img = io.BytesIO(contents)
        res = predict(img)  # انت هنا تستخدم فانكشن مش موديل
        return {**res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(prompt: str):
    """Endpoint to interact with the chatbot."""
    chatbot = MedicalChatbot()
    return StreamingResponse(chatbot.get_chat_response(prompt), media_type="text/plain")

# ✅ Root endpoint
@app.get('/')
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "API is running ✅"}
