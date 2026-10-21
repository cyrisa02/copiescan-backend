import os
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from groq import Groq
import instructor
from dotenv import load_dotenv
import httpx

load_dotenv()

app = FastAPI(title="FactureScan & Correcteur IA API")

# --- CONFIGURATION CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION CLIENT (VIA .ENV) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_APPS_SCRIPT_URL = os.getenv("GOOGLE_APPS_SCRIPT_URL")

if not GROQ_API_KEY or not GOOGLE_APPS_SCRIPT_URL:
    raise ValueError("Variables d'environnement GROQ_API_KEY ou GOOGLE_APPS_SCRIPT_URL manquantes")

# --- SCHÉMAS PYDANTIC FACTURE ---
class ExtractionFacture(BaseModel):
    date: str = Field(description="Format YYYY-MM-DD")
    fournisseur: str
    montant_ht: float
    montant_tva: float
    montant_ttc: float
    numero_facture: Optional[str] = None
    lignes: Optional[List[dict]] = None

# --- SCHÉMAS PYDANTIC CORRECTION FRANÇAIS ---
class PageAnalysis(BaseModel):
    """Schéma compact pour l'analyse d'une page"""
    contenu: str = Field(description="Résumé du contenu (50 mots max)")
    erreurs: List[str] = Field(description="Max 3 erreurs principales")
    points_forts: List[str] = Field(description="Max 2 points forts")

class ErreurIdentifiee(BaseModel):
    type_erreur: str = Field(description="Type: Orthographe, Syntaxe, Contre-sens, Hors-sujet, Structure")
    citation: Optional[str] = Field(description="Extrait de la copie", default=None)
    explication: str = Field(description="Pourquoi c'est une erreur")
    suggestion: str = Field(description="Comment corriger")

class CorrectionResult(BaseModel):
    note: float = Field(description="Note sur 20")
    appreciation_generale: str = Field(description="Appréciation globale (2 phrases max)")
    points_forts: List[str] = Field(description="3 points forts max")
    axes_amelioration: List[str] = Field(description="3 axes d'amélioration max")
    erreurs_details: List[ErreurIdentifiee] = Field(description="Erreurs principales")
    structure_evaluee: str = Field(description="Évaluation structure (1 phrase)")

# --- SCHÉMAS DE REQUÊTE ---
class ImageRequest(BaseModel):
    image_base64: str

class CopyCorrectionRequest(BaseModel):
    images_base64: List[str] = Field(description="Liste des images Base64 (multi-pages)")
    matiere: str = Field(description="Matière scolaire", default="Français")
    consigne: str = Field(description="Consigne ou sujet")
    bareme: float = Field(description="Note maximale", default=20)

class ChatMessage(BaseModel):
    role: str = Field(description="user ou assistant")
    content: str = Field(description="Contenu du message")

class ChatCorrectionRequest(BaseModel):
    messages: List[ChatMessage] = Field(description="Historique de la conversation")
    correction_context: str = Field(description="La correction originale complète")

# --- CLIENT GROQ ---
client = instructor.from_groq(
    Groq(api_key=GROQ_API_KEY),
    mode=instructor.Mode.JSON
)

# --- FONCTION WEBHOOK ASYNCHRONE ---
async def send_to_google_sheets(payload: dict):
    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.post(
                GOOGLE_APPS_SCRIPT_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"[SHEETS] Status: {response.status_code}")
    except Exception as e:
        print(f"[SHEETS] Erreur critique: {e}")

# --- ENDPOINT FACTURE ---
@app.post("/extract")
async def extract_invoice(request: ImageRequest, background_tasks: BackgroundTasks):
    try:
        image_base64 = request.image_base64
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        image_data_uri = f"data:image/jpeg;base64,{image_base64}"
        
        extraction = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            response_model=ExtractionFacture,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extrais les données de cette facture."},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }
            ],
            max_tokens=800
        )
        
        background_tasks.add_task(send_to_google_sheets, extraction.model_dump())
        return extraction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT CORRECTION COPIES MULTI-PAGES ---
@app.post("/correct-copy")
async def correct_copy(request: CopyCorrectionRequest):
    try:
        print(f"[CORRECTION] Réception de {len(request.images_base64)} page(s)")
        
        pages_analyses = []
        
        for idx, img_base64 in enumerate(request.images_base64):
            print(f"[CORRECTION] Traitement page {idx + 1}/{len(request.images_base64)}")
            
            if "," in img_base64:
                img_base64 = img_base64.split(",")[1]
            image_data_uri = f"data:image/jpeg;base64,{img_base64}"
            
            system_prompt = """Prof de français. Analyse cette page.
            Réponds en JSON compact avec :
            - contenu: résumé 50 mots
            - erreurs: max 3 erreurs
            - points_forts: max 2"""
            
            page_analysis = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                response_model=PageAnalysis,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Page {idx + 1}"},
                            {"type": "image_url", "image_url": {"url": image_data_uri}}
                        ]
                    }
                ],
                max_tokens=700
            )
            pages_analyses.append(page_analysis)
            print(f"[CORRECTION] Page {idx + 1} analysée")
            
            if idx < len(request.images_base64) - 1:
                print("[CORRECTION] Pause 5s (limite Groq)...")
                await asyncio.sleep(5)
        
        print("[CORRECTION] Synthèse finale...")
        
        synthesis_parts = [f"Sujet: {request.consigne}\nBarème: {request.bareme}/20\n"]
        
        for idx, analysis in enumerate(pages_analyses):
            synthesis_parts.append(f"=== PAGE {idx + 1} ===")
            synthesis_parts.append(f"Contenu: {analysis.contenu}")
            synthesis_parts.append(f"Erreurs: {'; '.join(analysis.erreurs)}")
            synthesis_parts.append(f"Points forts: {'; '.join(analysis.points_forts)}")
            synthesis_parts.append("")
        
        synthesis_text = "\n".join(synthesis_parts)
        
        system_prompt_final = """Prof de français, correcteur Bac.
        Fournis une évaluation complète basée sur les analyses de pages.
        Sois concis et précis."""
        
        correction = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            response_model=CorrectionResult,
            messages=[
                {"role": "system", "content": system_prompt_final},
                {"role": "user", "content": synthesis_text}
            ],
            max_tokens=900
        )
        
        print("[CORRECTION] Terminée avec succès")
        return correction
        
    except Exception as e:
        print(f"[CORRECTION] Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

# --- ENDPOINT CHAT CORRECTION ---
@app.post("/chat-correction")
async def chat_correction(request: ChatCorrectionRequest):
    try:
        print(f"[CHAT] Nouveau message, historique: {len(request.messages)} messages")
        
        system_prompt = f"""Tu es un professeur de Lettres Modernes bienveillant.
Tu viens de corriger cette copie avec l'évaluation suivante :

{request.correction_context}

Tu réponds aux questions sur cette correction. Tes réponses doivent être :
- Concises (max 100 mots)
- Pédagogiques et encourageantes
- Basées strictement sur la correction fournie
- En français"""

        api_messages = [{"role": "system", "content": system_prompt}]
        
        for msg in request.messages:
            api_messages.append({"role": msg.role, "content": msg.content})
        
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=api_messages,
            max_tokens=400
        )
        
        return {"response": response.choices[0].message.content}
        
    except Exception as e:
        print(f"[CHAT] Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))