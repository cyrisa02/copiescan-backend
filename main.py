import os
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

# --- SCHÉMAS PYDANTIC ---

class ExtractionFacture(BaseModel):
    date: str = Field(description="Format YYYY-MM-DD")
    fournisseur: str
    montant_ht: float
    montant_tva: float
    montant_ttc: float
    numero_facture: Optional[str] = None
    lignes: Optional[List[dict]] = None

class ErreurIdentifiee(BaseModel):
    type_erreur: str = Field(description="Type: Orthographe, Syntaxe, Contre-sens, Hors-sujet, Structure")
    citation: Optional[str] = Field(description="Extrait de la copie concernée", default=None)
    explication: str = Field(description="Pourquoi c'est une erreur ou une faiblesse")
    suggestion: str = Field(description="Comment reformuler ou améliorer")

class CorrectionResult(BaseModel):
    note: float = Field(description="Note sur 20")
    appreciation_generale: str = Field(description="Appréciation globale style bulletin scolaire")
    points_forts: List[str] = Field(description="Qualités rédactionnelles et analytiques")
    axes_amelioration: List[str] = Field(description="Conseils précis pour monter en niveau")
    erreurs_details: List[ErreurIdentifiee] = Field(description="Analyse détaillée des faiblesses")
    structure_evaluee: str = Field(description="Évaluation de la structure (intro, développement, conclusion)")

class ImageRequest(BaseModel):
    image_base64: str

class CopyCorrectionRequest(BaseModel):
    image_base64: str
    matiere: str = Field(description="Matière scolaire", default="Français")
    consigne: str = Field(description="Consigne ou sujet de la dissertation/commentaire")
    bareme: float = Field(description="Note maximale possible", default=20)

# --- CLIENT GROQ ---
client = instructor.from_groq(
    Groq(api_key=GROQ_API_KEY),
    mode=instructor.Mode.JSON
)

# --- FONCTION WEBHOOK ASYNCHRONE ---
async def send_to_google_sheets(payload: dict):
    """Envoie les données vers le Sheet configuré en arrière-plan"""
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

@app.post("/extract")
async def extract_invoice(request: ImageRequest, background_tasks: BackgroundTasks):
    try:
        image_base64 = request.image_base64
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        image_data_uri = f"data:image/jpeg;base64,{image_base64}"
        
        # Extraction IA
        extraction = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            response_model=ExtractionFacture,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extrais les données de cette facture. Retourne null si illisible."},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Envoi AUTOMATIQUE vers le sheet (non bloquant)
        background_tasks.add_task(send_to_google_sheets, extraction.model_dump())
        
        return extraction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/correct-copy")
async def correct_copy(request: CopyCorrectionRequest):
    """
    Corrige une copie de français (dissertation/commentaire).
    RGPD: Aucune donnée n'est stockée. Traitement à la volée uniquement.
    """
    try:
        image_base64 = request.image_base64
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        image_data_uri = f"data:image/jpeg;base64,{image_base64}"
        
        system_prompt = """Tu es un professeur de Lettres Modernes expérimenté, correcteur au Baccalauréat.
Ta mission est de corriger une copie de français (dissertation ou commentaire composé).

Critères d'évaluation stricts :
1. Respect du sujet : L'élève a-t-il répondu à la problématique ? Y a-t-il des hors-sujets ?
2. Structure : La copie suit-elle le plan attendu (Intro claire, Développement en parties/sous-parties, Conclusion bilans) ?
3. Argumentation : Les idées sont-elles étayées par des exemples littéraires précis ?
4. Expression : Qualité de la langue, richesse du vocabulaire, absence de fautes d'orthographe/grammaire.

Ton ton doit être :
- Exigeant sur la méthode (plan, citations).
- Bienveillant sur le fond (valoriser les efforts d'analyse).
- Constructif (donner des pistes concrètes pour améliorer la note).

Si la copie est illisible ou trop courte, signale-le immédiatement."""

        # Correction IA
        correction = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            response_model=CorrectionResult,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": f"Corrige cette copie.\n\nSujet/Consigne: {request.consigne}\nBarème: {request.bareme}/20\n\nAnalyse la copie et fournis une évaluation complète."
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": image_data_uri}
                        }
                    ]
                }
            ],
            max_tokens=2500
        )
        
        # PAS de stockage - RGPD compliant
        return correction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la correction: {str(e)}")