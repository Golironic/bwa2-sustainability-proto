import os
import io
from urllib import response
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
import json
from typing import Optional, Dict, Any, Union
import io
from google.cloud import aiplatform
from dotenv import load_dotenv
import requests

load_dotenv()  # Load environment variables from .env file

# 1. Initialize Gemini Client (Make sure GEMINI_API_KEY is in your environment variables)
client = genai.Client(api_key=os.environ.get("GEMINI_API_Key"))

# 2. Initialize Firebase Firestore (Make sure your service account JSON file is downloaded)
cred = credentials.Certificate(os.environ.get("FIREBASE_CREDENTIALS_PATH"))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Define the exact output structure required for the shared Firestore schema
class CitizenReportSchema(BaseModel):
    detected_issue: str
    severity: int # 1 to 5
    description: str

def analyze_citizen_report(image_input: Union[str, bytes], user_text: str = "") -> Optional[CitizenReportSchema]:
    """
    Sends a citizen photo and optional text to Gemini Multimodal, 
    enforcing a structured JSON output matching the schema.
    """
    try:
        # Load the image using Pillow
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = Image.open(io.BytesIO(image_input))

        img.thumbnail((800, 800))
        
        prompt = f"""
        Analyze this citizen-submitted environmental report image and description.
        Citizen description: "{user_text}"
        
        Task:
        1. Categorize the issue (e.g., 'garbage_burning', 'industrial_pollution', 'stubble_burning', 'illegal_dumping', 'other').
        2. Assign a severity score from 1 (minor) to 5 (critical/hazardous).
        3. Provide a clear, concise summary description for authorities.
        """

        # Call Gemini 3.6 Flash (fast & multimodal) with structured output
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CitizenReportSchema,
                max_output_tokens=1000,),
        )

        # The response text will automatically conform to the JSON schema
        parsed = response.parsed
        if not isinstance(parsed, CitizenReportSchema):
            raise TypeError(f"Unexpected parsed response type: {type(parsed).__name__}")
        return parsed

    except Exception as e:
        print(f"Error processing report with Gemini: {e}")
        return None

def save_to_firestore(collection_name: str, doc_id: str, report_data: Union[BaseModel, Dict[str, Any]]) -> Optional[str]:
    """
    Updates an existing citizen report document, nesting the AI output under `ai_analysis`.
    """
    try:
        if isinstance(report_data, BaseModel):
            analysis_data = report_data.model_dump()
        elif isinstance(report_data, dict):
            analysis_data = report_data
        else:
            raise TypeError(f"Unsupported report_data type: {type(report_data).__name__}")

        doc_ref = db.collection(collection_name).document(doc_id)
        write_result = doc_ref.update({"ai_analysis": analysis_data})

        print(f"   Updated Firestore doc: {doc_id}")
        print(f"   Committed  : {write_result.update_time}")
        return doc_ref.id
    except Exception as e:
        print(f"Error writing to Firestore: {e}")
        return None

def get_report_from_firestore(collection_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches an existing citizen report and returns its photo_url and user_text.
    """
    try:
        doc_ref = db.collection(collection_name).document(doc_id)
        doc = doc_ref.get()

        if not doc.exists:
            print(f"Error: No document found with ID {doc_id}")
            return None

        data = doc.to_dict()
        if data is None:
            print(f"Error: Document {doc_id} has no data")
            return None
        photo_url = data.get("photo_url")
        user_text = data.get("user_text", "")

        if not photo_url:
            print(f"Error: Document {doc_id} has no photo_url")
            return None

        return {"photo_url": photo_url, "user_text": user_text}

    except Exception as e:
        print(f"Error fetching report from Firestore: {e}")
        return None

def download_image(photo_url: str) -> Optional[bytes]:
    """
    Downloads image bytes from a Cloudinary (or any public) URL.
    """
    try:
        resp = requests.get(photo_url, timeout=10)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"Error downloading image from {photo_url}: {e}")
        return None

# # --- Example Usage for Testing ---
if __name__ == "__main__":
    # Test with a dummy image and text from firestore and cloudinary
    collection_name = "citizen_reports"  # Firestore collection for citizen reports
    sample_doc_id = "2FfEgs63sJd0mY7L9mKg"
    doc_ref = db.collection(collection_name).document(sample_doc_id)
    doc = doc_ref.get()

    data = doc.to_dict()
    if data is None:
        print(f"Error: Document {sample_doc_id} has no data")
        exit(1)

    sample_text = data.get("description", "")

    sample_image_url_cloudinary = data.get("image_url")
    if not sample_image_url_cloudinary:
        print(f"Error: Document {sample_doc_id} has no image_url")
        exit(1)
        
    sample_image = download_image(sample_image_url_cloudinary)
    
    if sample_image is not None:
        result_json = analyze_citizen_report(sample_image, sample_text)
        print (f"Gemini Analysis Result: {result_json}")
        if result_json:
            save_to_firestore(collection_name, sample_doc_id, result_json.model_dump())
    else:
        print(f"Please place a sample image named '{sample_image}' in your directory to test!")

def process_and_save_report(doc_id: str, collection_name: str) -> Dict[str, Any]:
    report_data = get_report_from_firestore(collection_name, doc_id)
    if not report_data:
        return {"status": "error", "message": "Failed to fetch report from Firestore"}

    image_bytes = download_image(report_data["photo_url"])
    if not image_bytes:
        return {"status": "error", "message": "Failed to download image"}

    analysis_result = analyze_citizen_report(image_bytes, report_data["user_text"])
    if not analysis_result:
        return {"status": "error", "message": "Failed to analyze image with Gemini"}

    updated_id = save_to_firestore(collection_name, doc_id, analysis_result)

    return {
        "status": "success",
        "document_id": updated_id,
        "data": {"ai_analysis": analysis_result.model_dump()}
    }


