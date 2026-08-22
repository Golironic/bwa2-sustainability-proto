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


# 1. Initialize Gemini Client (Make sure GEMINI_API_KEY is in your environment variables)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 2. Initialize Firebase Firestore (Make sure your service account JSON file is downloaded)
cred = credentials.Certificate("C:\\ZShikhar\\Build with AI\\IMPORTANT temp\\firebase-key-parth.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Define the exact output structure required for the shared Firestore schema
class CitizenReportSchema(BaseModel):
    category: str
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

def save_to_firestore(collection_name: str, report_data: Union[BaseModel, Dict[str, Any]]) -> Optional[str]:
    """
    Writes the processed AI results into Firestore using the shared schema field names.
    Accepts either a Pydantic model or a plain dict.
    """
    try:
        if isinstance(report_data, BaseModel):
            doc_data = report_data.model_dump()
        elif isinstance(report_data, dict):
            doc_data = report_data
        else:
            raise TypeError(f"Unsupported report_data type: {type(report_data).__name__}")

        # doc_data["created_at"] = firestore.SERVER_TIMESTAMP

        doc_ref = db.collection(collection_name).document()
        write_result = doc_ref.set(doc_data)

        print(f"   Successfully wrote to Firestore!")
        print(f"   Collection : {collection_name}")
        print(f"   Document ID: {doc_ref.id}")
        print(f"   Committed  : {write_result.update_time}")
        return doc_ref.id
    except Exception as e:
        print(f"Error writing to Firestore: {e}")

# # --- Example Usage for Testing ---
# if __name__ == "__main__":
#     # Test with a dummy image and text locally
#     # (Drop a sample image named 'test_small_smoke.jpg' in your project folder to test)
#     sample_image = "test_small_smoke.jpg" 
#     sample_text = "Lots of thick smoke coming from a pile of leaves and plastic near the park."
    
#     if os.path.exists(sample_image):
#         result_json = analyze_citizen_report(sample_image, sample_text)
#         print (f"Gemini Analysis Result: {result_json}")
#         # Uncomment below once Firestore is initialized
#         if result_json:
#             save_to_firestore("citizen-reports", result_json.model_dump())
#     else:
        # print(f"Please place a sample image named '{sample_image}' in your directory to test!")

def process_and_save_report(image_input: Union[str, bytes], user_text: str = "", collection_name: str = "citizen_reports") -> Dict[str, Any]:
    """
    Main entry point for Backend Developer:
    Runs Gemini analysis AND writes directly to Firestore in one call.
    """
    analysis_result = analyze_citizen_report(image_input, user_text)
    
    if not analysis_result:
        return {"status": "error", "message": "Failed to analyze image with Gemini"}

    doc_id = save_to_firestore(collection_name, analysis_result)

    return {
        "status": "success",
        "document_id": doc_id,
        "data": analysis_result.model_dump()
}

# not usable yet, but can be used in future for voice notes if pricing allows.
"""
# class VoiceReportSchema(BaseModel):
#     transcription: str
#     category: str
#     severity: int

# def process_voice_note(audio_path: str) -> VoiceReportSchema:
#     ""
#     Sends an audio voice note directly to Gemini. 
#     It automatically transcribes multi-language audio and categorizes it.
#     ""
#     # Upload audio file using the Files API (recommended for audio/video files)
#     print(f"Uploading audio file: {audio_path}...")
#     audio_file = client.files.upload(file=audio_path)

#     prompt = ""
#     Listen to this citizen voice note regarding an environmental issue.
#     1. Transcribe what the citizen said word-for-word,and translate it if necessary into English.
#     2. Categorize the issue (e.g., 'garbage_burning', 'smog', 'illegal_dumping').
#     3. Rate the severity from 1 to 5.
#     ""

#     response = client.models.generate_content(
#         model='gemini-3.6-flash',
#         contents=[audio_file, prompt],
#         config=types.GenerateContentConfig(
#             response_mime_type="application/json",
#             response_schema=VoiceReportSchema,
#             max_output_tokens=2000,
#         ),
#     )

#     parsed = response.parsed
#     if not isinstance(parsed, VoiceReportSchema):
#         raise TypeError(f"Unexpected parsed response type: {type(parsed).__name__}")
#     return parsed

# Test it out
# if __name__ == "__main__":
#     # Drop a sample voice file named 'test_audio.mp3' in your folder to test
#     audio_sample = "test_audio.mp3"
#     if os.path.exists(audio_sample):
#         result = process_voice_note(audio_sample)
#         print(f"Transcription: {result.transcription}")
#         print(f"Category: {result.category}")
#         print(f"Severity: {result.severity}/5")
#     else:
#         print(f"Place an audio file named '{audio_sample}' in your directory to test!") 
finally:
    if audio_file:
        genai_client.files.delete(name=audio_file.name)
"""

