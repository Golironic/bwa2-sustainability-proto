import os
from urllib import response
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
import json


# 1. Initialize Gemini Client (Make sure GEMINI_API_KEY is in your environment variables)
# Alternatively, pass api_key="YOUR_KEY" directly.
#api_key=os.environ.get("GEMINI_API_KEY")
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

def analyze_citizen_report(image_path: str, user_text: str):
    """
    Sends a citizen photo and optional text to Gemini Multimodal, 
    enforcing a structured JSON output matching the schema.
    """
    try:
        # Load the image using Pillow
        img = Image.open(image_path)
        img.thumbnail((100, 100))
        #TODO: Increase resolution to 800x800 (or less) before or after opening pull request. Currently reduced to save tokens for testing.

        
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
        print("AI Analysis Result:", response.text)
        return response.text

    except Exception as e:
        print(f"Error processing report with Gemini: {e}")
        return None

def save_to_firestore(report_data: dict):
    """
    Writes the processed AI results into Firestore using the shared schema field names.
    """
    try:
        doc_ref = db.collection('citizen_reports').document()
        doc_ref.set(report_data)
        print("Successfully wrote report data to Firestore!")
    except Exception as e:
        print(f"Error writing to Firestore: {e}")

# --- Example Usage for Testing ---
# if __name__ == "__main__":
#     # Test with a dummy image and text locally
#     # (Drop a sample image named 'test_small_smoke.jpg' in your project folder to test)
#     sample_image = "test_small_smoke.jpg" 
#     sample_text = "Lots of thick smoke coming from a pile of leaves and plastic near the park."
    
#     if os.path.exists(sample_image):
#         result_json = analyze_citizen_report(sample_image, sample_text)
#         # Uncomment below once Firestore is initialized
#         # if result_json:
#         #     save_to_firestore(json.loads(result_json))
#     else:
#         print(f"Please place a sample image named '{sample_image}' in your directory to test!")

# # Define the strict structure you want back
# class CitizenReport(BaseModel):
#     category: str = Field(description="The type of environmental issue, e.g., 'garbage_burning', 'illegal_dumping', 'smoke'")
#     severity: int = Field(description="Severity score from 1 (minor) to 5 (critical/hazardous)")
#     description: str = Field(description="A concise summary description of the issue observed in the photo")

# # working(tested on multiple images present in the folder)
# def analyze_photo(image_path: str) -> CitizenReport:
#     #"""Sends a citizen photo to Gemini and returns structured category, severity, and description."""
    
#     # Optional: Resize image to save tokens before sending
#     img = Image.open(image_path)
#     img.thumbnail((100, 100)) # Keeps resolution high enough for AI, low enough to save tokens
#     #TODO: Increase resolution to 800x800 (or less) before or after opening pull request. Currently reduced to save tokens for testing.
    
#     prompt = """
#     Analyze this citizen-submitted environmental photo. 
#     Classify the issue, rate its severity from 1 to 5, and write a brief description.
#     """

#     response = client.models.generate_content(
#         model='gemini-3.6-flash',
#         contents=[img, prompt],
#         config=types.GenerateContentConfig(
#             response_mime_type="application/json", #makes it so that the response is structured JSON
#             response_schema=CitizenReport, #schema(database) for the structured output
#             max_output_tokens=1000, # Saves tokens
#         ),
#     )
    
#     # response.parsed will automatically be structured based on your Pydantic class!
#     parsed = response.parsed
#     if not isinstance(parsed, CitizenReport):
#         raise TypeError(f"Unexpected parsed response type: {type(parsed).__name__}")
#     return parsed

class VoiceReportSchema(BaseModel):
    transcription: str
    category: str
    severity: int

def process_voice_note(audio_path: str) -> VoiceReportSchema:
    """
    Sends an audio voice note directly to Gemini. 
    It automatically transcribes multi-language audio and categorizes it.
    """
    # Upload audio file using the Files API (recommended for audio/video files)
    print(f"Uploading audio file: {audio_path}...")
    audio_file = client.files.upload(file=audio_path)

    prompt = """
    Listen to this citizen voice note regarding an environmental issue.
    1. Transcribe what the citizen said word-for-word,and translate it if necessary into English.
    2. Categorize the issue (e.g., 'garbage_burning', 'smog', 'illegal_dumping').
    3. Rate the severity from 1 to 5.
    """

    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=[audio_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VoiceReportSchema,
            max_output_tokens=2000,
        ),
    )

    parsed = response.parsed
    if not isinstance(parsed, VoiceReportSchema):
        raise TypeError(f"Unexpected parsed response type: {type(parsed).__name__}")
    return parsed

# Test it out
if __name__ == "__main__":
    # Drop a sample voice file named 'test_audio.mp3' in your folder to test
    audio_sample = "test_audio.mp3"
    if os.path.exists(audio_sample):
        result = process_voice_note(audio_sample)
        print(f"Transcription: {result.transcription}")
        print(f"Category: {result.category}")
        print(f"Severity: {result.severity}/5")
    else:
        print(f"Place an audio file named '{audio_sample}' in your directory to test!")