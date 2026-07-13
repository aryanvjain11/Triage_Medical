from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
from dotvenv import load_dotenv
import os
import json
import re
import webbrowser
from threading import Timer
from datetime import datetime
a=1
load_dotenv()

app = Flask(__name__)
 
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize Groq Client safely
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("[WARN] GROQ_API_KEY environment variable not found. Engaging Vital-Linked Local Fallback Engine.")
    client = None
else:
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

DB_FILE = "triage_database.json"

LOCAL_MED_TRANSLATIONS = {
    "es": {
        "pecho": "chest pain", "respirar": "difficulty breathing", "sangre": "bleeding", "cabeza": "head trauma",
        "dolor": "pain", "corazon": "heart", "fiebre": "fever", "estomago": "abdominal pain", "mareo": "dizziness"
    },
    "hi": {
        "pain": "pain", "saans": "dyspnea", "seene": "chest pain", "khoon": "bleeding", "ch चक्कर": "dizziness",
        "bukhar": "fever", "sir": "headache"
    }
}

def log_to_local_database(record):
    try:
        if os.path.exists(DB_FILE):
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(record)
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[DB ERROR] Failed writing to local registry: {e}")

def local_fallback_engine(user_text, vitals, allergies="", medications=""):
    text_lower = user_text.lower()
    detected_langs = ["es" if any(k in text_lower for k in ["dolor", "pecho", "respira"]) else "en"]
    lang = detected_langs[0]
    
    translated_terms = []
    if lang in LOCAL_MED_TRANSLATIONS:
        for key, eng in LOCAL_MED_TRANSLATIONS[lang].items():
            if key in text_lower:
                translated_terms.append(eng)
    
    translation_str = f"Patient presents with: {', '.join(translated_terms) if translated_terms else user_text}"
    if allergies: translation_str += f" | Known Allergies: {allergies}"
    if medications: translation_str += f" | Active Medications: {medications}"
    
    # --- Deep Vitals Clinical Evaluation Extraction ---
    try: hr = float(vitals.get('hr', 75) or 75)
    except ValueError: hr = 75

    try: spo2 = float(vitals.get('spo2', 98) or 98)
    except ValueError: spo2 = 98

    try: temp = float(vitals.get('temp', 98.6) or 98.6)
    except ValueError: temp = 98.6

    systolic = 120
    bp_string = str(vitals.get('bp', '120/80'))
    bp_digits = re.findall(r'\d+', bp_string)
    if bp_digits:
        systolic = float(bp_digits[0])

    severity = "MEDIUM"
    urgency_score = 3
    red_flags = []
    category = "General Medical Outpatient Clinic"

    # ESI Core Algorithm Threshold Rules
    if spo2 < 92 or systolic < 85 or systolic > 210 or hr > 130:
        severity = "CRITICAL"
        urgency_score = 1
        category = "Emergency Resuscitation Unit"
        if spo2 < 92: red_flags.append("Severe Hypoxemia Vector")
        if systolic < 85: red_flags.append("Hemodynamic Shock Limit Reached")
        if systolic > 210: red_flags.append("Malignant Hypertensive State")
        if hr > 130: red_flags.append("Extreme Unstable Tachycardia")
    elif spo2 <= 94 or hr > 110 or hr < 50 or temp >= 103.0 or temp <= 95.0 or systolic > 160 or systolic < 90:
        severity = "HIGH"
        urgency_score = 2
        category = "Acute Medical Intervention Zone"
        if temp >= 103.0: red_flags.append("High-Grade Pyrexia / Potential Sepsis Pathway")
        if temp <= 95.0: red_flags.append("Hypothermia Warning Core")
        if systolic > 160 or systolic < 90: red_flags.append("Aberrant Blood Pressure Range")
        if hr > 110 or hr < 50: red_flags.append("Cardiac Rate Drift Warning")

    if ("chest" in translation_str.lower() or "breath" in translation_str.lower()) and urgency_score > 2:
        severity = "HIGH"
        urgency_score = 2
        category = "Cardiology Rapid Triage Unit"

    return {
        "translation": f"[Vitals-Linked Local Fallback] {translation_str}",
        "native_audio_script": "Hemos recibido sus signos vitales y datos clínicos con éxito." if lang == "es" else "Intake parameters integrated successfully into local records.",
        "symptoms": translated_terms if translated_terms else ["Unspecified Symptom Cluster"],
        "body_part": "Cardiopulmonary System" if urgency_score <= 2 else "General Evaluation Needed",
        "pain_level": "Severe" if urgency_score == 1 else "Moderate",
        "duration": "Acute Presentation",
        "severity": severity,
        "urgency_score": urgency_score,
        "category": category,
        "red_flags": red_flags,
        "confidence": 0.75,
        "explanation": f"Calculated safely via strict local rule boundaries. Evaluated fields: HR={hr}, BP Systolic={systolic}, SpO2={spo2}%, Temp={temp}°F."
    }

def build_prompt(user_text, language, vitals, allergies, medications):
    return f"""
You are an expert Clinical Emergency Department ESI Triage Analyst.
Patient Input Language: {language}
Allergies: {allergies} | Current Medications: {medications}

MANDATORY CRITICAL ASSESSMENT VECTORS (Vital Parameters):
- Heart Rate (HR): {vitals.get('hr')} bpm
- Blood Pressure (BP): {vitals.get('bp')} mmHg
- Oxygen Saturation (SpO2): {vitals.get('spo2')}%
- Temperature (Temp): {vitals.get('temp')}°F

ESI SCORE ASSIGNMENT CRITERIA MANDATE:
1. LEVEL 1 (CRITICAL): If SpO2 < 90%, Systolic BP < 85 mmHg, HR > 130 bpm, or systemic unconsciousness/arrest signs are present.
2. LEVEL 2 (HIGH): If SpO2 is 90-94%, Systolic BP > 180 mmHg or < 90 mmHg, Temp >= 103.0°F or <= 95.0°F, HR > 110 bpm or < 50 bpm, or if severe sudden chest pain or respiratory distress is declared.
3. LEVEL 3 (MEDIUM): Hemodynamically stable parameters but requires multi-resource diagnostic screening.

DIRECTIVE: Translate the user text to medical English prose inside the "translation" field. Do not copy or echo the original foreign text into English fields.

Return ONLY a valid JSON object matching this schema precisely:
{{
  "translation": "Objective medical English translation of the history of present illness",
  "native_audio_script": "Short, reassuring summary in the patient's native language explaining their steps.",
  "symptoms": ["Symptom1", "Symptom2"],
  "body_part": "Anatomical zone",
  "pain_level": "Mild/Moderate/Severe",
  "duration": "Timeline profile",
  "severity": "CRITICAL/HIGH/MEDIUM/LOW",
  "urgency_score": 1,
  "category": "Destination Medical Department",
  "red_flags": ["Specific life safety concerns, vital irregularities, or allergy alerts"],
  "confidence": 0.98,
  "explanation": "Explicit clinical rationale showing how both symptoms and specific vital boundaries directly influenced the selected ESI classification level"
}}
User Text: "{user_text}"
"""

@app.route("/")
def index():
    try:
        with open("Index.html", "r", encoding="utf-8") as f:
            return render_template_string(f.read())
    except FileNotFoundError:
        return "Index.html file not found in execution directory.", 404

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    data = request.json or {}
    user_text = data.get("text", "")
    language = data.get("language", "English")
    vitals = data.get("vitals", {"hr": "", "bp": "", "spo2": "", "temp": ""})
    allergies = data.get("allergies", "")
    medications = data.get("medications", "")

    if not client:
        parsed = local_fallback_engine(user_text, vitals, allergies, medications)
    else:
        prompt = build_prompt(user_text, language, vitals, allergies, medications)
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            parsed = json.loads(resp.choices[0].message.content)
        except Exception:
            parsed = local_fallback_engine(user_text, vitals, allergies, medications)

    db_record = {
        "timestamp": datetime.now().isoformat(),
        "input_language": language,
        "original_text": user_text,
        "vitals": vitals,
        "allergies": allergies,
        "medications": medications,
        "processed_output": parsed
    }
    log_to_local_database(db_record)
    return jsonify(parsed)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    msg = data.get("message", "")
    language = data.get("language", "English")
    
    if not client:
        if language == "Spanish":
            return jsonify({"reply": "Entendido. Estoy aquí para asistirlo con preguntas de la sala de espera."})
        elif language == "Hindi":
            return jsonify({"reply": "मैं समझता हूँ। प्रतीक्षा करते समय सामान्य सहायता के लिए मैं यहाँ हूँ।"})
        return jsonify({"reply": "Understood. I am here to help answer waiting room logistics inquiries while you wait."})
        
    chat_prompt = f"""
    You are a friendly, non-medical lobby hospitality receptionist assistant.
    Current Selected Language Context: {language}

    CRITICAL DISCIPLINE INSTRUCTIONS:
    - You must respond entirely in the requested language: {language}.
    - Keep your answer short, warm, supportive, and administrative (1-3 sentences maximum).
    - If the user tries to seek direct medical advice, diagnosis, treatment help, or report their main physical complications in this chat box (e.g., asking how to treat a headache, chest pains, or cuts), you MUST politely explain that you are an administrative desk assistant and cannot give medical guidance. Instruct them to type or speak their clinical concerns into the 'Chief Medical Concerns & Symptoms' terminal panel on the left side of the dashboard instead so the clinical engine can safely analyze their vital limits.
    
    Patient Message: "{msg}"
    """
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": chat_prompt}],
            temperature=0.4,
            max_tokens=250,
        )
        reply = resp.choices[0].message.content.strip()
        return jsonify({"reply": reply})
    except Exception:
        return jsonify({"reply": "Hospitality desk processing buffer busy. Please standby."})

if __name__ == "__main__":
    Timer(1.0, lambda: webbrowser.open_new("http://127.0.0.1:5000/")).start()
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))