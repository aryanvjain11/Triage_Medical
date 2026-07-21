from flask import Flask, request, jsonify, render_template_string
from groq import Groq
from dotenv import load_dotenv
import os
import json
import re
import random
import webbrowser
from threading import Timer
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)

INDEX2_FILE = BASE_DIR / "Index2.html"
LANDING_FILE = BASE_DIR / "landing.html"
DB_FILE = BASE_DIR / "triage_database.json"

CLINICAL_KEYWORD_MAP = {
    "chest": ["chest pain", "chest pressure", "thoracic discomfort", "cardiac pressure", "heart pain"],
    "breath": ["dyspnea", "shortness of breath", "difficulty breathing", "trouble breathing", "respiratory distress", "breathlessness"],
    "bleed": ["bleeding", "hemorrhage", "blood loss", "heavy bleeding"],
    "head": ["head trauma", "head injury", "headache", "concussion"],
    "pain": ["pain", "ache", "discomfort", "soreness"],
    "fever": ["fever", "pyrexia", "temperature"],
    "faint": ["syncope", "fainting", "dizziness", "lightheadedness"],
    "vomit": ["vomiting", "nausea", "throwing up"],
    "seizure": ["seizure", "convulsion", "fit"],
    "wheeze": ["wheezing", "asthma", "wheeze"],
    "stroke": ["stroke", "facial droop", "slurred speech", "weakness", "facial drooping"],
}


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# Initialize Native Groq Client safely
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

if not groq_client:
    print("[WARN] No Groq credentials found. Engaging Vital-Linked Local Fallback Engine.")

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
        if DB_FILE.exists():
            with DB_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(record)
        with DB_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[DB ERROR] Failed writing to local registry: {e}")


def extract_clinical_keywords(user_text, vitals=None):
    text_lower = (user_text or "").lower()
    matched = []

    for token, variants in CLINICAL_KEYWORD_MAP.items():
        if any(variant in text_lower for variant in variants):
            matched.extend(variants)

    if not matched:
        for token, variants in CLINICAL_KEYWORD_MAP.items():
            if token in text_lower:
                matched.extend(variants)

    if vitals:
        try:
            hr = float(vitals.get('hr', 75) or 75)
        except ValueError:
            hr = 75
        if hr > 130:
            matched.append("tachycardia")
        if hr < 50:
            matched.append("bradycardia")

        try:
            spo2 = float(vitals.get('spo2', 98) or 98)
        except ValueError:
            spo2 = 98
        if spo2 < 92:
            matched.append("hypoxemia")
        if spo2 <= 94:
            matched.append("oxygen desaturation")

    return sorted(set(matched))[:8]


def infer_symptom_terms(user_text):
    text = (user_text or "").strip()
    if not text:
        return ["reported symptoms"]

    lowered = text.lower()
    if any(phrase in lowered for phrase in ["difficulty breathing", "shortness of breath", "trouble breathing"]):
        return ["difficulty breathing"]
    if any(phrase in lowered for phrase in ["chest pain", "chest pressure"]):
        return ["chest pain"]
    if any(phrase in lowered for phrase in ["bleeding", "heavy bleeding"]):
        return ["bleeding"]
    if any(phrase in lowered for phrase in ["headache", "head injury", "head trauma"]):
        return ["headache"]
    if any(phrase in lowered for phrase in ["dizziness", "fainting", "lightheadedness"]):
        return ["dizziness"]

    tokens = re.findall(r"[a-zA-Z]{3,}", text)
    return [tokens[0]] if tokens else ["reported symptoms"]


def run_model_completion(model_name, prompt_content, temperature=0.3, max_tokens=300):
    if not groq_client:
        raise ValueError("Groq client is uninitialized. Verify your GROQ_API_KEY.")
        
    clean_model = model_name
    if "groq/" in model_name:
        clean_model = model_name.split("/")[-1]

    completion = groq_client.chat.completions.create(
        model=clean_model,
        messages=[{"role": "user", "content": prompt_content}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return completion


def local_fallback_engine(user_text, vitals, allergies="", medications=""):
    text_lower = user_text.lower()
    detected_langs = ["es" if any(k in text_lower for k in ["dolor", "pecho", "respira"]) else "en"]
    lang = detected_langs[0]
    clinical_keywords = extract_clinical_keywords(user_text, vitals)

    translated_terms = []
    if lang in LOCAL_MED_TRANSLATIONS:
        for key, eng in LOCAL_MED_TRANSLATIONS[lang].items():
            if key in text_lower:
                translated_terms.append(eng)

    translation_str = f"Patient presents with: {', '.join(translated_terms) if translated_terms else user_text}"
    if clinical_keywords:
        translation_str += f" | Clinical keywords: {', '.join(clinical_keywords)}"
    if allergies:
        translation_str += f" | Known Allergies: {allergies}"
    if medications:
        translation_str += f" | Active Medications: {medications}"

    try:
        hr = float(vitals.get('hr', 75) or 75)
    except ValueError:
        hr = 75

    try:
        spo2 = float(vitals.get('spo2', 98) or 98)
    except ValueError:
        spo2 = 98

    try:
        temp = float(vitals.get('temp', 98.6) or 98.6)
    except ValueError:
        temp = 98.6

    systolic = 120
    bp_string = str(vitals.get('bp', '120/80'))
    bp_digits = re.findall(r'\d+', bp_string)
    if bp_digits:
        systolic = float(bp_digits[0])

    severity = "MEDIUM"
    urgency_score = 3
    red_flags = []
    category = "General Medical Outpatient Clinic"

    if spo2 < 92 or systolic < 85 or systolic > 210 or hr > 130:
        severity = "CRITICAL"
        urgency_score = 1
        category = "Emergency Resuscitation Unit"
        if spo2 < 92:
            red_flags.append("Severe Hypoxemia Vector")
        if systolic < 85:
            red_flags.append("Hemodynamic Shock Limit Reached")
        if systolic > 210:
            red_flags.append("Malignant Hypertensive State")
        if hr > 130:
            red_flags.append("Extreme Unstable Tachycardia")
    elif spo2 <= 94 or hr > 110 or hr < 50 or temp >= 103.0 or temp <= 95.0 or systolic > 160 or systolic < 90:
        severity = "HIGH"
        urgency_score = 2
        category = "Acute Medical Intervention Zone"
        if temp >= 103.0:
            red_flags.append("High-Grade Pyrexia / Potential Sepsis Pathway")
        if temp <= 95.0:
            red_flags.append("Hypothermia Warning Core")
        if systolic > 160 or systolic < 90:
            red_flags.append("Aberrant Blood Pressure Range")
        if hr > 110 or hr < 50:
            red_flags.append("Cardiac Rate Drift Warning")

    if any(token in translation_str.lower() for token in ["chest", "breath", "dyspnea", "cardiac", "respiratory", "hypoxemia"]):
        severity = "HIGH"
        urgency_score = 2
        category = "Cardiology Rapid Triage Unit"

    symptom_candidates = translated_terms or [kw.replace("_", " ") for kw in clinical_keywords[:3]] or infer_symptom_terms(user_text)

    explanation_text = f"Calculated safely via strict local rule boundaries. Evaluated fields: HR={hr}, BP Systolic={systolic}, SpO2={spo2}%, Temp={temp}°F. Keyword cues detected: {', '.join(clinical_keywords) if clinical_keywords else 'none'}."

    return {
        "translation": f"[Vitals-Linked Local Fallback] {translation_str}",
        "native_audio_script": explanation_text,
        "symptoms": symptom_candidates,
        "body_part": "Cardiopulmonary System" if urgency_score <= 2 else "General Evaluation Needed",
        "pain_level": "Severe" if urgency_score == 1 else "Moderate",
        "duration": "Acute Presentation",
        "severity": severity,
        "urgency_score": urgency_score,
        "category": category,
        "red_flags": red_flags,
        "confidence": 0.75,
        "explanation": explanation_text
    }


def build_prompt(user_text, language, vitals, allergies, medications):
    clinical_keywords = extract_clinical_keywords(user_text, vitals)
    keyword_line = f"Clinical keyword cues: {', '.join(clinical_keywords) if clinical_keywords else 'none'}"
    return f"""
You are an expert Clinical Emergency Department ESI Triage Analyst.
Patient Input Language: {language}
Allergies: {allergies} | Current Medications: {medications}
{keyword_line}

MANDATORY CRITICAL ASSESSMENT VECTORS (Vital Parameters):
- Heart Rate (HR): {vitals.get('hr')} bpm
- Blood Pressure (BP): {vitals.get('bp')} mmHg
- Oxygen Saturation (SpO2): {vitals.get('spo2')}%
- Temperature (Temp): {vitals.get('temp')}°F

ESI SCORE ASSIGNMENT CRITERIA MANDATE:
1. LEVEL 1 (CRITICAL): If SpO2 < 90%, Systolic BP < 85 mmHg, HR > 130 bpm, or systemic unconsciousness/arrest signs are present.
2. LEVEL 2 (HIGH): If SpO2 is 90-94%, Systolic BP > 180 mmHg or < 90 mmHg, Temp >= 103.0°F or <= 95.0°F, HR > 110 bpm or < 50 bpm, or if severe sudden chest pain or respiratory distress is declared.
3. LEVEL 3 (MEDIUM): Hemodynamically stable parameters but requires multi-resource diagnostic screening.

DIRECTIVE: Translate the user text to medical English prose inside the "translation" field. Do not copy or echo the original foreign text into English fields. Ensure "native_audio_script" contains the complete clinical rationale explanation so that it can be read aloud directly via text-to-speech.

Return ONLY a valid JSON object matching this schema precisely:
{{
  "translation": "Objective medical English translation of the history of present illness",
  "native_audio_script": "Explicit clinical rationale explaining how symptoms and vital boundaries determined this ESI level, written in plain text for speech output.",
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


def build_local_chat_reply(message, language):
    text = (message or "").strip().lower()
    if any(k in text for k in ["clean", "wound", "cut", "bandage", "bleed", "injur"]):
        replies = {
            "Spanish": ["Para un corte pequeño, lave la herida con agua y jabón suave, aplique presión limpia si sigue sangrando y cubra con apósito estéril. Si hay mucha sangre, dolor intenso, signos de infección o una herida profunda, busque atención médica urgente.", "Si la herida es profunda, muy sucia, no cierra bien o está en la cara, consulte a un profesional de salud."],
            "Hindi": ["छोटे कट के लिए, घाव को साफ पानी और नरम साबुन से धोएँ, यदि खून बहता रहे तो साफ दबाव डालें और स्टरल बाँडेज से ढक दें। यदि बहुत खून बह रहा है, दर्द बहुत तेज है, या घाव गहरा है तो तुरंत चिकित्सा सहायता लें।", "यदि घाव गहरा, बहुत गंदा, या चेहरे पर है तो स्वास्थ्य पेशेवर से सलाह लें।"],
            "French": ["Pour une petite coupure, lavez la plaie avec de l'eau et du savon doux, appliquez une pression propre si elle saigne encore, puis couvrez-la avec un pansement stérile. Si le saignement est important, si la douleur est forte ou si la plaie est profonde, consultez un professionnel de santé rapidement.", "Si la plaie est profonde, très sale, ou sur le visage, obtenez une évaluation médicale."],
            "Mandarin": ["对于较小的伤口，先用清水和温和肥皂清洗，若仍有出血可施加干净的压力并覆盖无菌纱布。如果出血较多、疼痛剧烈、伤口较深，建议尽快就医。", "若伤口较深、很脏或位于面部，请尽快寻求医疗帮助。"],
        }
        return random.choice(replies.get(language, ["For a small cut, rinse it with clean water, apply gentle pressure if needed, and cover it with a clean bandage. Seek urgent care if it is deep, bleeding heavily, or shows signs of infection.", "If the wound is deep, dirty, or on the face, seek medical evaluation promptly."]))

    if any(k in text for k in ["wait", "time", "delay", "how long"]):
        replies = {
            "Spanish": ["Entiendo su preocupación por la espera. El personal puede confirmar el tiempo estimado y la prioridad de su turno.", "Podemos revisar la información de espera y las opciones de la sala de espera para usted."],
            "Hindi": ["मैं समझ रहा हूँ कि आपको प्रतीक्षा के बारे में जानकारी चाहिए। स्टाफ आपकी प्रतीक्षा अवधि और प्रक्रियाओं के बारे में जानकारी दे सकता है।", "मैं आपकी प्रतीक्षा समय और सुविधाओं के बारे में मदद कर सकता हूँ।"],
            "French": ["Je peux vous aider à clarifier les délais et les services disponibles pendant l'attente.", "Votre temps d'attente et les commodités du service peuvent être confirmés par le personnel."],
            "Mandarin": ["我可以帮您了解等待时间和候诊室的服务安排。", "我可以为您说明当前等待情况以及可用设施。"],
        }
        return random.choice(replies.get(language, ["I can help explain wait times and the front-desk process.", "I can help with waiting-room logistics and available services."]))

    if any(k in text for k in ["amenit", "bathroom", "parking", "wifi", "coffee", "food"]):
        replies = {
            "Spanish": ["Hay comodidades disponibles en la sala de espera, y el personal puede orientarle si necesita ayuda específica.", "Podemos revisar las comodidades y servicios disponibles para su comodidad."],
            "Hindi": ["प्रतीक्षा कक्ष में सुविधाएँ उपलब्ध हैं, और स्टाफ आपकी सहायता कर सकता है।", "मैं आपकी सुविधाओं और उपलब्ध सेवाओं के बारे में जानकारी दे सकता हूँ।"],
            "French": ["Des commodités sont disponibles dans la salle d'attente, et le personnel puede vous guider.", "Je peux vous orienter vers les services et commodités disponibles."],
            "Mandarin": ["候诊室内有可用设施，工作人员也可以为您提供具体帮助。", "我可以为您说明当前可用的服务与设施。"],
        }
        return random.choice(replies.get(language, ["I can help with the waiting-room amenities and available services.", "I can point you to the facilities and support available right now."]))

    if any(k in text for k in ["pain", "chest", "breath", "bleed", "head", "faint", "vomit", "seizure", "stroke"]):
        replies = {
            "Spanish": ["Para preocupaciones clínicas, por favor describa sus síntomas en el panel de entrada principal para que el sistema pueda evaluarlos de forma segura.", "Su reporte clínico es importante; use el panel de síntomas para una evaluación de triaje segura."],
            "Hindi": ["क्लिनिकल चिंता के लिए कृपया अपने लक्षणों को मुख्य इनपुट क्षेत्र में लिखें ताकि सिस्टम उन्हें सुरक्षित रूप से मूल्यांकित कर सके।", "कृपया मुख्य लक्षण क्षेत्र में अपने संकेत दर्ज करें ताकि सुरक्षित ट्राइएज मूल्यांकन हो सके।"],
            "French": ["Pour des préoccupations cliniques, veuillez décrire vos symptômes dans le panneau principal afin que le système puisse les évaluer en sécurité.", "Votre rapport clinique peut être analysé en toute sécurité via le panneau de symptômes."],
            "Mandarin": ["如涉及临床症状，请在主输入面板中描述您的症状，以便系统安全地进行评估。", "请将临床症状输入主症状面板，系统将进行安全分诊评估。"],
        }
        return random.choice(replies.get(language, ["For clinical concerns, please enter your symptoms in the main intake panel so the triage engine can assess them safely.", "Please share your symptoms in the main intake panel for a safe clinical review."]))

    replies = {
        "Spanish": ["Gracias por su mensaje. Puedo ayudar con información general de la sala de espera y los próximos pasos.", "Estoy aquí para ayudar con preguntas sencillas sobre la espera, los servicios y el proceso de atención."],
        "Hindi": ["धन्यवाद। मैं प्रतीक्षा कक्ष, सेवाओं और प्रक्रिया के बारे में सामान्य सहायता दे सकता हूँ।", "मैं आपकी प्रतीक्षा, सेवाओं और आगे की प्रक्रियाओं के बारे में मदद कर सकता हूँ।"],
        "French": ["Merci pour votre message. Je peux aider avec les informations générales sur l'attente et les prochaines étapes.", "Je suis là pour répondre aux questions simples sur l'attente, les services et le parcours de soins."],
        "Mandarin": ["感谢您的留言。我可以协助您了解候诊室流程、服务信息和接下来需要做什么。", "我可以为您说明当前等待情况以及可用设施。"],
    }
    return random.choice(replies.get(language, ["Thank you for your message. I can help with general waiting-room questions and next steps.", "I am here to help with general questions about the waiting process and available services."]))


def inject_ui_shell(html_content):
    if "app2-ui-shell-style" in html_content:
        return html_content

    shell_html = """
    <style id="app2-ui-shell-style">
      .app2-shell-toggle{position:fixed;right:18px;bottom:18px;z-index:9999;background:#0b1d33;color:#fff;border:1px solid rgba(0,180,255,.3);padding:10px 12px;border-radius:999px;cursor:pointer;box-shadow:0 10px 30px rgba(0,0,0,.3);}
      .app2-shell-panel{position:fixed;right:18px;bottom:72px;z-index:9998;width:min(300px,calc(100vw - 32px));background:rgba(5,12,23,.94);border:1px solid rgba(0,180,255,.25);border-radius:16px;padding:16px;color:#eef7ff;backdrop-filter:blur(12px);box-shadow:0 20px 40px rgba(0,0,0,.35);display:none;}
      .app2-shell-panel.open{display:block;}
      .app2-shell-panel label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#76d7ff;margin-top:10px;margin-bottom:6px;}
      .app2-shell-panel select,.app2-shell-panel button{width:100%;padding:8px 10px;border-radius:8px;border:1px solid rgba(255,255,255,.12);background:#0f2138;color:#fff;margin-bottom:8px;}
      .app2-shell-panel button{cursor:pointer;background:linear-gradient(90deg,#00b4ff,#0f70ff);}
      .app2-shell-panel .hint{font-size:11px;color:#87a7c6;margin-top:6px;}
      body[data-theme='midnight']{--bg:#04080f;--surface:#0a1628;--surface2:#0d1f38;--surface3:#122b45;--text:#f2fbff;--muted:rgba(160,200,255,.5);}
      body[data-theme='ocean']{--bg:#06131e;--surface:#0b2230;--surface2:#103645;--surface3:#17475b;--text:#f2fbff;--muted:#9ec9db;}
      body[data-theme='forest']{--bg:#07130b;--surface:#0d2417;--surface2:#133b21;--surface3:#195130;--text:#f1ffef;--muted:#a7e0b5;}
      body[data-theme='alert']{--bg:#180606;--surface:#2a0a0a;--surface2:#3d1111;--surface3:#5b1717;--text:#fff4f4;--muted:#f0b4b4;}
      body[data-theme='midnight'] .app2-shell-toggle{background:#081222;}
      body[data-theme='ocean'] .app2-shell-toggle{background:#09334b;}
      body[data-theme='forest'] .app2-shell-toggle{background:#133b21;}
      body[data-theme='alert'] .app2-shell-toggle{background:#491111;}
    </style>
    <button class="app2-shell-toggle" id="app2-shell-toggle" type="button">⚙️ Controls</button>
    <div class="app2-shell-panel" id="app2-shell-panel">
      <div id="app2-shell-title" style="font-size:13px;font-weight:700;margin-bottom:10px;">Studio Controls</div>
      <label id="app2-theme-label" for="app2-theme-select">Visual Theme</label>
      <select id="app2-theme-select">
        <option value="midnight">Midnight</option>
        <option value="ocean">Ocean</option>
        <option value="forest">Forest</option>
        <option value="alert">High Alert</option>
      </select>
      <label id="app2-music-label" for="app2-music-select">Music Theme</label>
      <select id="app2-music-select">
        <option value="off">Off</option>
        <option value="calm">Calm</option>
        <option value="pulse">Pulse</option>
        <option value="focus">Focus</option>
      </select>
      <button id="app2-music-toggle" type="button">Play/Stop</button>
      <div id="app2-music-hint" class="hint">Visual themes adjust the dashboard palette and music themes add ambient soundscapes.</div>
    </div>
    """

    shell_script = """
    <script>
      (() => {
        const shellToggle = document.getElementById('app2-shell-toggle');
        const shellPanel = document.getElementById('app2-shell-panel');
        const themeSelect = document.getElementById('app2-theme-select');
        const musicSelect = document.getElementById('app2-music-select');
        const musicToggle = document.getElementById('app2-music-toggle');
        const root = document.body;
        const storedTheme = localStorage.getItem('app2-theme') || 'midnight';
        const storedMusic = localStorage.getItem('app2-music') || 'off';
        themeSelect.value = storedTheme;
        musicSelect.value = storedMusic;
        root.setAttribute('data-theme', storedTheme);
        shellToggle.addEventListener('click', () => shellPanel.classList.toggle('open'));

        const applyTheme = () => { const value = themeSelect.value; root.setAttribute('data-theme', value); localStorage.setItem('app2-theme', value); };
        themeSelect.addEventListener('change', applyTheme);

        let audioCtx = null;
        let gainNode = null;
        let oscillators = [];
        let musicTimer = null;

        const stopMusic = () => {
          if (musicTimer) clearInterval(musicTimer);
          oscillators.forEach((node) => { try { node.stop(); } catch (e) {} node.disconnect(); });
          oscillators = [];
          if (gainNode) {
            try { gainNode.gain.setValueAtTime(0.0001, audioCtx.currentTime); } catch (e) {}
            gainNode.disconnect();
          }
          if (audioCtx) {
            try { audioCtx.suspend().catch(() => {}); } catch (e) {}
          }
          gainNode = null; musicTimer = null;
        };

        const playMusic = async (mode) => {
          if (mode === 'off') { stopMusic(); return; }
          if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            gainNode = audioCtx.createGain();
            gainNode.connect(audioCtx.destination);
            gainNode.gain.value = 0.04;
          }
          await audioCtx.resume().catch(() => {});
          oscillators.forEach((node) => { try { node.stop(); } catch (e) {} node.disconnect(); });
          oscillators = [];

          const pattern = mode === 'calm' ? [220, 330, 440] : mode === 'pulse' ? [330, 440, 550] : [260, 392, 523];
          const now = audioCtx.currentTime;
          pattern.forEach((freq, idx) => {
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            const filter = audioCtx.createBiquadFilter();
            filter.type = 'sine';
            filter.frequency.value = freq;
            osc.type = mode === 'focus' ? 'triangle' : 'sine';
            osc.frequency.setValueAtTime(freq, now + idx * 0.05);
            gain.gain.setValueAtTime(0.0001, now + idx * 0.05);
            gain.gain.exponentialRampToValueAtTime(0.018, now + idx * 0.05 + 0.08);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + idx * 0.05 + 0.32);
            osc.connect(filter); filter.connect(gain); gain.connect(gainNode);
            osc.start(now + idx * 0.05);
            osc.stop(now + idx * 0.05 + 0.33);
            oscillators.push(osc);
          });

          let idx = 0;
          musicTimer = setInterval(() => {
            if (!audioCtx || !gainNode) return;
            const note = pattern[idx % pattern.length];
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            const filter = audioCtx.createBiquadFilter();
            filter.type = 'sine';
            filter.frequency.value = note;
            osc.type = mode === 'focus' ? 'triangle' : 'sine';
            osc.frequency.setValueAtTime(note, audioCtx.currentTime);
            gain.gain.setValueAtTime(0.0001, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.014, audioCtx.currentTime + 0.05);
            gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.2);
            osc.connect(filter); filter.connect(gain); gain.connect(gainNode);
            osc.start(); osc.stop(audioCtx.currentTime + 0.22);
            oscillators.push(osc);
            idx += 1;
          }, 900);
          return true;
        };

        musicSelect.addEventListener('change', async () => {
          localStorage.setItem('app2-music', musicSelect.value);
          if (musicToggle.dataset.active === 'true') {
            await playMusic(musicSelect.value);
          }
        });

        musicToggle.addEventListener('click', async () => {
          if (musicToggle.dataset.active === 'true') {
            stopMusic();
            musicToggle.dataset.active = 'false';
            musicToggle.textContent = 'Play/Stop';
          } else {
            await playMusic(musicSelect.value);
            musicToggle.dataset.active = 'true';
            musicToggle.textContent = 'Stop';
          }
        });

        if (storedMusic !== 'off') {
          musicSelect.value = storedMusic;
          musicToggle.dataset.active = 'true';
          musicToggle.textContent = 'Stop';
        }
      })();
    </script>
    """

    return html_content.replace("</body>", shell_html + shell_script + "</body>")


def serve_html(path):
    target_path = path
    if not target_path.exists():
        alt_lower = path.parent / path.name.lower()
        if alt_lower.exists():
            target_path = alt_lower

    try:
        with target_path.open("r", encoding="utf-8") as f:
            html_content = f.read()
        return render_template_string(inject_ui_shell(html_content))
    except FileNotFoundError:
        return f"HTML file '{path.name}' not found in execution directory.", 404


@app.route("/")
def index():
    return serve_html(LANDING_FILE)


@app.route("/Index2.html")
@app.route("/index2.html")
@app.route("/dashboard")
def dashboard():
    return serve_html(INDEX2_FILE)


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

    selected_model = (data.get("model") or "groq/llama-3.3-70b-versatile").strip()
    
    if not groq_client:
        parsed = local_fallback_engine(user_text, vitals, allergies, medications)
    else:
        prompt = build_prompt(user_text, language, vitals, allergies, medications)
        try:
            resp = run_model_completion(
                selected_model,
                prompt,
                temperature=0.0
            )
            parsed = json.loads(resp.choices[0].message.content)
        except Exception as err:
            print(f"[CRITICAL ANALYSIS ERROR]: {err}")
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

    selected_model = (data.get("model") or "groq/llama-3.3-70b-versatile").strip()
    if not groq_client:
        return jsonify({"reply": build_local_chat_reply(msg, language)})

    chat_prompt = f"""
    You are a helpful clinical support assistant for a hospital waiting area.
    Current Selected Language Context: {language}

    CRITICAL DISCIPLINE INSTRUCTIONS:
    - Respond entirely in the requested language: {language}.
    - Keep responses concise, practical, and supportive.
    - You may give general first-aid and wellness guidance, but you must not present yourself as a doctor or diagnose a condition.
    - If the user describes severe symptoms such as chest pain, trouble breathing, heavy bleeding, fainting, stroke-like symptoms, or severe trauma, advise them to seek urgent medical help immediately and suggest using the main triage intake panel for a structured assessment.
    - For minor wound care, provide simple safe steps such as cleaning, pressure, and covering the injury.
    - For non-medical questions, answer as a helpful receptionist.

    Patient Message: "{msg}"
    """
    try:
        resp = run_model_completion(
            selected_model,
            chat_prompt,
            temperature=0.4,
            max_tokens=250,
        )
        reply = resp.choices[0].message.content.strip()
        return jsonify({"reply": reply})
    except Exception as err:
        print(f"[CRITICAL CHAT ERROR]: {err}")
        return jsonify({"reply": build_local_chat_reply(msg, language)})


@app.route('/debug-api-test', methods=['GET'])
def debug_api_test():
    groq_key_exists = bool(os.getenv("GROQ_API_KEY"))
    visible_env_keys = list(os.environ.keys())
    
    diagnostic_info = {
        "groq_key_present_in_environment": groq_key_exists,
        "api_test_connection": "PENDING",
        "all_visible_environment_keys": visible_env_keys,
        "current_working_directory": str(Path.cwd()),
        "env_file_exists": os.path.exists(".env")
    }
    
    try:
        if not groq_key_exists:
            raise ValueError("GROQ_API_KEY is completely missing from the Render environment variables.")
            
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        test_resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Ping"}],
            max_tokens=5
        )
        diagnostic_info["api_test_connection"] = "SUCCESS"
        diagnostic_info["api_test_response"] = test_resp.choices[0].message.content
    except Exception as e:
        diagnostic_info["api_test_connection"] = "FAILED"
        diagnostic_info["error_logs"] = str(e)
        
    return jsonify(diagnostic_info)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))