# Voice Assisted Cognitive Agent for Visually Impaired

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv)
![YOLOv3](https://img.shields.io/badge/YOLO-v3--tiny-orange)
![Tesseract](https://img.shields.io/badge/Tesseract-5.x-blueviolet)

**Cognitive AI — Micro Project (E1CSC 25608)**
**Alliance University, Bengaluru**

</div>

---

## 📌 Project Overview

A speech-based cognitive agent that assists visually impaired users by:
- 🎙 **Listening** to voice commands via browser microphone
- 👁 **Identifying objects** using YOLOv3-tiny + laptop webcam
- 📖 **Reading text** from environment using Tesseract OCR
- 💬 **Answering queries** through keyword-based NLP
- 🔊 **Speaking responses** aloud via Text-to-Speech

### Cognitive Interaction Loop

```
Listen → Transcribe → Parse Intent → Dispatch Action → Respond
   ↑                                                        │
   └────────────────── Feedback Loop ─────────────────────-─┘
```

---

## 👨‍🏫 Project Details

| Field        | Details                                      |
|--------------|----------------------------------------------|
| Course       | Cognitive AI — Micro Project (E1CSC 25608)   |
| University   | Alliance University, Bengaluru               |
| Guided by    | Dr. Jayabhaduri R                            |
| Batch        | Batch - 6                                    |
| Team Members | Vishal BR, Janani Rindhiya C M, Nitish Varma D, Lokesh U, Gayathri G |

---

## 📁 Project Structure

```
voice-cognitive-agent/
│
├── voice_cognitive_agent_web/        # Web version (run this)
│   ├── app.py                        # Flask backend — main server
│   ├── requirements.txt              # Python dependencies
│   ├── templates/
│   │   └── index.html                # Accessible frontend UI
│   └── static/                       # Static assets
│
├── models/                           # AI model files
│   ├── deploy.prototxt               # MobileNet config
│   ├── coco.names                    # 80 COCO class labels
│   ├── yolov3-tiny.cfg               # YOLO config
│   └── DOWNLOAD_WEIGHTS.md          # Instructions to download weights
│
├── docs/
│   └── project_report.md            # Project report summary
│
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Layer         | Technology                        |
|---------------|-----------------------------------|
| Frontend      | HTML5, CSS3, JavaScript           |
| Voice Input   | Web Speech API (STT)              |
| Voice Output  | SpeechSynthesis API (TTS)         |
| Backend       | Python 3.11, Flask 3.0            |
| Object Det.   | YOLOv3-tiny via OpenCV DNN        |
| OCR           | Tesseract 5 (LSTM engine)         |
| NLP           | Keyword-based intent classifier   |
| Webcam        | OpenCV cv2.VideoCapture(0)        |

---

## 🎯 Intent Categories

| Intent            | Keywords Detected                               |
|-------------------|-------------------------------------------------|
| `read_text`       | "read", "scan", "ocr", "what does it say"       |
| `identify_object` | "what is this", "identify", "what do you see"   |
| `time_date`       | "time", "date", "today", "clock"                |
| `help`            | "help", "assist", "what can you do"             |
| `stop`            | "stop", "exit", "quit", "bye"                   |
| `answer_query`    | "what", "who", "where", "how", "why"            |

---

## ⚙️ Setup & Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/voice-cognitive-agent.git
cd voice-cognitive-agent
```

### Step 2 — Create Virtual Environment (Python 3.11)

```powershell
# Windows
py -3.11 -m venv venv
.\venv\Scripts\activate
```

```bash
# Mac / Linux
python3.11 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r voice_cognitive_agent_web/requirements.txt
```

### Step 4 — Install Tesseract OCR

- **Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki
  Install to `C:\Program Files\Tesseract-OCR\`
- **Mac:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

### Step 5 — Download YOLO Model Weights

The weights file is too large for GitHub. Download it manually:

```bash
# Option 1: Python download
python -c "
import urllib.request
urllib.request.urlretrieve(
    'https://pjreddie.com/media/files/yolov3-tiny.weights',
    'models/yolov3-tiny.weights'
)
print('Done!')
"

# Option 2: Direct download (paste in browser)
# https://pjreddie.com/media/files/yolov3-tiny.weights
```

Place the downloaded file in the `models/` folder.

### Step 6 — Run the App

```bash
cd voice_cognitive_agent_web
python app.py
```

Open **http://localhost:5000** in Google Chrome.

---

## 🗣️ Voice Commands

| Say This                 | Action                                |
|--------------------------|---------------------------------------|
| `"what do you see"`      | Object detection via laptop webcam    |
| `"read this"`            | OCR text reading via camera           |
| `"what time is it"`      | Speaks current time                   |
| `"what is today's date"` | Speaks today's date                   |
| `"who are you"`          | Agent self-introduction               |
| `"help"`                 | Lists all available commands          |
| `"stop"`                 | Exits the agent                       |

---

## ♿ Accessibility Features

- **Atkinson Hyperlegible font** — designed for low vision users
- **High contrast dark theme** — black background with bright green accent
- **Giant 180px tap-to-speak button** — primary control for blind users
- **Auto-speak every response** — SpeechSynthesis reads all answers aloud
- **ARIA live regions** — screen readers (NVDA/JAWS) announce all changes
- **Keyboard shortcuts** — Space = mic toggle, Enter = send text
- **20px minimum font size** throughout the interface
- **Skip-to-content link** for keyboard-only navigation

---

## 📊 Demo Results

| Object Detected | Confidence |
|-----------------|------------|
| Person          | 92%        |
| Laptop          | 87%        |
| Clock           | 77%        |

- **Response time:** ~1.2 seconds
- **Intent confidence:** 85%
- **COCO object classes:** 80
- **OCR languages:** 100+

---

## 🔮 Future Scope

- GPT integration for complex question answering
- Offline speech recognition using Whisper / Wav2Vec
- Mobile app version
- Real-time navigation assistance
- Watson Assistant for advanced NLP
- Multi-language support

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🤝 Acknowledgements

- YOLOv3 by Joseph Redmon — https://pjreddie.com
- Tesseract OCR by Google — https://github.com/tesseract-ocr
- OpenCV — https://opencv.org
- Flask — https://flask.palletsprojects.com
- Alliance University for the project opportunity

---

<div align="center">
Made by <b>Batch-6</b> | Alliance University | E1CSC 25608<br>
Guided by <b>Dr. Jayabhaduri R</b>
</div>
