from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import datetime, os, base64

app = Flask(__name__)
CORS(app)

# ── Lazy imports ───────────────────────────────────────────────────────────────
def get_cv2():
    import cv2
    return cv2

def get_pytesseract():
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return pytesseract

# ── YOLOv3-tiny Model ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "..", "models")  # relative path for GitHub
CFG_PATH    = os.path.join(MODEL_DIR, "yolov3-tiny.cfg")
WEIGHTS_PATH= os.path.join(MODEL_DIR, "yolov3-tiny.weights")
NAMES_PATH  = os.path.join(MODEL_DIR, "coco.names")


_yolo_net    = None
_yolo_labels = None

def load_yolo():
    global _yolo_net, _yolo_labels

    if _yolo_net is not None:
        return _yolo_net, _yolo_labels, None

    # Check all files exist
    missing = []
    for p in [CFG_PATH, WEIGHTS_PATH, NAMES_PATH]:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        return None, None, f"Missing model files: {', '.join(missing)}"

    # Check weights file is large enough (must be >30MB)
    wsize = os.path.getsize(WEIGHTS_PATH)
    if wsize < 30_000_000:
        return None, None, (
            f"yolov3-tiny.weights is too small ({wsize} bytes). "
            "Please re-download it from https://pjreddie.com/media/files/yolov3-tiny.weights"
        )

    try:
        cv2 = get_cv2()
        net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(NAMES_PATH, "r") as f:
            labels = [line.strip() for line in f.readlines()]

        _yolo_net    = net
        _yolo_labels = labels
        return net, labels, None

    except Exception as e:
        return None, None, f"YOLO load failed: {e}"

# ── Intent Parser ──────────────────────────────────────────────────────────────
INTENT_PATTERNS = [
    ("read_text",       ["read", "scan", "ocr", "what does it say", "read this", "read text"]),
    ("identify_object", ["what is this", "identify", "detect", "what do you see",
                         "describe", "what is in front", "identify object", "see"]),
    ("stop",            ["stop", "exit", "quit", "bye", "goodbye"]),
    ("help",            ["help", "assist", "what can you do", "commands"]),
    ("time_date",       ["time", "date", "today", "day", "clock"]),
    ("answer_query",    ["what", "who", "where", "when", "how", "why", "tell me", "explain"]),
]

def parse_intent(text):
    tl = text.lower().strip()
    for intent, keywords in INTENT_PATTERNS:
        for kw in keywords:
            if kw in tl:
                return intent, 0.85
    return "answer_query", 0.4

def process_query(text):
    raw = text.lower()
    if "your name" in raw or "who are you" in raw:
        return "I am your Voice-Assisted Cognitive Agent, designed to help visually impaired users."
    if "hello" in raw or "hi" in raw:
        return "Hello! I am ready to assist you. Say help to see available commands."
    if "weather" in raw:
        return "I don't have live internet access right now to check weather."
    if "joke" in raw:
        return "Why do programmers prefer dark mode? Because light attracts bugs!"
    return f"You asked: {text}. I understood your query. For complex questions, please connect me to an online service."

# ── Webcam capture ─────────────────────────────────────────────────────────────
def capture_frame():
    cv2 = get_cv2()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "Could not open laptop webcam. Please check camera permissions."
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None, "Could not capture image from webcam."
    return frame, None

def frame_to_base64(frame):
    cv2 = get_cv2()
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode('utf-8')

# ── YOLO Object Detection ──────────────────────────────────────────────────────
def detect_objects():
    frame, err = capture_frame()
    if err:
        return None, err

    net, labels, err = load_yolo()
    if err:
        return frame_to_base64(frame) if frame is not None else None, err

    try:
        cv2   = get_cv2()
        H, W  = frame.shape[:2]

        blob  = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Get output layer names
        layer_names = net.getLayerNames()
        out_indices  = net.getUnconnectedOutLayers()
        # Handle both flat and nested array from different OpenCV versions
        if isinstance(out_indices[0], (list, tuple)):
            output_layers = [layer_names[i[0]-1] for i in out_indices]
        else:
            output_layers = [layer_names[i-1] for i in out_indices]

        outputs = net.forward(output_layers)

        found = []
        seen  = set()
        for output in outputs:
            for detection in output:
                scores     = detection[5:]
                class_id   = int(scores.argmax())
                confidence = float(scores[class_id])
                if confidence > 0.4 and class_id < len(labels):
                    label = labels[class_id]
                    if label not in seen:
                        seen.add(label)
                        found.append(f"{label} ({int(confidence*100)}% confidence)")

        img_b64 = frame_to_base64(frame)
        if found:
            return img_b64, "I can see: " + ", ".join(found) + "."
        return img_b64, "No objects detected. Please ensure good lighting and point the camera clearly at objects."

    except Exception as e:
        return frame_to_base64(frame), f"Detection error: {e}"

# ── OCR Text Reading ───────────────────────────────────────────────────────────
def read_text():
    frame, err = capture_frame()
    if err:
        return None, err
    try:
        cv2  = get_cv2()
        tess = get_pytesseract()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = tess.image_to_string(thresh).strip()
        img_b64 = frame_to_base64(frame)
        if text:
            return img_b64, f"I can read the following text: {text}"
        return img_b64, "No readable text found. Please hold text closer to the camera in good lighting."
    except Exception as e:
        return None, f"OCR error: {e}. Make sure Tesseract is installed."

# ── Flask Routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No input received"}), 400

    intent, confidence = parse_intent(text)
    image_b64 = None

    if intent == "stop":
        response = "Goodbye! Stay safe."

    elif intent == "help":
        response = (
            "Available commands: "
            "Say identify or what do you see to detect objects using your laptop camera. "
            "Say read this to read text using your camera. "
            "Say what time is it for the current time. "
            "Say what is today's date for the date. "
            "Say who are you to learn about me. "
            "Say stop to exit."
        )

    elif intent == "time_date":
        now = datetime.datetime.now()
        if "time" in text.lower():
            response = f"The current time is {now.strftime('%I:%M %p')}."
        elif "date" in text.lower() or "today" in text.lower():
            response = f"Today is {now.strftime('%A, %B %d, %Y')}."
        else:
            response = f"It is {now.strftime('%A, %B %d, %Y')} and the time is {now.strftime('%I:%M %p')}."

    elif intent == "identify_object":
        image_b64, response = detect_objects()

    elif intent == "read_text":
        image_b64, response = read_text()

    else:
        response = process_query(text)

    result = {
        "response":   response,
        "intent":     intent,
        "confidence": round(confidence, 2),
        "action":     intent,
    }
    if image_b64:
        result["image"] = image_b64
    return jsonify(result)

@app.route("/webcam_snapshot")
def webcam_snapshot():
    try:
        frame, err = capture_frame()
        if err:
            return err, 503
        cv2 = get_cv2()
        _, buf = cv2.imencode('.jpg', frame)
        return Response(buf.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        return str(e), 500

@app.route("/health")
def health():
    # Report model status
    weights_ok = (
        os.path.exists(WEIGHTS_PATH) and
        os.path.getsize(WEIGHTS_PATH) > 30_000_000
    )
    return jsonify({
        "status": "ok",
        "yolo_cfg":     os.path.exists(CFG_PATH),
        "yolo_weights": weights_ok,
        "yolo_names":   os.path.exists(NAMES_PATH),
        "weights_size": os.path.getsize(WEIGHTS_PATH) if os.path.exists(WEIGHTS_PATH) else 0
    })

if __name__ == "__main__":
    print("=" * 50)
    print("  Voice Cognitive Agent — YOLO Edition")
    print(f"  CFG:     {'OK' if os.path.exists(CFG_PATH) else 'MISSING'}")
    print(f"  WEIGHTS: {'OK' if os.path.exists(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) > 30_000_000 else 'MISSING/INCOMPLETE'}")
    print(f"  NAMES:   {'OK' if os.path.exists(NAMES_PATH) else 'MISSING'}")
    print("  Open http://localhost:5000 in Chrome")
    print("=" * 50)
    app.run(debug=True, port=5000)
