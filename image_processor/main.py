# 役割: OpenCV, Tesseract OCR, YOLOv8 を用いたハイブリッド画像解析マイクロサービス
# AI向け役割: OCRによるテキスト領域の精密なバウンディングボックス結合と、YOLOによるUI要素（物体）の検出を並列提供するAPIサーバー。

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pytesseract import Output
from ultralytics import YOLO

# Tesseractのパス指定
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# YOLOモデルの読み込み
yolo_model = YOLO('yolov8n.pt') 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- データモデル定義 ---
class UIElement(BaseModel):
    text: str
    bounding_box: list[int]

class ScanResponse(BaseModel):
    elements: list[UIElement]

class YoloElement(BaseModel):
    label: str
    confidence: float
    bounding_box: list[int]

class YoloResponse(BaseModel):
    elements: list[YoloElement]

# ---------------------------------------------------------
# API 1: OCRフルスキャン (精密な座標結合ロジック)
# ---------------------------------------------------------
@app.post("/api/scan", response_model=ScanResponse)
async def scan_image_for_ui(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        height, width, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        custom_config = r'--oem 3 --psm 11'
        d = pytesseract.image_to_data(gray, output_type=Output.DICT, lang='jpn+eng', config=custom_config)
        
        raw_boxes = []
        n_boxes = len(d['text'])
        
        for i in range(n_boxes):
            text = d['text'][i].strip()
            conf = int(d['conf'][i])
            
            if conf < 10 or not text:
                continue
            if len(text) == 1 and not text.isalnum():
                continue
                
            raw_boxes.append({
                'text': text,
                'x_min': d['left'][i],
                'y_min': d['top'][i],
                'x_max': d['left'][i] + d['width'][i],
                'y_max': d['top'][i] + d['height'][i]
            })

        # Y中心座標(15px丸め) -> X座標の順でソート
        for b in raw_boxes:
            b['y_center'] = (b['y_min'] + b['y_max']) / 2
        raw_boxes.sort(key=lambda b: (b['y_center'] // 15, b['x_min']))

        merged_blocks = []
        for box in raw_boxes:
            text = box['text']
            left = box['x_min']
            top = box['y_min']
            right = box['x_max']
            bottom = box['y_max']
            box_h = bottom - top
            
            added = False
            for block in merged_blocks:
                y_overlap = max(0, min(block['y_max'], bottom) - max(block['y_min'], top))
                min_h = min(block['y_max'] - block['y_min'], box_h)
                
                if min_h > 0 and y_overlap > min_h * 0.3:
                    gap = left - block['x_max']
                    if -box_h * 2.0 <= gap <= box_h * 2.5:
                        block['text'] += text
                        block['x_min'] = min(block['x_min'], left)
                        block['y_min'] = min(block['y_min'], top)
                        block['x_max'] = max(block['x_max'], right)
                        block['y_max'] = max(block['y_max'], bottom)
                        added = True
                        break
            
            if not added:
                merged_blocks.append(box)

        ui_elements = []
        for block in merged_blocks:
            text = block['text']
            if len(text) < 2 and not text.isalnum():
                continue
                
            y = block['y_min']
            x = block['x_min']
            w = block['x_max'] - block['x_min']
            h = block['y_max'] - block['y_min']
            
            rel_ymin = int((y / height) * 1000)
            rel_xmin = int((x / width) * 1000)
            rel_ymax = int(((y + h) / height) * 1000)
            rel_xmax = int(((x + w) / width) * 1000)
            
            bbox = [rel_ymin, rel_xmin, rel_ymax, rel_xmax]
            ui_elements.append(UIElement(text=text, bounding_box=bbox))

        return ScanResponse(elements=ui_elements)

    except Exception as e:
        print(f"Error processing image in OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# API 2: YOLO物体検出 (UI要素のピクセルバウンディング)
# ---------------------------------------------------------
@app.post("/api/yolo", response_model=YoloResponse)
async def run_yolo_detection(image: UploadFile = File(...)):
    try:
        # FastAPIのFileは一度読むとポインタが進むため、ここで読み直す
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        height, width, _ = img.shape
        
        # YOLO推論実行 (信頼度25%以上)
        results = yolo_model(img, conf=0.25)
        
        yolo_elements = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                confidence = round(float(box.conf[0].item()), 2)
                label = result.names[class_id]
                
                rel_ymin = int((y1 / height) * 1000)
                rel_xmin = int((x1 / width) * 1000)
                rel_ymax = int((y2 / height) * 1000)
                rel_xmax = int((x2 / width) * 1000)
                
                yolo_elements.append(YoloElement(
                    label=f"{label} ({confidence})",
                    confidence=confidence,
                    bounding_box=[rel_ymin, rel_xmin, rel_ymax, rel_xmax]
                ))
                
        print(f"[YOLO] {len(yolo_elements)}個のオブジェクトを検出しました")
        return YoloResponse(elements=yolo_elements)

    except Exception as e:
        print(f"Error processing image in YOLO: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)