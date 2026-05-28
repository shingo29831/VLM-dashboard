# 役割: OpenCV, EasyOCR, YOLOv8, VLM を用いた画像解析マイクロサービス
# AI向け役割: 既存のOCR/YOLO機能に加えて、VLM(Vision-Language Model)を利用した画像テキスト解析APIエンドポイントを提供する。UIとロジックを疎結合に保つ。

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import easyocr
from ultralytics import YOLO
from typing import Optional

from vlm_service import analyze_image_with_vlm, get_available_models, TokenUsage

ocr_reader = easyocr.Reader(['ja', 'en'])

yolo_model = YOLO('ui-master-best.pt') 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class VLMAnalyzeResponse(BaseModel):
    text: str
    usage: Optional[TokenUsage] = None

# ---------------------------------------------------------
# API 1: EasyOCRフルスキャン (精密な座標結合ロジック)
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
        
        print("\n=== [DEBUG] EasyOCR スキャン開始 ===")
        results = ocr_reader.readtext(img)
        
        raw_boxes = []
        for (bbox, text, prob) in results:
            text = text.strip()
            if prob < 0.1 or not text:
                continue
            if len(text) == 1 and not text.isalnum():
                continue
                
            x_min = int(min([p[0] for p in bbox]))
            y_min = int(min([p[1] for p in bbox]))
            x_max = int(max([p[0] for p in bbox]))
            y_max = int(max([p[1] for p in bbox]))
            
            raw_boxes.append({
                'text': text,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max
            })

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
                        block['text'] += " " + text 
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
            text = block['text'].strip()
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

        print(f"=== [DEBUG] EasyOCR {len(ui_elements)}個のテキスト要素を抽出完了 ===")
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
        print("\n=== [DEBUG] YOLO 物体検出開始 ===")
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: raise HTTPException(status_code=400, detail="Invalid image format")
        height, width, _ = img.shape
        
        results = yolo_model(img, conf=0.25)
        yolo_elements = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                confidence = round(float(box.conf[0].item()), 2)
                label = result.names[class_id]
                
                yolo_elements.append(YoloElement(
                    label=f"{label} ({confidence})",
                    confidence=confidence,
                    bounding_box=[int((y1/height)*1000), int((x1/width)*1000), int((y2/height)*1000), int((x2/width)*1000)]
                ))
                
        print(f"=== [DEBUG] YOLO {len(yolo_elements)}個のUI要素を検出完了 ===")
        return YoloResponse(elements=yolo_elements)
    except Exception as e:
        print(f"Error processing image in YOLO: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# API 3: VLMによる画像解析 (外部LM Studio連携)
# ---------------------------------------------------------
@app.get("/api/models")
async def fetch_models():
    models = get_available_models()
    return {"models": models}

@app.post("/api/analyze", response_model=VLMAnalyzeResponse)
async def analyze_image_vlm(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: str = Form(...)
):
    try:
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
            
        result_text, usage_info = analyze_image_with_vlm(contents, prompt, model)
        
        return VLMAnalyzeResponse(
            text=result_text,
            usage=usage_info
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error processing image in VLM analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)