# 役割: OpenCV, EasyOCR, YOLOv8 を用いたハイブリッド画像解析マイクロサービス
# AI向け役割: Tesseractをより高精度な深層学習ベースのEasyOCRに換装。日本語・英語混じりのUIテキストを正確に読み取り、座標結合を行う。

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import easyocr # ★ Tesseractの代わりにEasyOCRをインポート
from ultralytics import YOLO

# ★ EasyOCRのリーダーを初期化（日本語と英語に対応）
# 初回起動時に自動で軽量な言語モデルがダウンロードされます
ocr_reader = easyocr.Reader(['ja', 'en'])

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
        
        # ★ EasyOCRでテキスト読み取りを実行
        print("\n=== [DEBUG] EasyOCR スキャン開始 ===")
        results = ocr_reader.readtext(img)
        
        raw_boxes = []
        for (bbox, text, prob) in results:
            text = text.strip()
            # 信頼度が極端に低いものや空文字はスキップ
            if prob < 0.1 or not text:
                continue
            if len(text) == 1 and not text.isalnum():
                continue
                
            # bboxは4つの角の座標 [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] なので最小/最大を計算
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

        # Y中心座標(15px丸め) -> X座標の順でソートして結合（前回の優秀なロジックを流用）
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
                        block['text'] += " " + text # EasyOCRは単語間のスペースを空ける
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
    # ...（前回と同じYOLOの処理）...
    try:
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
        return YoloResponse(elements=yolo_elements)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)