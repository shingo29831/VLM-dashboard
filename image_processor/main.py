import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pytesseract import Output

# Tesseractのパス指定 (環境に合わせて確認)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
    bounding_box: list[int] # [ymin, xmin, ymax, xmax] (0-1000の相対座標)

class ScanResponse(BaseModel):
    elements: list[UIElement]

@app.post("/api/scan", response_model=ScanResponse)
async def scan_image_for_ui(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        height, width, _ = img.shape
        
        # OCRの精度を上げるための前処理 (グレースケール化)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Tesseractの image_to_data を使用して、テキストとその座標を辞書形式で一括取得
        # psm 11: 疎らなテキスト（UIなど）をできるだけ多く見つける設定
        custom_config = r'--oem 3 --psm 11'
        d = pytesseract.image_to_data(gray, output_type=Output.DICT, lang='jpn+eng', config=custom_config)
        
        ui_elements = []
        n_boxes = len(d['text'])
        
        for i in range(n_boxes):
            text = d['text'][i].strip()
            
            # 空白や極端に短い記号のノイズは除外
            if not text or len(text) < 2:
                continue
                
            # ピクセル座標を取得
            x = d['left'][i]
            y = d['top'][i]
            w = d['width'][i]
            h = d['height'][i]
            
            # フロントエンド用に 0-1000 の相対座標に変換
            rel_ymin = int((y / height) * 1000)
            rel_xmin = int((x / width) * 1000)
            rel_ymax = int(((y + h) / height) * 1000)
            rel_xmax = int(((x + w) / width) * 1000)
            
            ui_elements.append(UIElement(
                text=text,
                bounding_box=[rel_ymin, rel_xmin, rel_ymax, rel_xmax]
            ))
            
        return ScanResponse(elements=ui_elements)

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)