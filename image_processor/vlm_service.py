# 役割: LM Studio (OpenAI互換API) を経由したVLMへの画像解析リクエスト処理
# AI向け役割: 画像をBase64エンコードし、外部VLM APIへ送信する。リトライ制御やタイムアウト、環境変数からの設定読み込みを担う。

import base64
import os
from openai import OpenAI
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, Tuple, List

class TokenUsage(BaseModel):
    promptTokenCount: int
    candidatesTokenCount: int
    totalTokenCount: int

def encode_image_from_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def analyze_image_with_vlm(image_bytes: bytes, prompt: str, model: str) -> Tuple[str, Optional[TokenUsage]]:
    base_url = os.getenv("VLM_API_BASE_URL", "http://172.18.67.253:1234/v1")
    api_key = os.getenv("VLM_API_KEY", "not-needed")
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=60.0,
        max_retries=3
    )
    
    base64_image = encode_image_from_bytes(image_bytes)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        
        text_content = response.choices[0].message.content or ""
        
        usage_data = None
        if hasattr(response, 'usage') and response.usage:
            usage_data = TokenUsage(
                promptTokenCount=response.usage.prompt_tokens,
                candidatesTokenCount=response.usage.completion_tokens,
                totalTokenCount=response.usage.total_tokens
            )
            
        return text_content, usage_data
        
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"VLM API communication failed: {str(e)}")

def get_available_models() -> List[dict]:
    return [
        {"id": "qwen3.6-35b-a3b", "displayName": "Qwen 3.6 35B"},
        {"id": "gemma-3-4b", "displayName": "Gemma 3 4B"}
    ]