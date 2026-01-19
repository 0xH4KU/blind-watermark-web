from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blind_watermark import WaterMark  # noqa: E402

app = FastAPI(title="Blind Watermark Web API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:6124",
        "http://127.0.0.1:6124",
        "http://mac.sgponte:6124",
        "https://mac.sgponte:6124",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=400, detail="Empty image payload")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="Image decode failed")
    return img


def map_png_compression(compression_ratio: Optional[int]) -> int:
    if compression_ratio is None:
        return 3
    ratio = max(0, min(100, compression_ratio))
    return int(round((100 - ratio) / 11.111))


def encode_image(
    img: np.ndarray, output_format: str, compression_ratio: Optional[int]
) -> Tuple[bytes, str]:
    fmt = output_format.lower()
    if fmt not in ("png", "jpg", "jpeg"):
        raise HTTPException(status_code=400, detail="output_format must be png or jpg")

    if fmt in ("jpg", "jpeg") and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if fmt in ("jpg", "jpeg"):
        quality = 95 if compression_ratio is None else max(0, min(100, compression_ratio))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        mime = "image/jpeg"
    else:
        compression = map_png_compression(compression_ratio)
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        mime = "image/png"

    if not ok:
        raise HTTPException(status_code=500, detail="Image encode failed")
    return buf.tobytes(), mime


@app.get("/")
def root() -> dict:
    return {"status": "ok", "message": "Blind Watermark Web API"}


@app.post("/api/embed/text")
async def embed_text(
    image: UploadFile = File(...),
    watermark: str = Form(...),
    password_img: int = Form(1),
    password_wm: int = Form(1),
    output_format: str = Form("png"),
    compression_ratio: Optional[int] = Form(None),
) -> dict:
    data = await image.read()
    img = decode_image(data)

    wm = WaterMark(password_wm=password_wm, password_img=password_img)
    wm.read_img(img=img)
    wm.read_wm(watermark, mode="str")

    try:
        embedded = wm.embed()
    except AssertionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    encoded, mime = encode_image(embedded, output_format=output_format, compression_ratio=compression_ratio)
    return {
        "image_base64": base64.b64encode(encoded).decode("ascii"),
        "mime_type": mime,
        "wm_size": int(wm.wm_size),
    }


@app.post("/api/extract/text")
async def extract_text(
    image: UploadFile = File(...),
    wm_size: int = Form(...),
    password_img: int = Form(1),
    password_wm: int = Form(1),
) -> dict:
    if wm_size <= 0:
        raise HTTPException(status_code=400, detail="wm_size must be > 0")

    data = await image.read()
    img = decode_image(data)

    wm = WaterMark(password_wm=password_wm, password_img=password_img)
    try:
        text = wm.extract(embed_img=img, wm_shape=[wm_size], mode="str")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Extraction failed") from exc

    return {"watermark": text}
