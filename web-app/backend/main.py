from __future__ import annotations

import base64
import re
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
from blind_watermark import att  # noqa: E402
from blind_watermark.recover import estimate_crop_parameters, recover_crop  # noqa: E402

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


def decode_gray_image(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=400, detail="Empty image payload")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Image decode failed")
    return img


def normalize_host_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def trim_uniform_border(
    img: np.ndarray,
    *,
    tol: int = 10,
    coverage: float = 0.98,
    max_trim_ratio: float = 0.2,
) -> np.ndarray:
    height, width = img.shape[:2]
    max_top = int(height * max_trim_ratio)
    max_bottom = int(height * max_trim_ratio)
    max_left = int(width * max_trim_ratio)
    max_right = int(width * max_trim_ratio)

    if img.ndim == 2:
        def row_is_border(row: np.ndarray, ref: float) -> bool:
            return float(np.mean(np.abs(row - ref) <= tol)) >= coverage

        def col_is_border(col: np.ndarray, ref: float) -> bool:
            return float(np.mean(np.abs(col - ref) <= tol)) >= coverage

        top_ref = float(np.median(img[0, :]))
        bottom_ref = float(np.median(img[-1, :]))
        left_ref = float(np.median(img[:, 0]))
        right_ref = float(np.median(img[:, -1]))
    else:
        def row_is_border(row: np.ndarray, ref: np.ndarray) -> bool:
            return float(np.mean(np.all(np.abs(row - ref) <= tol, axis=1))) >= coverage

        def col_is_border(col: np.ndarray, ref: np.ndarray) -> bool:
            return float(np.mean(np.all(np.abs(col - ref) <= tol, axis=1))) >= coverage

        top_ref = np.median(img[0, :, :], axis=0)
        bottom_ref = np.median(img[-1, :, :], axis=0)
        left_ref = np.median(img[:, 0, :], axis=0)
        right_ref = np.median(img[:, -1, :], axis=0)

    top = 0
    while top < max_top and row_is_border(img[top, ...], top_ref):
        top += 1

    bottom = height
    while height - bottom < max_bottom and bottom > top + 1 and row_is_border(img[bottom - 1, ...], bottom_ref):
        bottom -= 1

    left = 0
    while left < max_left and col_is_border(img[:, left, ...], left_ref):
        left += 1

    right = width
    while width - right < max_right and right > left + 1 and col_is_border(img[:, right - 1, ...], right_ref):
        right -= 1

    if top == 0 and bottom == height and left == 0 and right == width:
        return img
    if top >= bottom or left >= right:
        return img
    return img[top:bottom, left:right]


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

    if img.ndim == 3 and fmt in ("jpg", "jpeg") and img.shape[2] == 4:
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


def parse_bits(raw_bits: str) -> np.ndarray:
    cleaned = re.sub(r"[^01]", "", raw_bits or "")
    if not cleaned:
        raise HTTPException(status_code=400, detail="watermark_bits must contain 0/1 values")
    return np.array([char == "1" for char in cleaned])


def clamp_ratio(value: float, name: str) -> float:
    if value < 0 or value > 1:
        raise HTTPException(status_code=400, detail=f"{name} must be between 0 and 1")
    return value


def resolve_crop_loc(
    img_shape: Tuple[int, int],
    x1: Optional[int],
    y1: Optional[int],
    x2: Optional[int],
    y2: Optional[int],
    x1r: Optional[float],
    y1r: Optional[float],
    x2r: Optional[float],
    y2r: Optional[float],
) -> Tuple[int, int, int, int]:
    height, width = img_shape
    if all(value is not None for value in (x1, y1, x2, y2)):
        loc = (int(x1), int(y1), int(x2), int(y2))
    else:
        rx1 = clamp_ratio(x1r if x1r is not None else 0.1, "x1r")
        ry1 = clamp_ratio(y1r if y1r is not None else 0.1, "y1r")
        rx2 = clamp_ratio(x2r if x2r is not None else 0.9, "x2r")
        ry2 = clamp_ratio(y2r if y2r is not None else 0.9, "y2r")
        if rx1 >= rx2 or ry1 >= ry2:
            raise HTTPException(status_code=400, detail="x1r/x2r and y1r/y2r must define a valid box")
        loc = (int(width * rx1), int(height * ry1), int(width * rx2), int(height * ry2))

    x1v, y1v, x2v, y2v = loc
    if not (0 <= x1v < x2v <= width and 0 <= y1v < y2v <= height):
        raise HTTPException(status_code=400, detail="Crop coordinates must be inside the image bounds")
    return loc


def attach_wm_bits(wm: WaterMark, wm_bits: np.ndarray) -> None:
    wm.wm_bit = wm_bits
    wm.wm_size = wm_bits.size
    np.random.RandomState(wm.password_wm).shuffle(wm.wm_bit)
    wm.bwm_core.read_wm(wm.wm_bit)


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
    img = normalize_host_image(decode_image(data))

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
    img = normalize_host_image(decode_image(data))

    wm = WaterMark(password_wm=password_wm, password_img=password_img)
    try:
        text = wm.extract(embed_img=img, wm_shape=[wm_size], mode="str")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Extraction failed") from exc

    return {"watermark": text}


@app.post("/api/embed/image")
async def embed_image(
    image: UploadFile = File(...),
    watermark_image: UploadFile = File(...),
    password_img: int = Form(1),
    password_wm: int = Form(1),
    output_format: str = Form("png"),
    compression_ratio: Optional[int] = Form(None),
) -> dict:
    data = await image.read()
    img = normalize_host_image(decode_image(data))
    wm_data = await watermark_image.read()
    wm_img = decode_gray_image(wm_data)
    wm_shape = {"width": int(wm_img.shape[1]), "height": int(wm_img.shape[0])}

    wm = WaterMark(password_wm=password_wm, password_img=password_img)
    wm.read_img(img=img)
    attach_wm_bits(wm, wm_img.flatten() > 128)

    try:
        embedded = wm.embed()
    except AssertionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    encoded, mime = encode_image(embedded, output_format=output_format, compression_ratio=compression_ratio)
    return {
        "image_base64": base64.b64encode(encoded).decode("ascii"),
        "mime_type": mime,
        "wm_shape": wm_shape,
        "wm_size": int(wm.wm_size),
    }


@app.post("/api/embed/bit")
async def embed_bit(
    image: UploadFile = File(...),
    watermark_bits: str = Form(...),
    password_img: int = Form(1),
    password_wm: int = Form(1),
    output_format: str = Form("png"),
    compression_ratio: Optional[int] = Form(None),
) -> dict:
    data = await image.read()
    img = normalize_host_image(decode_image(data))
    bits = parse_bits(watermark_bits)

    wm = WaterMark(password_wm=password_wm, password_img=password_img)
    wm.read_img(img=img)
    wm.read_wm(bits, mode="bit")

    try:
        embedded = wm.embed()
    except AssertionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    encoded, mime = encode_image(embedded, output_format=output_format, compression_ratio=compression_ratio)
    return {
        "image_base64": base64.b64encode(encoded).decode("ascii"),
        "mime_type": mime,
        "wm_size": int(wm.wm_size),
        "bit_string": "".join("1" if bit else "0" for bit in bits),
    }


@app.post("/api/extract/image")
async def extract_image(
    image: UploadFile = File(...),
    wm_width: int = Form(...),
    wm_height: int = Form(...),
    password_img: int = Form(1),
    password_wm: int = Form(1),
    output_format: str = Form("png"),
    compression_ratio: Optional[int] = Form(None),
) -> dict:
    if wm_width <= 0 or wm_height <= 0:
        raise HTTPException(status_code=400, detail="wm_width and wm_height must be > 0")

    data = await image.read()
    img = normalize_host_image(decode_image(data))

    wm = WaterMark(password_wm=password_wm, password_img=password_img)
    wm.wm_size = int(wm_width * wm_height)
    wm_avg = wm.bwm_core.extract(img=img, wm_shape=(wm_height, wm_width))
    wm_bits = wm.extract_decrypt(wm_avg=wm_avg)
    wm_img = (255 * wm_bits.reshape((wm_height, wm_width))).astype(np.uint8)

    encoded, mime = encode_image(wm_img, output_format=output_format, compression_ratio=compression_ratio)
    return {
        "image_base64": base64.b64encode(encoded).decode("ascii"),
        "mime_type": mime,
    }


@app.post("/api/extract/bit")
async def extract_bit(
    image: UploadFile = File(...),
    wm_size: int = Form(...),
    password_img: int = Form(1),
    password_wm: int = Form(1),
) -> dict:
    if wm_size <= 0:
        raise HTTPException(status_code=400, detail="wm_size must be > 0")

    data = await image.read()
    img = normalize_host_image(decode_image(data))

    wm = WaterMark(password_wm=password_wm, password_img=password_img)
    try:
        bits = wm.extract(embed_img=img, wm_shape=[wm_size], mode="bit")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Extraction failed") from exc

    bit_list = [int(bit) for bit in bits]
    return {"bits": bit_list, "bit_string": "".join(str(bit) for bit in bit_list)}


@app.post("/api/attack")
async def attack_image(
    image: UploadFile = File(...),
    attack: str = Form(...),
    angle: Optional[float] = Form(None),
    ratio: Optional[float] = Form(None),
    count: Optional[int] = Form(None),
    out_width: Optional[int] = Form(None),
    out_height: Optional[int] = Form(None),
    scale: Optional[float] = Form(None),
    x1: Optional[int] = Form(None),
    y1: Optional[int] = Form(None),
    x2: Optional[int] = Form(None),
    y2: Optional[int] = Form(None),
    x1r: Optional[float] = Form(None),
    y1r: Optional[float] = Form(None),
    x2r: Optional[float] = Form(None),
    y2r: Optional[float] = Form(None),
    output_format: str = Form("png"),
    compression_ratio: Optional[int] = Form(None),
) -> dict:
    data = await image.read()
    img = ensure_bgr(decode_image(data))

    attack_key = attack.strip().lower()
    loc = None
    applied_scale = None

    if attack_key == "rotate":
        angle_value = 45.0 if angle is None else angle
        attacked = att.rot_att(input_img=img, angle=angle_value)
    elif attack_key == "resize":
        if not out_width or not out_height:
            raise HTTPException(status_code=400, detail="out_width and out_height are required for resize")
        if out_width <= 0 or out_height <= 0:
            raise HTTPException(status_code=400, detail="out_width and out_height must be > 0")
        attacked = att.resize_att(input_img=img, out_shape=(int(out_width), int(out_height)))
    elif attack_key == "brightness":
        ratio_value = 0.9 if ratio is None else ratio
        if ratio_value <= 0:
            raise HTTPException(status_code=400, detail="ratio must be > 0")
        attacked = att.bright_att(input_img=img, ratio=ratio_value)
    elif attack_key == "salt_pepper":
        ratio_value = 0.01 if ratio is None else ratio
        if ratio_value < 0 or ratio_value > 1:
            raise HTTPException(status_code=400, detail="ratio must be between 0 and 1")
        attacked = att.salt_pepper_att(input_img=img, ratio=ratio_value)
    elif attack_key == "mask":
        ratio_value = 0.1 if ratio is None else ratio
        if ratio_value <= 0 or ratio_value >= 1:
            raise HTTPException(status_code=400, detail="ratio must be between 0 and 1")
        count_value = 3 if count is None else max(1, count)
        attacked = att.shelter_att(input_img=img, ratio=ratio_value, n=count_value)
    elif attack_key in ("crop", "screenshot"):
        loc = resolve_crop_loc(img.shape[:2], x1, y1, x2, y2, x1r, y1r, x2r, y2r)
        applied_scale = None
        if attack_key == "screenshot":
            applied_scale = 0.7 if scale is None else scale
            if applied_scale <= 0:
                raise HTTPException(status_code=400, detail="scale must be > 0")
        attacked = att.cut_att3(input_img=img, loc=loc, scale=applied_scale)
    elif attack_key == "vcut":
        ratio_value = 0.8 if ratio is None else ratio
        ratio_value = clamp_ratio(ratio_value, "ratio")
        height, width = img.shape[:2]
        loc = (0, 0, width, int(height * ratio_value))
        attacked = att.cut_att3(input_img=img, loc=loc, scale=None)
    elif attack_key == "hcut":
        ratio_value = 0.8 if ratio is None else ratio
        ratio_value = clamp_ratio(ratio_value, "ratio")
        height, width = img.shape[:2]
        loc = (0, 0, int(width * ratio_value), height)
        attacked = att.cut_att3(input_img=img, loc=loc, scale=None)
    else:
        raise HTTPException(status_code=400, detail="Unknown attack type")

    attacked = np.clip(attacked, 0, 255).astype(np.uint8)
    encoded, mime = encode_image(attacked, output_format=output_format, compression_ratio=compression_ratio)
    payload = {
        "image_base64": base64.b64encode(encoded).decode("ascii"),
        "mime_type": mime,
        "attack": attack_key,
        "out_shape": {"height": int(attacked.shape[0]), "width": int(attacked.shape[1])},
    }
    if loc is not None:
        payload["loc"] = {"x1": int(loc[0]), "y1": int(loc[1]), "x2": int(loc[2]), "y2": int(loc[3])}
    if applied_scale is not None:
        payload["scale"] = float(applied_scale)
    return payload


@app.post("/api/recover/crop")
async def recover_crop_api(
    template_image: UploadFile = File(...),
    original_image: Optional[UploadFile] = File(None),
    x1: Optional[int] = Form(None),
    y1: Optional[int] = Form(None),
    x2: Optional[int] = Form(None),
    y2: Optional[int] = Form(None),
    image_width: Optional[int] = Form(None),
    image_height: Optional[int] = Form(None),
    scale_min: float = Form(0.5),
    scale_max: float = Form(2.0),
    search_num: Optional[int] = Form(200),
    output_format: str = Form("png"),
    compression_ratio: Optional[int] = Form(None),
) -> dict:
    template_data = await template_image.read()
    template_raw = decode_image(template_data)
    template_color = ensure_bgr(template_raw)

    loc = None
    score = None
    scale_infer = None
    image_shape = None
    trimmed = False

    loc_known = all(value is not None for value in (x1, y1, x2, y2))
    if loc_known:
        if original_image is not None:
            original_raw = decode_image(await original_image.read())
            image_shape = original_raw.shape[:2]
        elif image_width and image_height:
            image_shape = (image_height, image_width)
        else:
            raise HTTPException(
                status_code=400,
                detail="image_width and image_height are required when original_image is not provided",
            )
        loc = (int(x1), int(y1), int(x2), int(y2))
    else:
        if original_image is None:
            raise HTTPException(status_code=400, detail="original_image is required for parameter estimation")
        original_raw = decode_image(await original_image.read())
        original_gray = to_gray(original_raw)
        if scale_min <= 0 or scale_max <= 0 or scale_max < scale_min:
            raise HTTPException(status_code=400, detail="scale_min/scale_max must be positive and min <= max")
        if scale_min != scale_max and search_num is None:
            raise HTTPException(status_code=400, detail="search_num is required when scale_min != scale_max")
        template_gray = to_gray(template_color)
        (x1, y1, x2, y2), image_shape, score, scale_infer = estimate_crop_parameters(
            ori_img=original_gray,
            tem_img=template_gray,
            scale=(scale_min, scale_max),
            search_num=search_num or 200,
        )
        loc = (int(x1), int(y1), int(x2), int(y2))

        trimmed_template = trim_uniform_border(template_color)
        if trimmed_template.shape != template_color.shape:
            trimmed_gray = to_gray(trimmed_template)
            (tx1, ty1, tx2, ty2), t_shape, t_score, t_scale = estimate_crop_parameters(
                ori_img=original_gray,
                tem_img=trimmed_gray,
                scale=(scale_min, scale_max),
                search_num=search_num or 200,
            )
            if t_score > score:
                template_color = trimmed_template
                loc = (int(tx1), int(ty1), int(tx2), int(ty2))
                image_shape = t_shape
                score = t_score
                scale_infer = t_scale
                trimmed = True

    recovered = recover_crop(tem_img=template_color, loc=loc, image_o_shape=image_shape)
    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    encoded, mime = encode_image(recovered, output_format=output_format, compression_ratio=compression_ratio)
    score_value = float(score) if score is not None else None
    scale_value = float(scale_infer) if scale_infer is not None else None
    return {
        "image_base64": base64.b64encode(encoded).decode("ascii"),
        "mime_type": mime,
        "loc": {"x1": loc[0], "y1": loc[1], "x2": loc[2], "y2": loc[3]},
        "image_shape": {"height": int(image_shape[0]), "width": int(image_shape[1])},
        "score": score_value,
        "scale_infer": scale_value,
        "trimmed": trimmed,
    }
