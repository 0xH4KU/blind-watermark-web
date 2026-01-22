# Web App (Local)

This is a minimal local web app wrapper around the `blind_watermark` Python library.
This fork adds local UI flows for text, image, and bit watermarks plus crop/screenshot recovery.

## Features

- Embed text, image, and bit watermarks.
- Extract text, image, and bit watermarks (requires wm_size or wm_shape).
- Recover from crop/screenshot attacks using the original image as a reference.
- Generate attack samples (rotate, crop, screenshot, resize, noise, mask, brightness) to validate robustness.
- Set passwords and output formats from the UI settings panel.
- One-click recover + extract text flow (recover image then extract text in a single action).
- Manual crop helper to trim screenshot borders before recovery.

## Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r web-app/backend/requirements.txt
cd web-app/backend
python3 -m uvicorn main:app --reload --port 6123
```

## Frontend

```bash
cd web-app/frontend
npm install
npm run dev -- --port 6124
```

Open `http://localhost:6124` and point the API base to `http://localhost:6123`.

## PM2 (optional)

```bash
pm2 start ecosystem.config.cjs
```

Stop with `pm2 stop watermark-backend watermark-frontend`.
