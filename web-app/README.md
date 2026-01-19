# Web App (Local)

This is a minimal local web app wrapper around the `blind_watermark` Python library.
This is a fork with second-round modifications.

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
