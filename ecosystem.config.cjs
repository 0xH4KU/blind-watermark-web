module.exports = {
  apps: [
    {
      name: "watermark-backend",
      cwd: "./web-app/backend",
      script: "/Users/HAKU/github/blind-watermark-web/.venv/bin/python",
      args: "-m uvicorn main:app --host 0.0.0.0 --port 6123",
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
    {
      name: "watermark-frontend",
      cwd: "./web-app/frontend",
      script: "npm",
      args: "run dev -- --host 0.0.0.0 --port 6124",
    },
  ],
};
