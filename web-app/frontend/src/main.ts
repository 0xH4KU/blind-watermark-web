import "./style.css";

const DEFAULT_BASE = "http://localhost:6123";

 type StatusKind = "loading" | "good" | "bad";

const apiBaseInput = document.querySelector<HTMLInputElement>("#apiBase");

const embedForm = document.querySelector<HTMLFormElement>("#embedForm");
const embedStatus = document.querySelector<HTMLDivElement>("#embedStatus");
const embedPreview = document.querySelector<HTMLImageElement>("#embedPreview");
const embedDownload = document.querySelector<HTMLAnchorElement>("#embedDownload");
const embedWmSize = document.querySelector<HTMLSpanElement>("#embedWmSize");

const extractForm = document.querySelector<HTMLFormElement>("#extractForm");
const extractStatus = document.querySelector<HTMLDivElement>("#extractStatus");
const extractOutput = document.querySelector<HTMLTextAreaElement>("#extractOutput");
const extractWmSize = document.querySelector<HTMLInputElement>("#extractWmSize");
const copyExtract = document.querySelector<HTMLButtonElement>("#copyExtract");

if (!apiBaseInput) {
  throw new Error("Missing API base input");
}
if (!embedForm || !embedStatus || !embedPreview || !embedDownload || !embedWmSize) {
  throw new Error("Missing embed controls");
}
if (!extractForm || !extractStatus || !extractOutput || !extractWmSize || !copyExtract) {
  throw new Error("Missing extract controls");
}

const savedBase = localStorage.getItem("apiBase");
apiBaseInput.value = savedBase ?? DEFAULT_BASE;
const savedWmSize = localStorage.getItem("lastWmSize");
if (savedWmSize) {
  extractWmSize.value = savedWmSize;
}

function getApiBase(): string {
  const raw = apiBaseInput.value.trim();
  const normalized = raw.endsWith("/") ? raw.slice(0, -1) : raw;
  return normalized || DEFAULT_BASE;
}

function setStatus(el: HTMLElement, message: string, kind: StatusKind): void {
  el.textContent = message;
  el.classList.remove("good", "bad");
  if (kind === "good") {
    el.classList.add("good");
  }
  if (kind === "bad") {
    el.classList.add("bad");
  }
}

async function readError(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as { detail?: string };
    if (data.detail) {
      return data.detail;
    }
  } catch {
    // ignore
  }
  return `Request failed (${response.status})`;
}

apiBaseInput.addEventListener("change", () => {
  localStorage.setItem("apiBase", apiBaseInput.value.trim());
});

embedForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(embedStatus, "Embedding...", "loading");
  embedPreview.classList.remove("shown");
  embedWmSize.textContent = "-";
  embedDownload.classList.add("disabled");
  embedDownload.removeAttribute("href");

  const formData = new FormData(embedForm);

  try {
    const response = await fetch(`${getApiBase()}/api/embed/text`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const data = (await response.json()) as {
      image_base64: string;
      mime_type: string;
      wm_size: number;
    };

    const src = `data:${data.mime_type};base64,${data.image_base64}`;
    embedPreview.src = src;
    embedPreview.classList.add("shown");
    embedDownload.href = src;
    embedDownload.download = data.mime_type === "image/png" ? "watermarked.png" : "watermarked.jpg";
    embedDownload.classList.remove("disabled");
    embedWmSize.textContent = String(data.wm_size);
    extractWmSize.value = String(data.wm_size);
    localStorage.setItem("lastWmSize", String(data.wm_size));

    setStatus(embedStatus, "Done", "good");
  } catch (err) {
    const message = err instanceof Error ? err.message : "Embed failed";
    setStatus(embedStatus, message, "bad");
  }
});

extractForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(extractStatus, "Extracting...", "loading");
  extractOutput.value = "";

  const formData = new FormData(extractForm);
  try {
    const response = await fetch(`${getApiBase()}/api/extract/text`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const data = (await response.json()) as { watermark: string };
    extractOutput.value = data.watermark ?? "";
    setStatus(extractStatus, "Recovered", "good");
  } catch (err) {
    const message = err instanceof Error ? err.message : "Extract failed";
    setStatus(extractStatus, message, "bad");
  }
});

copyExtract.addEventListener("click", async () => {
  if (!extractOutput.value) {
    setStatus(extractStatus, "Nothing to copy", "bad");
    return;
  }
  try {
    await navigator.clipboard.writeText(extractOutput.value);
    setStatus(extractStatus, "Copied to clipboard", "good");
  } catch {
    setStatus(extractStatus, "Copy failed", "bad");
  }
});
