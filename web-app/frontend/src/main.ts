const FALLBACK_BASE = "http://localhost:6123";
const DEFAULT_BASE = (() => {
  if (typeof window === "undefined") {
    return FALLBACK_BASE;
  }
  const host = window.location.hostname;
  if (host === "localhost" || host === "127.0.0.1") {
    return FALLBACK_BASE;
  }
  const protocol = window.location.protocol;
  return `${protocol}//${host}:6123`;
})();

type StatusKind = "loading" | "good" | "bad";

function requireElement<T extends HTMLElement>(selector: string): T {
  const el = document.querySelector<T>(selector);
  if (!el) {
    throw new Error(`Missing element: ${selector}`);
  }
  return el;
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

function clearImageResult(preview: HTMLImageElement, download: HTMLAnchorElement): void {
  preview.classList.remove("shown");
  download.classList.add("disabled");
  download.removeAttribute("href");
}

function showImageResult(
  preview: HTMLImageElement,
  download: HTMLAnchorElement,
  src: string,
  filename: string,
): void {
  preview.src = src;
  preview.classList.add("shown");
  download.href = src;
  download.download = filename;
  download.classList.remove("disabled");
}

function formatLoc(loc: { x1: number; y1: number; x2: number; y2: number }): string {
  return `${loc.x1}, ${loc.y1}, ${loc.x2}, ${loc.y2}`;
}

function storeValue(key: string, value: string): void {
  const trimmed = value.trim();
  if (trimmed) {
    localStorage.setItem(key, trimmed);
  } else {
    localStorage.removeItem(key);
  }
}

function restoreValue(input: HTMLInputElement | HTMLSelectElement, key: string): void {
  const saved = localStorage.getItem(key);
  if (saved !== null) {
    input.value = saved;
  }
}

function persistValue(input: HTMLInputElement | HTMLSelectElement, key: string): void {
  restoreValue(input, key);
  input.addEventListener("change", () => {
    storeValue(key, input.value);
  });
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

function dropEmptyFields(formData: FormData, keys: string[]): void {
  for (const key of keys) {
    const value = formData.get(key);
    if (value === "" || value === null) {
      formData.delete(key);
    }
  }
}

function dropEmptyFile(formData: FormData, key: string): void {
  const value = formData.get(key);
  if (value instanceof File && value.name === "" && value.size === 0) {
    formData.delete(key);
  }
}

async function setFileInputFromDataUrl(
  input: HTMLInputElement,
  dataUrl: string,
  filename: string,
): Promise<void> {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  const file = new File([blob], filename, { type: blob.type });
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

async function loadImageFromFile(file: File): Promise<HTMLImageElement> {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.decoding = "async";
  const loadPromise = new Promise<HTMLImageElement>((resolve, reject) => {
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Image load failed"));
  });
  img.src = url;
  try {
    return await loadPromise;
  } finally {
    URL.revokeObjectURL(url);
  }
}

const apiBaseInput = requireElement<HTMLInputElement>("#apiBase");
const passwordImgInput = requireElement<HTMLInputElement>("#passwordImg");
const passwordWmInput = requireElement<HTMLInputElement>("#passwordWm");
const outputFormatSelect = requireElement<HTMLSelectElement>("#outputFormat");
const compressionRatioInput = requireElement<HTMLInputElement>("#compressionRatio");

const embedForm = requireElement<HTMLFormElement>("#embedForm");
const embedStatus = requireElement<HTMLDivElement>("#embedStatus");
const embedPreview = requireElement<HTMLImageElement>("#embedPreview");
const embedDownload = requireElement<HTMLAnchorElement>("#embedDownload");
const embedWmSize = requireElement<HTMLSpanElement>("#embedWmSize");

const extractForm = requireElement<HTMLFormElement>("#extractForm");
const extractStatus = requireElement<HTMLDivElement>("#extractStatus");
const extractTextImage = requireElement<HTMLInputElement>("#extractImage");
const extractOutput = requireElement<HTMLTextAreaElement>("#extractOutput");
const extractWmSize = requireElement<HTMLInputElement>("#extractWmSize");
const copyExtract = requireElement<HTMLButtonElement>("#copyExtract");

const embedImageForm = requireElement<HTMLFormElement>("#embedImageForm");
const embedImageStatus = requireElement<HTMLDivElement>("#embedImageStatus");
const embedImagePreview = requireElement<HTMLImageElement>("#embedImagePreview");
const embedImageDownload = requireElement<HTMLAnchorElement>("#embedImageDownload");
const embedImageWmShape = requireElement<HTMLSpanElement>("#embedImageWmShape");
const embedImageWmSize = requireElement<HTMLSpanElement>("#embedImageWmSize");

const extractImageForm = requireElement<HTMLFormElement>("#extractImageForm");
const extractImageStatus = requireElement<HTMLDivElement>("#extractImageStatus");
const extractImageInput = requireElement<HTMLInputElement>("#extractImageSource");
const extractImagePreview = requireElement<HTMLImageElement>("#extractImagePreview");
const extractImageDownload = requireElement<HTMLAnchorElement>("#extractImageDownload");
const extractImageWidth = requireElement<HTMLInputElement>("#extractImageWidth");
const extractImageHeight = requireElement<HTMLInputElement>("#extractImageHeight");

const embedBitForm = requireElement<HTMLFormElement>("#embedBitForm");
const embedBitStatus = requireElement<HTMLDivElement>("#embedBitStatus");
const embedBitPreview = requireElement<HTMLImageElement>("#embedBitPreview");
const embedBitDownload = requireElement<HTMLAnchorElement>("#embedBitDownload");
const embedBitSize = requireElement<HTMLSpanElement>("#embedBitSize");

const extractBitForm = requireElement<HTMLFormElement>("#extractBitForm");
const extractBitStatus = requireElement<HTMLDivElement>("#extractBitStatus");
const extractBitInput = requireElement<HTMLInputElement>("#extractBitImage");
const extractBitOutput = requireElement<HTMLTextAreaElement>("#extractBitOutput");
const extractBitSize = requireElement<HTMLInputElement>("#extractBitSize");
const copyBit = requireElement<HTMLButtonElement>("#copyBit");

const attackForm = requireElement<HTMLFormElement>("#attackForm");
const attackStatus = requireElement<HTMLDivElement>("#attackStatus");
const attackType = requireElement<HTMLSelectElement>("#attackType");
const attackPreview = requireElement<HTMLImageElement>("#attackPreview");
const attackDownload = requireElement<HTMLAnchorElement>("#attackDownload");
const attackOutShape = requireElement<HTMLSpanElement>("#attackOutShape");
const attackLoc = requireElement<HTMLSpanElement>("#attackLoc");
const attackScaleValue = requireElement<HTMLSpanElement>("#attackScaleValue");
const attackToText = requireElement<HTMLButtonElement>("#attackToText");
const attackToImage = requireElement<HTMLButtonElement>("#attackToImage");
const attackToBit = requireElement<HTMLButtonElement>("#attackToBit");
const attackToRecover = requireElement<HTMLButtonElement>("#attackToRecover");
const attackOptionBlocks = Array.from(document.querySelectorAll<HTMLElement>(".attack-options"));

const recoverForm = requireElement<HTMLFormElement>("#recoverForm");
const recoverStatus = requireElement<HTMLDivElement>("#recoverStatus");
const recoverTemplate = requireElement<HTMLInputElement>("#recoverTemplate");
const recoverPreview = requireElement<HTMLImageElement>("#recoverPreview");
const recoverDownload = requireElement<HTMLAnchorElement>("#recoverDownload");
const recoverLoc = requireElement<HTMLSpanElement>("#recoverLoc");
const recoverScale = requireElement<HTMLSpanElement>("#recoverScale");
const recoverScore = requireElement<HTMLSpanElement>("#recoverScore");
const recoverCropCanvas = requireElement<HTMLCanvasElement>("#recoverCropCanvas");
const recoverCropApply = requireElement<HTMLButtonElement>("#recoverCropApply");
const recoverCropReset = requireElement<HTMLButtonElement>("#recoverCropReset");
const recoverTextWmSize = requireElement<HTMLInputElement>("#recoverTextWmSize");
const recoverTextOutput = requireElement<HTMLTextAreaElement>("#recoverTextOutput");
const copyRecoverText = requireElement<HTMLButtonElement>("#copyRecoverText");
const recoverExtractText = requireElement<HTMLButtonElement>("#recoverExtractText");

type AttackPayload = {
  dataUrl: string;
  filename: string;
  mimeType: string;
};

type RecoverResponse = {
  image_base64: string;
  mime_type: string;
  loc: { x1: number; y1: number; x2: number; y2: number };
  score: number | null;
  scale_infer: number | null;
};

let lastAttack: AttackPayload | null = null;

type CropSelection = { x: number; y: number; w: number; h: number };
type CropPoint = { x: number; y: number };

const cropperState = {
  img: null as HTMLImageElement | null,
  selection: null as CropSelection | null,
  dragging: false,
  start: null as CropPoint | null,
  scaleX: 1,
  scaleY: 1,
};

function renderCropper(): void {
  const ctx = recoverCropCanvas.getContext("2d");
  if (!ctx) {
    return;
  }

  const img = cropperState.img;
  if (!img) {
    recoverCropCanvas.width = 320;
    recoverCropCanvas.height = 200;
    ctx.clearRect(0, 0, recoverCropCanvas.width, recoverCropCanvas.height);
    ctx.fillStyle = "#f7f6f1";
    ctx.fillRect(0, 0, recoverCropCanvas.width, recoverCropCanvas.height);
    ctx.fillStyle = "#6c6c66";
    ctx.font = "14px sans-serif";
    ctx.fillText("Load attacked image", 16, 110);
    return;
  }

  const container = recoverCropCanvas.parentElement;
  const maxHeight = 320;
  const maxWidth = Math.max(1, (container?.clientWidth ?? 480) - 16);
  let scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
  if (!Number.isFinite(scale) || scale <= 0) {
    scale = 1;
  }

  const canvasWidth = Math.max(1, Math.round(img.width * scale));
  const canvasHeight = Math.max(1, Math.round(img.height * scale));
  recoverCropCanvas.width = canvasWidth;
  recoverCropCanvas.height = canvasHeight;
  cropperState.scaleX = img.width / canvasWidth;
  cropperState.scaleY = img.height / canvasHeight;

  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  ctx.drawImage(img, 0, 0, canvasWidth, canvasHeight);

  if (cropperState.selection) {
    const sel = cropperState.selection;
    const x = sel.x / cropperState.scaleX;
    const y = sel.y / cropperState.scaleY;
    const w = sel.w / cropperState.scaleX;
    const h = sel.h / cropperState.scaleY;
    ctx.save();
    ctx.strokeStyle = "#1f1f1d";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x + 0.5, y + 0.5, w, h);
    ctx.restore();
  }
}

function getCanvasPoint(event: PointerEvent): CropPoint {
  const rect = recoverCropCanvas.getBoundingClientRect();
  const scaleX = recoverCropCanvas.width / rect.width;
  const scaleY = recoverCropCanvas.height / rect.height;
  const x = clamp((event.clientX - rect.left) * scaleX, 0, recoverCropCanvas.width);
  const y = clamp((event.clientY - rect.top) * scaleY, 0, recoverCropCanvas.height);
  return { x, y };
}

function canvasToImage(point: CropPoint): CropPoint {
  return {
    x: point.x * cropperState.scaleX,
    y: point.y * cropperState.scaleY,
  };
}

async function loadCropperFromFile(file: File): Promise<void> {
  try {
    cropperState.img = await loadImageFromFile(file);
    cropperState.selection = null;
    cropperState.start = null;
    cropperState.dragging = false;
    renderCropper();
  } catch {
    cropperState.img = null;
    cropperState.selection = null;
    renderCropper();
    setStatus(recoverStatus, "Failed to load cropper image", "bad");
  }
}

const savedBase = localStorage.getItem("apiBase");
apiBaseInput.value = savedBase ?? DEFAULT_BASE;

persistValue(passwordImgInput, "passwordImg");
persistValue(passwordWmInput, "passwordWm");
persistValue(outputFormatSelect, "outputFormat");
persistValue(compressionRatioInput, "compressionRatio");

const savedTextWmSize = localStorage.getItem("lastTextWmSize") ?? localStorage.getItem("lastWmSize");
if (savedTextWmSize) {
  extractWmSize.value = savedTextWmSize;
}
if (savedTextWmSize) {
  recoverTextWmSize.value = savedTextWmSize;
}

const savedBitWmSize = localStorage.getItem("lastBitWmSize");
if (savedBitWmSize) {
  extractBitSize.value = savedBitWmSize;
}

const savedImageWmWidth = localStorage.getItem("lastImageWmWidth");
const savedImageWmHeight = localStorage.getItem("lastImageWmHeight");
if (savedImageWmWidth) {
  extractImageWidth.value = savedImageWmWidth;
}
if (savedImageWmHeight) {
  extractImageHeight.value = savedImageWmHeight;
}

apiBaseInput.addEventListener("change", () => {
  localStorage.setItem("apiBase", apiBaseInput.value.trim());
});

extractWmSize.addEventListener("change", () => storeValue("lastTextWmSize", extractWmSize.value));
recoverTextWmSize.addEventListener("change", () => {
  storeValue("lastTextWmSize", recoverTextWmSize.value);
  if (!extractWmSize.value.trim()) {
    extractWmSize.value = recoverTextWmSize.value;
  }
});
extractBitSize.addEventListener("change", () => storeValue("lastBitWmSize", extractBitSize.value));
extractImageWidth.addEventListener("change", () => storeValue("lastImageWmWidth", extractImageWidth.value));
extractImageHeight.addEventListener("change", () => storeValue("lastImageWmHeight", extractImageHeight.value));

function getApiBase(): string {
  const raw = apiBaseInput.value.trim();
  const normalized = raw.endsWith("/") ? raw.slice(0, -1) : raw;
  return normalized || DEFAULT_BASE;
}

function appendPasswords(formData: FormData): void {
  formData.set("password_img", passwordImgInput.value.trim() || "1");
  formData.set("password_wm", passwordWmInput.value.trim() || "1");
}

function appendOutputOptions(formData: FormData): void {
  formData.set("output_format", outputFormatSelect.value);
  const ratio = compressionRatioInput.value.trim();
  if (ratio) {
    formData.set("compression_ratio", ratio);
  }
}

function buildRecoverFormData(outputFormatOverride?: string): FormData {
  const formData = new FormData(recoverForm);
  appendOutputOptions(formData);
  if (outputFormatOverride) {
    formData.set("output_format", outputFormatOverride);
  }
  dropEmptyFile(formData, "original_image");
  dropEmptyFields(formData, [
    "x1",
    "y1",
    "x2",
    "y2",
    "image_width",
    "image_height",
    "scale_min",
    "scale_max",
    "search_num",
  ]);
  return formData;
}

function applyRecoverResult(data: RecoverResponse): string {
  const src = `data:${data.mime_type};base64,${data.image_base64}`;
  showImageResult(
    recoverPreview,
    recoverDownload,
    src,
    data.mime_type === "image/png" ? "recovered.png" : "recovered.jpg",
  );
  recoverLoc.textContent = formatLoc(data.loc);
  recoverScale.textContent =
    data.scale_infer === null || data.scale_infer === undefined ? "-" : data.scale_infer.toFixed(3);
  recoverScore.textContent =
    data.score === null || data.score === undefined ? "-" : data.score.toFixed(3);
  return src;
}

async function extractTextFromDataUrl(dataUrl: string, wmSize: string): Promise<string> {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  const file = new File([blob], "recovered.png", { type: blob.type });
  const formData = new FormData();
  formData.set("image", file);
  formData.set("wm_size", wmSize);
  appendPasswords(formData);

  const extractResponse = await fetch(`${getApiBase()}/api/extract/text`, {
    method: "POST",
    body: formData,
  });
  if (!extractResponse.ok) {
    throw new Error(await readError(extractResponse));
  }
  const data = (await extractResponse.json()) as { watermark: string };
  return data.watermark ?? "";
}

function updateAttackOptions(): void {
  const selected = attackType.value;
  for (const block of attackOptionBlocks) {
    const isActive = block.dataset.attack === selected;
    block.classList.toggle("active", isActive);
    const inputs = block.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(
      "input, select, textarea",
    );
    inputs.forEach((input) => {
      input.disabled = !isActive;
    });
  }
}

async function useAttackOnInput(input: HTMLInputElement, label: string): Promise<void> {
  if (!lastAttack) {
    setStatus(attackStatus, "Run an attack first", "bad");
    return;
  }
  try {
    await setFileInputFromDataUrl(input, lastAttack.dataUrl, lastAttack.filename);
    setStatus(attackStatus, `Loaded into ${label}`, "good");
  } catch {
    setStatus(attackStatus, "Failed to load attacked image", "bad");
  }
}

embedForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(embedStatus, "Embedding...", "loading");
  clearImageResult(embedPreview, embedDownload);
  embedWmSize.textContent = "-";

  const formData = new FormData(embedForm);
  appendPasswords(formData);
  appendOutputOptions(formData);

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
    showImageResult(
      embedPreview,
      embedDownload,
      src,
      data.mime_type === "image/png" ? "watermarked.png" : "watermarked.jpg",
    );
    embedWmSize.textContent = String(data.wm_size);
    extractWmSize.value = String(data.wm_size);
    recoverTextWmSize.value = String(data.wm_size);
    localStorage.setItem("lastTextWmSize", String(data.wm_size));
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
  appendPasswords(formData);

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

embedImageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(embedImageStatus, "Embedding...", "loading");
  clearImageResult(embedImagePreview, embedImageDownload);
  embedImageWmShape.textContent = "-";
  embedImageWmSize.textContent = "-";

  const formData = new FormData(embedImageForm);
  appendPasswords(formData);
  appendOutputOptions(formData);

  try {
    const response = await fetch(`${getApiBase()}/api/embed/image`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const data = (await response.json()) as {
      image_base64: string;
      mime_type: string;
      wm_shape: { width: number; height: number };
      wm_size: number;
    };
    const src = `data:${data.mime_type};base64,${data.image_base64}`;
    showImageResult(
      embedImagePreview,
      embedImageDownload,
      src,
      data.mime_type === "image/png" ? "watermarked.png" : "watermarked.jpg",
    );
    embedImageWmShape.textContent = `${data.wm_shape.width} x ${data.wm_shape.height}`;
    embedImageWmSize.textContent = String(data.wm_size);
    extractImageWidth.value = String(data.wm_shape.width);
    extractImageHeight.value = String(data.wm_shape.height);
    storeValue("lastImageWmWidth", extractImageWidth.value);
    storeValue("lastImageWmHeight", extractImageHeight.value);

    setStatus(embedImageStatus, "Done", "good");
  } catch (err) {
    const message = err instanceof Error ? err.message : "Embed failed";
    setStatus(embedImageStatus, message, "bad");
  }
});

extractImageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(extractImageStatus, "Extracting...", "loading");
  clearImageResult(extractImagePreview, extractImageDownload);

  const formData = new FormData(extractImageForm);
  appendPasswords(formData);
  appendOutputOptions(formData);

  try {
    const response = await fetch(`${getApiBase()}/api/extract/image`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const data = (await response.json()) as {
      image_base64: string;
      mime_type: string;
    };
    const src = `data:${data.mime_type};base64,${data.image_base64}`;
    showImageResult(
      extractImagePreview,
      extractImageDownload,
      src,
      data.mime_type === "image/png" ? "watermark.png" : "watermark.jpg",
    );
    setStatus(extractImageStatus, "Recovered", "good");
  } catch (err) {
    const message = err instanceof Error ? err.message : "Extract failed";
    setStatus(extractImageStatus, message, "bad");
  }
});

embedBitForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(embedBitStatus, "Embedding...", "loading");
  clearImageResult(embedBitPreview, embedBitDownload);
  embedBitSize.textContent = "-";

  const formData = new FormData(embedBitForm);
  appendPasswords(formData);
  appendOutputOptions(formData);

  try {
    const response = await fetch(`${getApiBase()}/api/embed/bit`, {
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
      bit_string: string;
    };
    const src = `data:${data.mime_type};base64,${data.image_base64}`;
    showImageResult(
      embedBitPreview,
      embedBitDownload,
      src,
      data.mime_type === "image/png" ? "watermarked.png" : "watermarked.jpg",
    );
    embedBitSize.textContent = String(data.wm_size);
    extractBitSize.value = String(data.wm_size);
    storeValue("lastBitWmSize", String(data.wm_size));

    setStatus(embedBitStatus, `Done (bits: ${data.bit_string.length})`, "good");
  } catch (err) {
    const message = err instanceof Error ? err.message : "Embed failed";
    setStatus(embedBitStatus, message, "bad");
  }
});

extractBitForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(extractBitStatus, "Extracting...", "loading");
  extractBitOutput.value = "";

  const formData = new FormData(extractBitForm);
  appendPasswords(formData);

  try {
    const response = await fetch(`${getApiBase()}/api/extract/bit`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const data = (await response.json()) as {
      bits: number[];
      bit_string: string;
    };
    extractBitOutput.value = data.bit_string || data.bits.join("");
    setStatus(extractBitStatus, "Recovered", "good");
  } catch (err) {
    const message = err instanceof Error ? err.message : "Extract failed";
    setStatus(extractBitStatus, message, "bad");
  }
});

copyBit.addEventListener("click", async () => {
  if (!extractBitOutput.value) {
    setStatus(extractBitStatus, "Nothing to copy", "bad");
    return;
  }
  try {
    await navigator.clipboard.writeText(extractBitOutput.value);
    setStatus(extractBitStatus, "Copied to clipboard", "good");
  } catch {
    setStatus(extractBitStatus, "Copy failed", "bad");
  }
});

attackType.addEventListener("change", updateAttackOptions);
updateAttackOptions();

recoverTemplate.addEventListener("change", () => {
  const file = recoverTemplate.files?.[0];
  if (!file) {
    cropperState.img = null;
    cropperState.selection = null;
    renderCropper();
    return;
  }
  void loadCropperFromFile(file);
});

recoverCropCanvas.addEventListener("pointerdown", (event) => {
  if (!cropperState.img) {
    return;
  }
  recoverCropCanvas.setPointerCapture(event.pointerId);
  cropperState.dragging = true;
  const startCanvas = getCanvasPoint(event);
  cropperState.start = canvasToImage(startCanvas);
  cropperState.selection = { x: cropperState.start.x, y: cropperState.start.y, w: 1, h: 1 };
  renderCropper();
});

recoverCropCanvas.addEventListener("pointermove", (event) => {
  if (!cropperState.dragging || !cropperState.start || !cropperState.img) {
    return;
  }
  const point = canvasToImage(getCanvasPoint(event));
  const img = cropperState.img;
  const x1 = clamp(Math.min(cropperState.start.x, point.x), 0, img.width);
  const y1 = clamp(Math.min(cropperState.start.y, point.y), 0, img.height);
  const x2 = clamp(Math.max(cropperState.start.x, point.x), 0, img.width);
  const y2 = clamp(Math.max(cropperState.start.y, point.y), 0, img.height);
  cropperState.selection = { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
  renderCropper();
});

function stopCropDrag(): void {
  cropperState.dragging = false;
  cropperState.start = null;
}

recoverCropCanvas.addEventListener("pointerup", () => {
  stopCropDrag();
});

recoverCropCanvas.addEventListener("pointercancel", () => {
  stopCropDrag();
});

recoverCropCanvas.addEventListener("dblclick", () => {
  cropperState.selection = null;
  renderCropper();
});

recoverCropReset.addEventListener("click", () => {
  cropperState.selection = null;
  renderCropper();
});

recoverCropApply.addEventListener("click", async () => {
  const img = cropperState.img;
  const selection = cropperState.selection;
  if (!img || !selection) {
    setStatus(recoverStatus, "Select a crop area first", "bad");
    return;
  }
  if (selection.w < 4 || selection.h < 4) {
    setStatus(recoverStatus, "Crop area is too small", "bad");
    return;
  }
  const x = clamp(Math.round(selection.x), 0, img.width - 1);
  const y = clamp(Math.round(selection.y), 0, img.height - 1);
  const w = clamp(Math.round(selection.w), 1, img.width - x);
  const h = clamp(Math.round(selection.h), 1, img.height - y);

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    setStatus(recoverStatus, "Canvas not available", "bad");
    return;
  }
  ctx.drawImage(img, x, y, w, h, 0, 0, w, h);
  const dataUrl = canvas.toDataURL("image/png");
  try {
    await setFileInputFromDataUrl(recoverTemplate, dataUrl, "cropped.png");
    setStatus(recoverStatus, "Cropped image loaded", "good");
  } catch {
    setStatus(recoverStatus, "Failed to apply crop", "bad");
  }
});

window.addEventListener("resize", () => {
  if (cropperState.img) {
    renderCropper();
  }
});

renderCropper();

attackForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(attackStatus, "Attacking...", "loading");
  clearImageResult(attackPreview, attackDownload);
  attackOutShape.textContent = "-";
  attackLoc.textContent = "-";
  attackScaleValue.textContent = "-";

  const formData = new FormData(attackForm);
  appendOutputOptions(formData);

  try {
    const response = await fetch(`${getApiBase()}/api/attack`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const data = (await response.json()) as {
      image_base64: string;
      mime_type: string;
      attack: string;
      out_shape: { height: number; width: number };
      loc?: { x1: number; y1: number; x2: number; y2: number };
      scale?: number;
    };
    const src = `data:${data.mime_type};base64,${data.image_base64}`;
    const filename = data.mime_type === "image/png" ? "attacked.png" : "attacked.jpg";
    showImageResult(attackPreview, attackDownload, src, filename);
    attackOutShape.textContent = `${data.out_shape.width} x ${data.out_shape.height}`;
    attackLoc.textContent = data.loc ? formatLoc(data.loc) : "-";
    attackScaleValue.textContent =
      data.scale === null || data.scale === undefined ? "-" : data.scale.toFixed(3);
    lastAttack = { dataUrl: src, filename, mimeType: data.mime_type };

    setStatus(attackStatus, `Done (${data.attack})`, "good");
  } catch (err) {
    const message = err instanceof Error ? err.message : "Attack failed";
    setStatus(attackStatus, message, "bad");
  }
});

attackToText.addEventListener("click", async () => {
  await useAttackOnInput(extractTextImage, "text extract");
});

attackToImage.addEventListener("click", async () => {
  await useAttackOnInput(extractImageInput, "image extract");
});

attackToBit.addEventListener("click", async () => {
  await useAttackOnInput(extractBitInput, "bit extract");
});

attackToRecover.addEventListener("click", async () => {
  await useAttackOnInput(recoverTemplate, "recovery");
});

recoverForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(recoverStatus, "Recovering...", "loading");
  clearImageResult(recoverPreview, recoverDownload);
  recoverLoc.textContent = "-";
  recoverScale.textContent = "-";
  recoverScore.textContent = "-";
  recoverTextOutput.value = "";

  const formData = buildRecoverFormData();

  try {
    const response = await fetch(`${getApiBase()}/api/recover/crop`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const data = (await response.json()) as RecoverResponse;
    applyRecoverResult(data);
    setStatus(recoverStatus, "Recovered", "good");
  } catch (err) {
    const message = err instanceof Error ? err.message : "Recovery failed";
    setStatus(recoverStatus, message, "bad");
  }
});

recoverExtractText.addEventListener("click", async () => {
  const wmSizeValue = recoverTextWmSize.value.trim() || extractWmSize.value.trim();
  if (!wmSizeValue) {
    setStatus(recoverStatus, "wm_size is required", "bad");
    return;
  }
  setStatus(recoverStatus, "Recovering + extracting...", "loading");
  clearImageResult(recoverPreview, recoverDownload);
  recoverLoc.textContent = "-";
  recoverScale.textContent = "-";
  recoverScore.textContent = "-";
  recoverTextOutput.value = "";

  const formData = buildRecoverFormData("png");

  try {
    const response = await fetch(`${getApiBase()}/api/recover/crop`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    const data = (await response.json()) as RecoverResponse;
    const recoveredUrl = applyRecoverResult(data);
    const text = await extractTextFromDataUrl(recoveredUrl, wmSizeValue);
    recoverTextOutput.value = text;
    setStatus(recoverStatus, "Recovered + extracted", "good");
  } catch (err) {
    const message = err instanceof Error ? err.message : "Recover + extract failed";
    setStatus(recoverStatus, message, "bad");
  }
});

copyRecoverText.addEventListener("click", async () => {
  if (!recoverTextOutput.value) {
    setStatus(recoverStatus, "Nothing to copy", "bad");
    return;
  }
  try {
    await navigator.clipboard.writeText(recoverTextOutput.value);
    setStatus(recoverStatus, "Copied to clipboard", "good");
  } catch {
    setStatus(recoverStatus, "Copy failed", "bad");
  }
});
