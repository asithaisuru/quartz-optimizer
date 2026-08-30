const DEFAULT_PDF_FILENAME = "quartz-analysis.pdf";

export const resolveBackendUrl = (path, apiBaseUrl) => {
  if (!path) return null;
  if (/^https?:\/\//i.test(path)) return path;
  const base = String(apiBaseUrl || "").replace(/\/+$/, "");
  const suffix = String(path).replace(/^\/+/, "");
  return `${base}/${suffix}`;
};

export const sanitizePdfFilename = (value) => {
  const withoutControlCharacters = Array.from(
    String(value || DEFAULT_PDF_FILENAME),
  )
    .filter((character) => {
      const code = character.charCodeAt(0);
      return code >= 32 && code !== 127;
    })
    .join("");
  const cleaned = withoutControlCharacters
    .replace(/[/\\?%*:|"<>]/g, "-")
    .trim()
    .slice(0, 180);
  const filename = cleaned || DEFAULT_PDF_FILENAME;
  return filename.toLowerCase().endsWith(".pdf")
    ? filename
    : `${filename}.pdf`;
};

const responseFilename = (response, fallback) => {
  const header = response.headers?.get?.("content-disposition") || "";
  const encoded = header.match(/filename\*=UTF-8''([^;]+)/i);
  if (encoded) {
    try {
      return sanitizePdfFilename(decodeURIComponent(encoded[1]));
    } catch {
      return sanitizePdfFilename(fallback);
    }
  }
  const quoted = header.match(/filename="?([^";]+)"?/i);
  return sanitizePdfFilename(quoted?.[1] || fallback);
};

const errorMessage = async (response) => {
  try {
    const contentType = response.headers?.get?.("content-type") || "";
    if (contentType.includes("application/json")) {
      const body = await response.json();
      return body?.detail || body?.error || body?.message;
    }
    const body = await response.text();
    return body?.trim();
  } catch {
    return null;
  }
};

export const downloadPdf = async ({
  url,
  filename,
  fetchImpl = fetch,
  documentRef = document,
  urlApi = URL,
}) => {
  if (!url) throw new Error("The PDF report is not available yet.");

  let response;
  try {
    response = await fetchImpl(url, {
      method: "GET",
      headers: { Accept: "application/pdf" },
      cache: "no-store",
    });
  } catch {
    throw new Error("Could not reach the backend. Check that it is running.");
  }

  if (!response.ok) {
    const detail = await errorMessage(response);
    throw new Error(detail || `PDF download failed (${response.status}).`);
  }

  const contentType = response.headers?.get?.("content-type") || "";
  if (!contentType.toLowerCase().includes("application/pdf")) {
    throw new Error("The backend returned a non-PDF response.");
  }

  const blob = await response.blob();
  if (!blob.size) throw new Error("The downloaded PDF is empty.");
  const signature = new Uint8Array(await blob.slice(0, 5).arrayBuffer());
  if (String.fromCharCode(...signature) !== "%PDF-") {
    throw new Error("The downloaded file does not have a valid PDF signature.");
  }

  const objectUrl = urlApi.createObjectURL(blob);
  const anchor = documentRef.createElement("a");
  try {
    anchor.href = objectUrl;
    anchor.download = responseFilename(response, filename);
    anchor.hidden = true;
    documentRef.body.appendChild(anchor);
    anchor.click();
  } finally {
    anchor.remove();
    urlApi.revokeObjectURL(objectUrl);
  }

  return { filename: anchor.download, size: blob.size };
};
