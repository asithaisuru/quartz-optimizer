import assert from "node:assert/strict";
import test from "node:test";
import {
  downloadPdf,
  resolveBackendUrl,
  sanitizePdfFilename,
} from "../src/utils/pdfDownload.js";

const pdfBytes = new TextEncoder().encode("%PDF-1.7\nsynthetic");

const response = ({
  ok = true,
  status = 200,
  contentType = "application/pdf",
  disposition = 'attachment; filename="server-report.pdf"',
  body = pdfBytes,
} = {}) => ({
  ok,
  status,
  headers: {
    get(name) {
      if (name.toLowerCase() === "content-type") return contentType;
      if (name.toLowerCase() === "content-disposition") return disposition;
      return null;
    },
  },
  async blob() {
    return new Blob([body], { type: contentType });
  },
  async json() {
    return { detail: "Report not found." };
  },
  async text() {
    return "Report not found.";
  },
});

const browserHarness = () => {
  const events = [];
  const anchor = {
    href: "",
    download: "",
    hidden: false,
    click() {
      events.push("click");
    },
    remove() {
      events.push("remove");
    },
  };
  return {
    events,
    anchor,
    documentRef: {
      createElement(name) {
        assert.equal(name, "a");
        return anchor;
      },
      body: {
        appendChild(value) {
          assert.equal(value, anchor);
          events.push("append");
        },
      },
    },
    urlApi: {
      createObjectURL(blob) {
        assert.ok(blob.size > 0);
        events.push("create");
        return "blob:pdf-test";
      },
      revokeObjectURL(value) {
        assert.equal(value, "blob:pdf-test");
        events.push("revoke");
      },
    },
  };
};

test("resolves relative report URLs against the backend", () => {
  assert.equal(
    resolveBackendUrl("/api/jobs/abc/report/pdf", "http://localhost:8000/"),
    "http://localhost:8000/api/jobs/abc/report/pdf",
  );
});

test("sanitizes attachment filenames", () => {
  assert.equal(sanitizePdfFilename("../bad:name"), "..-bad-name.pdf");
});

test("downloads a PDF Blob and cleans up the object URL", async () => {
  const harness = browserHarness();
  const result = await downloadPdf({
    url: "http://localhost:8000/api/jobs/id/report/pdf",
    filename: "fallback.pdf",
    fetchImpl: async () => response(),
    documentRef: harness.documentRef,
    urlApi: harness.urlApi,
  });
  assert.equal(result.filename, "server-report.pdf");
  assert.deepEqual(
    harness.events,
    ["create", "append", "click", "remove", "revoke"],
  );
});

test("rejects JSON errors without creating a Blob URL", async () => {
  const harness = browserHarness();
  await assert.rejects(
    downloadPdf({
      url: "http://localhost:8000/api/jobs/id/report/pdf",
      fetchImpl: async () =>
        response({ ok: false, status: 404, contentType: "application/json" }),
      documentRef: harness.documentRef,
      urlApi: harness.urlApi,
    }),
    /Report not found/,
  );
  assert.deepEqual(harness.events, []);
});

test("rejects a non-PDF success response", async () => {
  const harness = browserHarness();
  await assert.rejects(
    downloadPdf({
      url: "http://localhost:8000/api/jobs/id/report/pdf",
      fetchImpl: async () => response({ contentType: "text/html" }),
      documentRef: harness.documentRef,
      urlApi: harness.urlApi,
    }),
    /non-PDF/,
  );
  assert.deepEqual(harness.events, []);
});
