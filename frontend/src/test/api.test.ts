import { describe, expect, it, vi, beforeEach } from "vitest";
import { apiClient } from "../services/api";

describe("apiClient", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("checkHealth returns healthy status", async () => {
    const mockHealth = {
      status: "healthy",
      service: "DocuMind AI",
      version: "0.3.0",
      environment: "development",
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth,
    } as Response);

    const result = await apiClient.checkHealth();
    expect(result.status).toBe("healthy");
    expect(result.service).toBe("DocuMind AI");
  });

  it("uploadDocument passes file via multipart form data", async () => {
    const mockUpload = {
      document_id: "doc-123",
      filename: "test.pdf",
      status: "processed",
      total_pages: 3,
      total_chunks: 6,
      created_at: "2026-10-25T18:00:00Z",
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockUpload,
    } as Response);

    const file = new File(["dummy pdf content"], "test.pdf", { type: "application/pdf" });
    const result = await apiClient.uploadDocument(file);

    expect(result.document_id).toBe("doc-123");
    expect(result.total_pages).toBe(3);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/documents/upload"),
      expect.objectContaining({ method: "POST" })
    );
  });

  it("handles API error responses and throws ApiError", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      statusText: "Bad Request",
      json: async () => ({
        success: false,
        error: "Uploaded file is empty (0 bytes).",
      }),
    } as Response);

    const file = new File([], "empty.pdf", { type: "application/pdf" });
    await expect(apiClient.uploadDocument(file)).rejects.toThrow("Uploaded file is empty (0 bytes).");
  });

  it("queryChat dispatches formatted request payload", async () => {
    const mockResponse = {
      answer: "DocuMind uses ChromaDB [1].",
      sources: [
        {
          source_id: 1,
          document_id: "doc-123",
          filename: "test.pdf",
          page_numbers: [1],
          excerpt: "ChromaDB vector persistence",
          similarity_score: 0.9,
        },
      ],
      insufficient_evidence: false,
      model_used: "gemini-2.5-flash",
      request_id: "req-abc",
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    } as Response);

    const result = await apiClient.queryChat({
      question: "How does vector search work?",
      document_ids: ["doc-123"],
    });

    expect(result.answer).toContain("DocuMind uses ChromaDB [1].");
    expect(result.sources).toHaveLength(1);
    expect(result.insufficient_evidence).toBe(false);
  });
});
