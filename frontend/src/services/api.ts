/**
 * Centralized API client for DocuMind AI FastAPI Backend.
 */

import {
  ChatQueryRequest,
  ChatQueryResponse,
  DocumentDeleteResponse,
  DocumentListResponse,
  DocumentMetadata,
  DocumentUploadResponse,
  HealthResponse,
} from "../types";

const BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";
const API_PREFIX = "/api/v1";

export class ApiError extends Error {
  public statusCode: number;
  public details?: Record<string, unknown>;

  constructor(message: string, statusCode: number, details?: Record<string, unknown>) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
    this.details = details;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP Error ${response.status}: ${response.statusText}`;
    let errorDetails: Record<string, unknown> | undefined;

    try {
      const errJson = await response.json();
      if (errJson.error) {
        errorMessage = errJson.error;
      } else if (typeof errJson.detail === "string") {
        errorMessage = errJson.detail;
      } else if (Array.isArray(errJson.detail) && errJson.detail.length > 0) {
        errorMessage = errJson.detail.map((e: { msg: string }) => e.msg).join(", ");
      }
      errorDetails = errJson.details;
    } catch {
      // Body not JSON
    }

    throw new ApiError(errorMessage, response.status, errorDetails);
  }

  return response.json() as Promise<T>;
}

export const apiClient = {
  /** Check backend health status */
  async checkHealth(): Promise<HealthResponse> {
    const res = await fetch(`${BASE_URL}${API_PREFIX}/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    return handleResponse<HealthResponse>(res);
  },

  /** Upload and index a PDF document */
  async uploadDocument(file: File): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${BASE_URL}${API_PREFIX}/documents/upload`, {
      method: "POST",
      body: formData,
      headers: { Accept: "application/json" },
    });
    return handleResponse<DocumentUploadResponse>(res);
  },

  /** List all indexed documents */
  async listDocuments(): Promise<DocumentListResponse> {
    const res = await fetch(`${BASE_URL}${API_PREFIX}/documents`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    return handleResponse<DocumentListResponse>(res);
  },

  /** Get metadata for a specific document ID */
  async getDocument(docId: string): Promise<DocumentMetadata> {
    const res = await fetch(`${BASE_URL}${API_PREFIX}/documents/${encodeURIComponent(docId)}`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    return handleResponse<DocumentMetadata>(res);
  },

  /** Delete a document and its vectors from ChromaDB */
  async deleteDocument(docId: string): Promise<DocumentDeleteResponse> {
    const res = await fetch(`${BASE_URL}${API_PREFIX}/documents/${encodeURIComponent(docId)}`, {
      method: "DELETE",
      headers: { Accept: "application/json" },
    });
    return handleResponse<DocumentDeleteResponse>(res);
  },

  /** Query indexed documents with grounded RAG */
  async queryChat(request: ChatQueryRequest): Promise<ChatQueryResponse> {
    const res = await fetch(`${BASE_URL}${API_PREFIX}/chat/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(request),
    });
    return handleResponse<ChatQueryResponse>(res);
  },
};
