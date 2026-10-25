/**
 * Typed interfaces strictly reflecting the DocuMind AI FastAPI backend schemas.
 */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
}

export interface DocumentMetadata {
  document_id: string;
  filename: string;
  file_size_bytes: number;
  total_pages: number;
  total_chunks: number;
  created_at: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  status: string;
  total_pages: number;
  total_chunks: number;
  created_at: string;
}

export interface DocumentListResponse {
  documents: DocumentMetadata[];
  total_count: number;
}

export interface DocumentDeleteResponse {
  success: boolean;
  message: string;
  document_id: string;
}

export interface CitationSource {
  source_id: number;
  document_id: string;
  filename: string;
  page_numbers: number[];
  excerpt: string;
  similarity_score?: number | null;
}

export interface ChatQueryRequest {
  question: string;
  document_ids?: string[] | null;
  top_k?: number | null;
}

export interface ChatQueryResponse {
  answer: string;
  sources: CitationSource[];
  insufficient_evidence: boolean;
  model_used?: string | null;
  request_id: string;
}

export interface ApiErrorResponse {
  success?: boolean;
  error?: string;
  detail?: string | Array<{ msg: string; loc: string[] }>;
  details?: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: CitationSource[];
  insufficient_evidence?: boolean;
  model_used?: string | null;
  timestamp: string;
  isError?: boolean;
}
