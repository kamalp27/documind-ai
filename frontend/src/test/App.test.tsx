import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { App } from "../App";
import { apiClient } from "../services/api";

vi.mock("../services/api", () => ({
  apiClient: {
    checkHealth: vi.fn(),
    listDocuments: vi.fn(),
    uploadDocument: vi.fn(),
    deleteDocument: vi.fn(),
    queryChat: vi.fn(),
  },
}));

describe("App Integration", () => {
  const mockDocs = [
    {
      document_id: "doc-1",
      filename: "ai_handbook.pdf",
      file_size_bytes: 10240,
      total_pages: 5,
      total_chunks: 10,
      created_at: "2026-10-25T12:00:00Z",
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.checkHealth).mockResolvedValue({
      status: "healthy",
      service: "DocuMind AI",
      version: "0.3.0",
      environment: "test",
    });
    vi.mocked(apiClient.listDocuments).mockResolvedValue({
      documents: mockDocs,
      total_count: 1,
    });
  });

  it("loads health status and documents on initial mount", async () => {
    render(<App />);

    // Header displays service name
    expect(screen.getByText(/DocuMind AI/i)).toBeInTheDocument();

    // Check health call
    await waitFor(() => {
      expect(apiClient.checkHealth).toHaveBeenCalled();
      expect(apiClient.listDocuments).toHaveBeenCalled();
    });

    // Check document count displayed in header badge or sidebar
    await waitFor(() => {
      expect(screen.getByText("ai_handbook.pdf")).toBeInTheDocument();
    });
  });

  it("switches to Documents tab and opens upload modal", async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("ai_handbook.pdf")).toBeInTheDocument();
    });

    // Click "Document Library" in sidebar
    const docButton = screen.getByRole("button", { name: /Document Library/i });
    fireEvent.click(docButton);

    // Check Indexed Documents title
    expect(screen.getByText(/Indexed Documents/i)).toBeInTheDocument();

    // Click "Upload PDF" button in sidebar
    const uploadBtns = screen.getAllByRole("button", { name: /Upload PDF/i });
    fireEvent.click(uploadBtns[0]);

    // Modal should appear
    expect(screen.getByText(/Upload PDF Document/i)).toBeInTheDocument();
  });
});
