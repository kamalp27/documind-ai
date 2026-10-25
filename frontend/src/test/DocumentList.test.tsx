import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DocumentList } from "../components/documents/DocumentList";
import { DocumentMetadata } from "../types";

const sampleDocs: DocumentMetadata[] = [
  {
    document_id: "doc-1",
    filename: "ai_whitepaper.pdf",
    file_size_bytes: 204800,
    total_pages: 5,
    total_chunks: 14,
    created_at: "2026-10-25T12:00:00Z",
  },
  {
    document_id: "doc-2",
    filename: "system_architecture.pdf",
    file_size_bytes: 512000,
    total_pages: 12,
    total_chunks: 32,
    created_at: "2026-10-25T13:00:00Z",
  },
];

describe("DocumentList", () => {
  it("renders empty state when no documents exist", () => {
    render(
      <DocumentList
        documents={[]}
        isLoading={false}
        onRefresh={vi.fn()}
        onDelete={vi.fn()}
        selectedDocId={null}
        onSelectDoc={vi.fn()}
      />
    );
    expect(screen.getByText(/No documents indexed yet/i)).toBeInTheDocument();
  });

  it("renders list of documents with metadata", () => {
    render(
      <DocumentList
        documents={sampleDocs}
        isLoading={false}
        onRefresh={vi.fn()}
        onDelete={vi.fn()}
        selectedDocId={null}
        onSelectDoc={vi.fn()}
      />
    );
    expect(screen.getByText("ai_whitepaper.pdf")).toBeInTheDocument();
    expect(screen.getByText("system_architecture.pdf")).toBeInTheDocument();
    expect(screen.getByText(/5 pages \(14 chunks\)/i)).toBeInTheDocument();
  });

  it("opens confirmation modal on delete click and calls onDelete upon confirmation", async () => {
    const deleteSpy = vi.fn().mockResolvedValue(undefined);
    render(
      <DocumentList
        documents={sampleDocs}
        isLoading={false}
        onRefresh={vi.fn()}
        onDelete={deleteSpy}
        selectedDocId={null}
        onSelectDoc={vi.fn()}
      />
    );

    const deleteButtons = screen.getAllByRole("button", { name: /Delete/i });
    fireEvent.click(deleteButtons[0]);

    // Modal appears
    expect(screen.getByText(/Confirm Document Deletion/i)).toBeInTheDocument();
    expect(screen.getByText(/Are you sure you want to permanently delete/i)).toBeInTheDocument();

    // Confirm
    const confirmBtn = screen.getByRole("button", { name: /Confirm Deletion/i });
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      expect(deleteSpy).toHaveBeenCalledWith("doc-1");
    });
  });
});
