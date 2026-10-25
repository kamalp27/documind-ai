import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DocumentUpload } from "../components/documents/DocumentUpload";
import { apiClient } from "../services/api";

describe("DocumentUpload", () => {
  it("renders drag-and-drop area and browse button", () => {
    render(<DocumentUpload onUploadSuccess={vi.fn()} />);
    expect(screen.getByText(/Click or drag PDF here to upload/i)).toBeInTheDocument();
    expect(screen.getByText(/Browse Local File/i)).toBeInTheDocument();
  });

  it("rejects non-PDF files client-side without calling API", async () => {
    const uploadSpy = vi.spyOn(apiClient, "uploadDocument");
    render(<DocumentUpload onUploadSuccess={vi.fn()} />);

    const input = screen.getByTestId("file-input");
    const textFile = new File(["not a pdf"], "notes.txt", { type: "text/plain" });

    fireEvent.change(input, { target: { files: [textFile] } });

    await waitFor(() => {
      expect(screen.getByText(/Unsupported file type/i)).toBeInTheDocument();
    });
    expect(uploadSpy).not.toHaveBeenCalled();
  });

  it("rejects empty files (0 bytes)", async () => {
    render(<DocumentUpload onUploadSuccess={vi.fn()} />);

    const input = screen.getByTestId("file-input");
    const emptyFile = new File([], "empty.pdf", { type: "application/pdf" });

    fireEvent.change(input, { target: { files: [emptyFile] } });

    await waitFor(() => {
      expect(screen.getByText(/Uploaded file is empty/i)).toBeInTheDocument();
    });
  });

  it("calls uploadDocument on valid PDF and triggers onUploadSuccess", async () => {
    const mockSuccess = {
      document_id: "doc-abc",
      filename: "valid.pdf",
      status: "processed",
      total_pages: 2,
      total_chunks: 5,
      created_at: "2026-10-25T18:00:00Z",
    };

    vi.spyOn(apiClient, "uploadDocument").mockResolvedValueOnce(mockSuccess);
    const successCallback = vi.fn();

    render(<DocumentUpload onUploadSuccess={successCallback} />);

    const input = screen.getByTestId("file-input");
    const validFile = new File(["%PDF-1.4 sample content"], "valid.pdf", {
      type: "application/pdf",
    });

    fireEvent.change(input, { target: { files: [validFile] } });

    await waitFor(() => {
      expect(screen.getByText(/"valid.pdf" successfully indexed!/i)).toBeInTheDocument();
    });
    expect(successCallback).toHaveBeenCalledWith(mockSuccess);
  });
});
