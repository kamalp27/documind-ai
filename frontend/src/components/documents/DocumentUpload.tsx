import React, { useRef, useState } from "react";
import { AlertCircle, CheckCircle2, FileUp, UploadCloud } from "lucide-react";
import { apiClient, ApiError } from "../../services/api";
import { DocumentUploadResponse } from "../../types";
import { Button } from "../ui/Button";

interface DocumentUploadProps {
  onUploadSuccess: (doc: DocumentUploadResponse) => void;
  maxSizeMB?: number;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onUploadSuccess,
  maxSizeMB = 10,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successInfo, setSuccessInfo] = useState<DocumentUploadResponse | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const maxSizeBytes = maxSizeMB * 1024 * 1024;

  const validateAndUpload = async (file: File) => {
    setErrorMessage(null);
    setSuccessInfo(null);

    // Client-side validation: extension
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setErrorMessage("Unsupported file type. Please select a valid PDF file.");
      return;
    }

    // Client-side validation: empty file
    if (file.size === 0) {
      setErrorMessage("Uploaded file is empty (0 bytes).");
      return;
    }

    // Client-side validation: max size
    if (file.size > maxSizeBytes) {
      setErrorMessage(`File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds the maximum allowed limit of ${maxSizeMB} MB.`);
      return;
    }

    setIsUploading(true);
    try {
      const response = await apiClient.uploadDocument(file);
      setSuccessInfo(response);
      onUploadSuccess(response);
    } catch (err) {
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage("Failed to upload document. Please check backend connection.");
      }
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!isUploading) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (isUploading) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndUpload(e.target.files[0]);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isUploading && fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${
            isDragging
              ? "var(--accent-primary)"
              : errorMessage
              ? "var(--status-error)"
              : "var(--border-subtle)"
          }`,
          backgroundColor: isDragging
            ? "var(--accent-muted)"
            : "var(--bg-surface)",
          borderRadius: "var(--radius-lg)",
          padding: "2.5rem 1.5rem",
          textAlign: "center",
          cursor: isUploading ? "wait" : "pointer",
          transition: "all var(--transition-fast)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "1rem",
        }}
        role="button"
        tabIndex={0}
        aria-label="Upload PDF document dropzone"
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            fileInputRef.current?.click();
          }
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          style={{ display: "none" }}
          onChange={handleFileChange}
          disabled={isUploading}
          data-testid="file-input"
        />

        <div
          style={{
            width: "3.25rem",
            height: "3.25rem",
            borderRadius: "var(--radius-full)",
            backgroundColor: "rgba(56, 189, 248, 0.1)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--accent-primary)",
          }}
        >
          {isUploading ? (
            <FileUp size={24} style={{ animation: "pulse 1.5s infinite" }} />
          ) : (
            <UploadCloud size={24} />
          )}
        </div>

        <div>
          <p style={{ fontWeight: 600, fontSize: "1rem", color: "var(--text-primary)", marginBottom: "0.25rem" }}>
            {isUploading ? "Extracting & Indexing PDF..." : "Click or drag PDF here to upload"}
          </p>
          <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)" }}>
            PDF documents up to {maxSizeMB} MB. Pages and chunks are extracted and indexed automatically.
          </p>
        </div>

        <Button
          variant="secondary"
          size="sm"
          isLoading={isUploading}
          disabled={isUploading}
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            fileInputRef.current?.click();
          }}
        >
          {isUploading ? "Indexing vectors..." : "Browse Local File"}
        </Button>
      </div>

      {/* Success Notification */}
      {successInfo && (
        <div
          style={{
            padding: "0.875rem 1rem",
            backgroundColor: "var(--status-success-bg)",
            border: "1px solid rgba(16, 185, 129, 0.3)",
            borderRadius: "var(--radius-md)",
            display: "flex",
            alignItems: "flex-start",
            gap: "0.75rem",
            fontSize: "0.875rem",
            color: "var(--text-primary)",
          }}
        >
          <CheckCircle2 size={18} style={{ color: "var(--status-success)", marginTop: "0.1rem", flexShrink: 0 }} />
          <div>
            <strong>"{successInfo.filename}" successfully indexed!</strong>
            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "0.2rem" }}>
              Extracted {successInfo.total_pages} {successInfo.total_pages === 1 ? "page" : "pages"} and created {successInfo.total_chunks} vector chunks.
            </div>
          </div>
        </div>
      )}

      {/* Error Callout */}
      {errorMessage && (
        <div
          style={{
            padding: "0.875rem 1rem",
            backgroundColor: "var(--status-error-bg)",
            border: "1px solid rgba(244, 63, 94, 0.3)",
            borderRadius: "var(--radius-md)",
            display: "flex",
            alignItems: "flex-start",
            gap: "0.75rem",
            fontSize: "0.875rem",
            color: "var(--text-primary)",
          }}
        >
          <AlertCircle size={18} style={{ color: "var(--status-error)", marginTop: "0.1rem", flexShrink: 0 }} />
          <div>
            <strong>Upload Failed:</strong>
            <div style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", marginTop: "0.2rem" }}>
              {errorMessage}
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.6; transform: scale(0.95); }
        }
      `}</style>
    </div>
  );
};
