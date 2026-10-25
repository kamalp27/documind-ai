import React, { useState } from "react";
import {
  AlertTriangle,
  Calendar,
  FileText,
  Layers,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { DocumentMetadata } from "../../types";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { Modal } from "../ui/Modal";

interface DocumentListProps {
  documents: DocumentMetadata[];
  isLoading: boolean;
  onRefresh: () => void;
  onDelete: (docId: string) => Promise<void>;
  selectedDocId: string | null;
  onSelectDoc: (docId: string | null) => void;
}

export const DocumentList: React.FC<DocumentListProps> = ({
  documents,
  isLoading,
  onRefresh,
  onDelete,
  selectedDocId,
  onSelectDoc,
}) => {
  const [docToDelete, setDocToDelete] = useState<DocumentMetadata | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (isoString: string): string => {
    try {
      const date = new Date(isoString);
      return date.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
      });
    } catch {
      return isoString;
    }
  };

  const handleConfirmDelete = async () => {
    if (!docToDelete) return;
    setIsDeleting(true);
    try {
      await onDelete(docToDelete.document_id);
      if (selectedDocId === docToDelete.document_id) {
        onSelectDoc(null);
      }
      setDocToDelete(null);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
      {/* Header bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "1rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <h2 style={{ fontSize: "1.125rem", fontWeight: 700, color: "var(--text-primary)" }}>
            Indexed Documents
          </h2>
          <Badge variant="default" size="sm">
            {documents.length}
          </Badge>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          {selectedDocId && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onSelectDoc(null)}
            >
              Clear Filter
            </Button>
          )}
          <Button
            variant="secondary"
            size="sm"
            onClick={onRefresh}
            isLoading={isLoading}
            leftIcon={<RefreshCw size={14} />}
            aria-label="Refresh document library"
          >
            Refresh
          </Button>
        </div>
      </div>

      {/* Loading Skeleton */}
      {isLoading && documents.length === 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          {[1, 2].map((n) => (
            <div
              key={n}
              style={{
                height: "80px",
                backgroundColor: "var(--bg-surface)",
                borderRadius: "var(--radius-md)",
                border: "1px solid var(--border-subtle)",
                animation: "pulse 1.5s infinite",
              }}
            />
          ))}
        </div>
      )}

      {/* Empty State */}
      {!isLoading && documents.length === 0 && (
        <div
          style={{
            padding: "3rem 1.5rem",
            backgroundColor: "var(--bg-surface)",
            border: "1px dashed var(--border-subtle)",
            borderRadius: "var(--radius-lg)",
            textAlign: "center",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "0.75rem",
          }}
        >
          <FileText size={32} style={{ color: "var(--text-muted)" }} />
          <div>
            <p style={{ fontWeight: 600, color: "var(--text-primary)" }}>No documents indexed yet</p>
            <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
              Upload a PDF document above to extract its text and generate ChromaDB vector embeddings.
            </p>
          </div>
        </div>
      )}

      {/* Document Cards Grid */}
      <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
        {documents.map((doc) => {
          const isSelected = selectedDocId === doc.document_id;
          return (
            <div
              key={doc.document_id}
              style={{
                padding: "1rem 1.25rem",
                backgroundColor: isSelected ? "var(--bg-surface-elevated)" : "var(--bg-surface)",
                border: `1px solid ${
                  isSelected ? "var(--border-focus)" : "var(--border-subtle)"
                }`,
                borderRadius: "var(--radius-md)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "1rem",
                transition: "all var(--transition-fast)",
                boxShadow: isSelected ? "var(--shadow-glow)" : "none",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "1rem",
                  flex: 1,
                  minWidth: 0,
                  cursor: "pointer",
                }}
                onClick={() => onSelectDoc(isSelected ? null : doc.document_id)}
                title="Click to toggle document filter in chat"
              >
                <div
                  style={{
                    width: "2.5rem",
                    height: "2.5rem",
                    borderRadius: "var(--radius-md)",
                    backgroundColor: isSelected ? "var(--accent-muted)" : "rgba(255, 255, 255, 0.04)",
                    color: isSelected ? "var(--accent-primary)" : "var(--text-secondary)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                  }}
                >
                  <FileText size={18} />
                </div>

                <div style={{ minWidth: 0, flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <h3
                      style={{
                        fontSize: "0.9375rem",
                        fontWeight: 600,
                        color: isSelected ? "var(--accent-primary)" : "var(--text-primary)",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {doc.filename}
                    </h3>
                    {isSelected && (
                      <Badge variant="accent" size="sm">
                        Active Filter
                      </Badge>
                    )}
                  </div>

                  <div
                    style={{
                      display: "flex",
                      flexWrap: "wrap",
                      alignItems: "center",
                      gap: "0.875rem",
                      fontSize: "0.75rem",
                      color: "var(--text-muted)",
                      marginTop: "0.25rem",
                    }}
                  >
                    <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                      <Layers size={13} /> {doc.total_pages} {doc.total_pages === 1 ? "page" : "pages"} ({doc.total_chunks} chunks)
                    </span>
                    <span>•</span>
                    <span>{formatFileSize(doc.file_size_bytes)}</span>
                    <span>•</span>
                    <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                      <Calendar size={13} /> {formatDate(doc.created_at)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexShrink: 0 }}>
                <Button
                  variant={isSelected ? "primary" : "secondary"}
                  size="sm"
                  onClick={() => onSelectDoc(isSelected ? null : doc.document_id)}
                >
                  {isSelected ? "Filter Active" : "Filter Chat"}
                </Button>
                <Button
                  variant="danger"
                  size="sm"
                  onClick={() => setDocToDelete(doc)}
                  aria-label={`Delete ${doc.filename}`}
                >
                  <Trash2 size={15} />
                </Button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Confirmation Modal */}
      <Modal
        isOpen={docToDelete !== null}
        onClose={() => !isDeleting && setDocToDelete(null)}
        title="Confirm Document Deletion"
      >
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: "0.75rem" }}>
            <AlertTriangle size={22} style={{ color: "var(--status-error)", flexShrink: 0 }} />
            <div>
              <p style={{ fontSize: "0.9375rem", color: "var(--text-primary)", fontWeight: 500 }}>
                Are you sure you want to permanently delete <strong>"{docToDelete?.filename}"</strong>?
              </p>
              <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
                This action will permanently purge all {docToDelete?.total_chunks} vector embeddings from ChromaDB and remove its metadata. This cannot be undone.
              </p>
            </div>
          </div>

          <div
            style={{
              display: "flex",
              justifyContent: "flex-end",
              gap: "0.75rem",
              marginTop: "0.5rem",
            }}
          >
            <Button
              variant="secondary"
              size="md"
              onClick={() => setDocToDelete(null)}
              disabled={isDeleting}
            >
              Cancel
            </Button>
            <Button
              variant="danger"
              size="md"
              onClick={handleConfirmDelete}
              isLoading={isDeleting}
            >
              Confirm Deletion
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
};
