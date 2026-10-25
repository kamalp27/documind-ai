import React, { useState } from "react";
import { Check, Copy, FileText, Layers, X } from "lucide-react";
import { CitationSource } from "../../types";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";

interface CitationDrawerProps {
  source: CitationSource | null;
  onClose: () => void;
}

export const CitationDrawer: React.FC<CitationDrawerProps> = ({ source, onClose }) => {
  const [copied, setCopied] = useState(false);

  if (!source) return null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(source.excerpt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard write failed
    }
  };

  const similarityPercentage =
    source.similarity_score != null
      ? `${Math.round(source.similarity_score * 100)}%`
      : null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 90,
        backgroundColor: "rgba(7, 9, 14, 0.6)",
        backdropFilter: "blur(4px)",
        display: "flex",
        justifyContent: "flex-end",
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: "100%",
          maxWidth: "460px",
          height: "100%",
          backgroundColor: "var(--bg-secondary)",
          borderLeft: "1px solid var(--border-subtle)",
          boxShadow: "var(--shadow-lg)",
          display: "flex",
          flexDirection: "column",
          animation: "slideIn 0.25s cubic-bezier(0.16, 1, 0.3, 1)",
        }}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Source citation preview"
      >
        {/* Drawer Header */}
        <div
          style={{
            padding: "1.25rem 1.5rem",
            borderBottom: "1px solid var(--border-subtle)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "1rem",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Badge variant="accent" size="md">
              Source [{source.source_id}]
            </Badge>
            {similarityPercentage && (
              <Badge variant="success" size="sm">
                {similarityPercentage} Match
              </Badge>
            )}
          </div>
          <button
            onClick={onClose}
            style={{
              color: "var(--text-muted)",
              padding: "0.25rem",
              borderRadius: "var(--radius-sm)",
              display: "flex",
              alignItems: "center",
            }}
            aria-label="Close citation drawer"
          >
            <X size={20} />
          </button>
        </div>

        {/* Drawer Content */}
        <div style={{ padding: "1.5rem", flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          {/* Metadata Card */}
          <div
            style={{
              padding: "1rem",
              backgroundColor: "var(--bg-surface)",
              borderRadius: "var(--radius-md)",
              border: "1px solid var(--border-subtle)",
              display: "flex",
              flexDirection: "column",
              gap: "0.75rem",
            }}
          >
            <div style={{ display: "flex", alignItems: "flex-start", gap: "0.75rem" }}>
              <FileText size={18} style={{ color: "var(--accent-primary)", marginTop: "0.15rem", flexShrink: 0 }} />
              <div>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Source Document
                </div>
                <div style={{ fontSize: "0.9375rem", fontWeight: 600, color: "var(--text-primary)", wordBreak: "break-word" }}>
                  {source.filename}
                </div>
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "1.5rem", paddingTop: "0.5rem", borderTop: "1px solid var(--border-subtle)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Layers size={15} style={{ color: "var(--text-muted)" }} />
                <span style={{ fontSize: "0.8125rem", color: "var(--text-secondary)" }}>
                  Page {source.page_numbers.join(", ")}
                </span>
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                ID: {source.document_id.slice(0, 8)}...
              </div>
            </div>
          </div>

          {/* Excerpt Section */}
          <div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: "0.5rem",
              }}
            >
              <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Verified Passage Excerpt
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCopy}
                leftIcon={copied ? <Check size={14} style={{ color: "var(--status-success)" }} /> : <Copy size={14} />}
              >
                {copied ? "Copied" : "Copy"}
              </Button>
            </div>

            <div
              style={{
                padding: "1.125rem",
                backgroundColor: "var(--bg-primary)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-md)",
                borderLeft: "3px solid var(--accent-primary)",
                fontSize: "0.875rem",
                lineHeight: 1.6,
                color: "var(--text-primary)",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}
            >
              "{source.excerpt}"
            </div>
          </div>

          {/* Traceability Note */}
          <div
            style={{
              padding: "0.75rem 1rem",
              backgroundColor: "rgba(56, 189, 248, 0.05)",
              border: "1px solid rgba(56, 189, 248, 0.15)",
              borderRadius: "var(--radius-md)",
              fontSize: "0.75rem",
              color: "var(--text-secondary)",
              lineHeight: 1.5,
            }}
          >
            <strong>Grounded Verification:</strong> This passage was retrieved directly from the vector database using cosine similarity over extracted PDF pages.
          </div>
        </div>
      </div>
      <style>{`
        @keyframes slideIn {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
      `}</style>
    </div>
  );
};
