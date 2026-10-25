import React from "react";
import {
  FileText,
  FolderOpen,
  MessageSquare,
  PlusCircle,
  ShieldCheck,
  Upload,
} from "lucide-react";
import { DocumentMetadata } from "../../types";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";

interface SidebarProps {
  currentTab: "chat" | "documents";
  onTabChange: (tab: "chat" | "documents") => void;
  documents: DocumentMetadata[];
  selectedDocId: string | null;
  onSelectDoc: (id: string | null) => void;
  onOpenUpload: () => void;
  onNewChat: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onTabChange,
  documents,
  selectedDocId,
  onSelectDoc,
  onOpenUpload,
  onNewChat,
}) => {
  return (
    <aside
      style={{
        width: "280px",
        backgroundColor: "var(--bg-secondary)",
        borderRight: "1px solid var(--border-subtle)",
        display: "flex",
        flexDirection: "column",
        height: "calc(100vh - 64px)",
        padding: "1.25rem 1rem",
        gap: "1.5rem",
        flexShrink: 0,
      }}
    >
      {/* Top Action Buttons */}
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        <Button
          variant="primary"
          size="md"
          onClick={() => {
            onNewChat();
            onTabChange("chat");
          }}
          leftIcon={<PlusCircle size={16} />}
        >
          New Conversation
        </Button>

        <Button
          variant="secondary"
          size="md"
          onClick={onOpenUpload}
          leftIcon={<Upload size={16} />}
        >
          Upload PDF
        </Button>
      </div>

      {/* Primary Navigation Tabs */}
      <nav style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
        <span
          style={{
            fontSize: "0.6875rem",
            fontWeight: 600,
            color: "var(--text-muted)",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
            paddingLeft: "0.5rem",
            marginBottom: "0.25rem",
          }}
        >
          Workspace Navigation
        </span>

        <button
          onClick={() => onTabChange("chat")}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
            padding: "0.65rem 0.75rem",
            borderRadius: "var(--radius-md)",
            fontSize: "0.875rem",
            fontWeight: 500,
            color: currentTab === "chat" ? "var(--text-primary)" : "var(--text-secondary)",
            backgroundColor: currentTab === "chat" ? "var(--bg-surface-elevated)" : "transparent",
            border: `1px solid ${currentTab === "chat" ? "var(--border-focus)" : "transparent"}`,
            transition: "all var(--transition-fast)",
            cursor: "pointer",
            width: "100%",
            textAlign: "left",
          }}
        >
          <MessageSquare size={17} style={{ color: currentTab === "chat" ? "var(--accent-primary)" : "var(--text-muted)" }} />
          <span>Chat Assistant</span>
        </button>

        <button
          onClick={() => onTabChange("documents")}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0.65rem 0.75rem",
            borderRadius: "var(--radius-md)",
            fontSize: "0.875rem",
            fontWeight: 500,
            color: currentTab === "documents" ? "var(--text-primary)" : "var(--text-secondary)",
            backgroundColor: currentTab === "documents" ? "var(--bg-surface-elevated)" : "transparent",
            border: `1px solid ${currentTab === "documents" ? "var(--border-focus)" : "transparent"}`,
            transition: "all var(--transition-fast)",
            cursor: "pointer",
            width: "100%",
            textAlign: "left",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <FolderOpen size={17} style={{ color: currentTab === "documents" ? "var(--accent-primary)" : "var(--text-muted)" }} />
            <span>Document Library</span>
          </div>
          <Badge variant="default" size="sm">
            {documents.length}
          </Badge>
        </button>
      </nav>

      {/* Quick Document Scope Filter */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            paddingLeft: "0.5rem",
            marginBottom: "0.5rem",
          }}
        >
          <span
            style={{
              fontSize: "0.6875rem",
              fontWeight: 600,
              color: "var(--text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Active Documents ({documents.length})
          </span>
          {selectedDocId && (
            <button
              onClick={() => onSelectDoc(null)}
              style={{
                fontSize: "0.6875rem",
                color: "var(--accent-primary)",
                cursor: "pointer",
              }}
            >
              Reset
            </button>
          )}
        </div>

        <div
          style={{
            flex: 1,
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: "0.25rem",
            paddingRight: "0.25rem",
          }}
        >
          {documents.length === 0 ? (
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", padding: "0.5rem", textAlign: "center" }}>
              No PDFs uploaded yet.
            </div>
          ) : (
            documents.map((doc) => {
              const isSelected = selectedDocId === doc.document_id;
              return (
                <button
                  key={doc.document_id}
                  onClick={() => onSelectDoc(isSelected ? null : doc.document_id)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem",
                    padding: "0.5rem 0.6rem",
                    borderRadius: "var(--radius-sm)",
                    fontSize: "0.8125rem",
                    color: isSelected ? "var(--accent-primary)" : "var(--text-secondary)",
                    backgroundColor: isSelected ? "var(--accent-muted)" : "transparent",
                    border: "1px solid transparent",
                    borderColor: isSelected ? "var(--border-focus)" : "transparent",
                    textAlign: "left",
                    cursor: "pointer",
                    width: "100%",
                  }}
                  title={doc.filename}
                >
                  <FileText size={14} style={{ flexShrink: 0 }} />
                  <span
                    style={{
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      flex: 1,
                    }}
                  >
                    {doc.filename}
                  </span>
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Architecture Badges Footer */}
      <div
        style={{
          padding: "0.875rem",
          backgroundColor: "var(--bg-surface)",
          borderRadius: "var(--radius-md)",
          border: "1px solid var(--border-subtle)",
          display: "flex",
          flexDirection: "column",
          gap: "0.4rem",
          fontSize: "0.6875rem",
          color: "var(--text-muted)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", color: "var(--text-secondary)", fontWeight: 600 }}>
          <ShieldCheck size={14} style={{ color: "var(--accent-primary)" }} />
          <span>Production Pipeline</span>
        </div>
        <div>• Embeddings: all-MiniLM-L6-v2</div>
        <div>• Vector Store: ChromaDB (Local)</div>
        <div>• Generation: Google Gemini API</div>
      </div>
    </aside>
  );
};
