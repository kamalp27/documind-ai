import React from "react";
import { Brain, Menu } from "lucide-react";
import { Badge } from "../ui/Badge";

interface HeaderProps {
  isBackendHealthy: boolean | null;
  backendVersion: string | null;
  onToggleMobileMenu?: () => void;
  documentCount: number;
}

export const Header: React.FC<HeaderProps> = ({
  isBackendHealthy,
  backendVersion,
  onToggleMobileMenu,
  documentCount,
}) => {
  return (
    <header
      style={{
        height: "64px",
        backgroundColor: "var(--bg-secondary)",
        borderBottom: "1px solid var(--border-subtle)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 1.5rem",
        zIndex: 50,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        {onToggleMobileMenu && (
          <button
            onClick={onToggleMobileMenu}
            style={{
              color: "var(--text-secondary)",
              padding: "0.4rem",
              borderRadius: "var(--radius-sm)",
              display: "flex",
              alignItems: "center",
            }}
            aria-label="Toggle navigation menu"
          >
            <Menu size={20} />
          </button>
        )}

        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div
            style={{
              width: "2.25rem",
              height: "2.25rem",
              borderRadius: "var(--radius-md)",
              background: "var(--accent-gradient)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--text-inverse)",
              boxShadow: "var(--shadow-glow)",
            }}
          >
            <Brain size={18} />
          </div>

          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <h1 style={{ fontSize: "1.125rem", fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
                DocuMind <span style={{ color: "var(--accent-primary)" }}>AI</span>
              </h1>
              <Badge variant="accent" size="sm">
                RAG Engine
              </Badge>
            </div>
            <p style={{ fontSize: "0.6875rem", color: "var(--text-muted)" }}>
              Document Intelligence & Verifiable Q&A
            </p>
          </div>
        </div>
      </div>

      {/* Status & Metrics */}
      <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <div
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              backgroundColor:
                isBackendHealthy === true
                  ? "var(--status-success)"
                  : isBackendHealthy === false
                  ? "var(--status-error)"
                  : "var(--status-warning)",
              boxShadow:
                isBackendHealthy === true
                  ? "0 0 8px var(--status-success)"
                  : "none",
            }}
          />
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            {isBackendHealthy === true
              ? `API Online (${backendVersion || "v0.3"})`
              : isBackendHealthy === false
              ? "API Offline"
              : "Connecting..."}
          </span>
        </div>

        <div style={{ height: "1.25rem", width: "1px", backgroundColor: "var(--border-subtle)" }} />

        <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
          <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{documentCount}</span> {documentCount === 1 ? "doc" : "docs"} indexed
        </div>
      </div>
    </header>
  );
};
