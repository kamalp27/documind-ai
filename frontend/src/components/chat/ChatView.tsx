import React, { useEffect, useRef, useState } from "react";
import {
  Bot,
  CornerDownLeft,
  Filter,
  HelpCircle,
  Info,
  Layers,
  Send,
  Sparkles,
  User,
} from "lucide-react";
import { apiClient, ApiError } from "../../services/api";
import { ChatMessage, CitationSource, DocumentMetadata } from "../../types";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { MarkdownMessage } from "./MarkdownMessage";

interface ChatViewProps {
  documents: DocumentMetadata[];
  selectedDocId: string | null;
  onSelectDoc: (id: string | null) => void;
  onOpenCitation: (source: CitationSource) => void;
}

export const ChatView: React.FC<ChatViewProps> = ({
  documents,
  selectedDocId,
  onSelectDoc,
  onOpenCitation,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const selectedDoc = documents.find((d) => d.document_id === selectedDocId);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isSubmitting]);

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const query = inputValue.trim();
    if (!query || isSubmitting) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsSubmitting(true);

    try {
      const response = await apiClient.queryChat({
        question: query,
        document_ids: selectedDocId ? [selectedDocId] : null,
      });

      const assistantMessage: ChatMessage = {
        id: response.request_id || `bot-${Date.now()}`,
        role: "assistant",
        content: response.answer,
        sources: response.sources,
        insufficient_evidence: response.insufficient_evidence,
        model_used: response.model_used,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      let errText = "Failed to communicate with DocuMind backend.";
      if (err instanceof ApiError) {
        errText = err.message;
      }
      const errorMessage: ChatMessage = {
        id: `err-${Date.now()}`,
        role: "assistant",
        content: errText,
        isError: true,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handlePromptSuggestion = (promptText: string) => {
    setInputValue(promptText);
    textareaRef.current?.focus();
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        backgroundColor: "var(--bg-secondary)",
        borderRadius: "var(--radius-lg)",
        border: "1px solid var(--border-subtle)",
        overflow: "hidden",
      }}
    >
      {/* Scope / Filter Banner */}
      <div
        style={{
          padding: "0.75rem 1.25rem",
          backgroundColor: "var(--bg-surface)",
          borderBottom: "1px solid var(--border-subtle)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          fontSize: "0.8125rem",
          color: "var(--text-secondary)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Filter size={15} style={{ color: "var(--accent-primary)" }} />
          <span>Searching scope:</span>
          {selectedDoc ? (
            <Badge variant="accent" size="sm">
              {selectedDoc.filename}
            </Badge>
          ) : (
            <Badge variant="default" size="sm">
              All Indexed Documents ({documents.length})
            </Badge>
          )}
        </div>

        {selectedDocId && (
          <button
            onClick={() => onSelectDoc(null)}
            style={{
              fontSize: "0.75rem",
              color: "var(--accent-primary)",
              textDecoration: "underline",
              cursor: "pointer",
            }}
          >
            Switch to All Documents
          </button>
        )}
      </div>

      {/* Messages Scroll Area */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "1.5rem",
          display: "flex",
          flexDirection: "column",
          gap: "1.5rem",
        }}
      >
        {/* Welcome Empty State */}
        {messages.length === 0 && (
          <div
            style={{
              margin: "auto",
              maxWidth: "540px",
              textAlign: "center",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "1.25rem",
              padding: "2rem 1rem",
            }}
          >
            <div
              style={{
                width: "3.5rem",
                height: "3.5rem",
                borderRadius: "var(--radius-lg)",
                backgroundColor: "var(--accent-muted)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "var(--accent-primary)",
                boxShadow: "var(--shadow-glow)",
              }}
            >
              <Sparkles size={28} />
            </div>

            <div>
              <h2 style={{ fontSize: "1.375rem", fontWeight: 700, color: "var(--text-primary)" }}>
                Ask anything about your documents
              </h2>
              <p style={{ fontSize: "0.875rem", color: "var(--text-muted)", marginTop: "0.35rem", lineHeight: 1.6 }}>
                DocuMind AI performs semantic retrieval over your indexed PDFs and synthesizes factual answers with page-level citations.
              </p>
            </div>

            {/* Starter Prompt Suggestions */}
            {documents.length > 0 ? (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", width: "100%", marginTop: "0.5rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Suggested queries
                </span>
                {[
                  "Summarize the key information in this document.",
                  "What are the main topics and details covered?",
                  "What skills, facts, or important details are mentioned?",
                ].map((prompt, idx) => (
                  <button
                    key={idx}
                    onClick={() => handlePromptSuggestion(prompt)}
                    style={{
                      padding: "0.75rem 1rem",
                      backgroundColor: "var(--bg-surface)",
                      border: "1px solid var(--border-subtle)",
                      borderRadius: "var(--radius-md)",
                      textAlign: "left",
                      fontSize: "0.875rem",
                      color: "var(--text-secondary)",
                      transition: "all var(--transition-fast)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = "var(--border-focus)";
                      e.currentTarget.style.color = "var(--text-primary)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = "var(--border-subtle)";
                      e.currentTarget.style.color = "var(--text-secondary)";
                    }}
                  >
                    <span>{prompt}</span>
                    <CornerDownLeft size={14} style={{ color: "var(--text-muted)" }} />
                  </button>
                ))}
              </div>
            ) : (
              <div
                style={{
                  padding: "1rem",
                  backgroundColor: "rgba(245, 158, 11, 0.08)",
                  border: "1px solid rgba(245, 158, 11, 0.2)",
                  borderRadius: "var(--radius-md)",
                  fontSize: "0.8125rem",
                  color: "var(--status-warning)",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                }}
              >
                <Info size={16} />
                <span>Upload a PDF document first to enable grounded question answering.</span>
              </div>
            )}
          </div>
        )}

        {/* Message Feed */}
        {messages.map((msg) => (
          <div
            key={msg.id}
            style={{
              display: "flex",
              gap: "1rem",
              alignItems: "flex-start",
              maxWidth: "850px",
              width: "100%",
              alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
              flexDirection: msg.role === "user" ? "row-reverse" : "row",
            }}
          >
            {/* Avatar */}
            <div
              style={{
                width: "2.25rem",
                height: "2.25rem",
                borderRadius: "var(--radius-md)",
                backgroundColor: msg.role === "user" ? "var(--bg-surface-elevated)" : "var(--accent-muted)",
                color: msg.role === "user" ? "var(--text-primary)" : "var(--accent-primary)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
              }}
            >
              {msg.role === "user" ? <User size={16} /> : <Bot size={18} />}
            </div>

            {/* Bubble */}
            <div
              style={{
                maxWidth: "720px",
                backgroundColor:
                  msg.role === "user"
                    ? "var(--accent-gradient)"
                    : msg.isError
                    ? "var(--status-error-bg)"
                    : "var(--bg-surface)",
                color: msg.role === "user" ? "var(--text-inverse)" : "var(--text-primary)",
                border: `1px solid ${
                  msg.isError
                    ? "rgba(244, 63, 94, 0.3)"
                    : msg.role === "user"
                    ? "transparent"
                    : "var(--border-subtle)"
                }`,
                borderRadius: "var(--radius-lg)",
                padding: "1rem 1.25rem",
                boxShadow: "var(--shadow-sm)",
                display: "flex",
                flexDirection: "column",
                gap: "0.75rem",
              }}
            >
              {msg.role === "assistant" && !msg.isError ? (
                <MarkdownMessage
                  content={msg.content}
                  sources={msg.sources}
                  onOpenCitation={onOpenCitation}
                />
              ) : (
                <div style={{ fontSize: "0.9375rem", lineHeight: 1.6, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                  {msg.content}
                </div>
              )}

              {/* Insufficient Evidence Notice */}
              {msg.insufficient_evidence && (
                <div
                  style={{
                    padding: "0.625rem 0.875rem",
                    backgroundColor: "rgba(245, 158, 11, 0.1)",
                    border: "1px solid rgba(245, 158, 11, 0.25)",
                    borderRadius: "var(--radius-sm)",
                    fontSize: "0.8125rem",
                    color: "var(--status-warning)",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem",
                  }}
                >
                  <HelpCircle size={15} style={{ flexShrink: 0 }} />
                  <span>
                    No document passages exceeded the semantic relevance threshold. The model withheld speculation.
                  </span>
                </div>
              )}

              {/* Citations Badges Row */}
              {msg.sources && msg.sources.length > 0 && (
                <div
                  style={{
                    paddingTop: "0.75rem",
                    borderTop: "1px solid var(--border-subtle)",
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.5rem",
                  }}
                >
                  <span
                    style={{
                      fontSize: "0.75rem",
                      fontWeight: 600,
                      color: "var(--text-muted)",
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                    }}
                  >
                    Verified Sources ({msg.sources.length})
                  </span>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                    {msg.sources.map((src) => (
                      <button
                        key={src.source_id}
                        onClick={() => onOpenCitation(src)}
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "0.35rem",
                          padding: "0.3rem 0.65rem",
                          backgroundColor: "var(--bg-secondary)",
                          border: "1px solid var(--border-focus)",
                          borderRadius: "var(--radius-full)",
                          fontSize: "0.75rem",
                          color: "var(--accent-primary)",
                          transition: "all var(--transition-fast)",
                          cursor: "pointer",
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = "var(--accent-muted)";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = "var(--bg-secondary)";
                        }}
                        title={`Click to view verified excerpt from ${src.filename} (Page ${src.page_numbers.join(", ")})`}
                      >
                        <Layers size={12} />
                        <span>[{src.source_id}] {src.filename} (p. {src.page_numbers.join(", ")})</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Message Footer */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  fontSize: "0.6875rem",
                  color: msg.role === "user" ? "rgba(0, 0, 0, 0.6)" : "var(--text-muted)",
                }}
              >
                <span>{msg.timestamp}</span>
                {msg.model_used && (
                  <span style={{ fontFamily: "var(--font-mono)" }}>
                    {msg.model_used}
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}

        {/* Loading Indicator */}
        {isSubmitting && (
          <div
            style={{
              display: "flex",
              gap: "1rem",
              alignItems: "flex-start",
              maxWidth: "720px",
            }}
          >
            <div
              style={{
                width: "2.25rem",
                height: "2.25rem",
                borderRadius: "var(--radius-md)",
                backgroundColor: "var(--accent-muted)",
                color: "var(--accent-primary)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Bot size={18} />
            </div>
            <div
              style={{
                padding: "1rem 1.25rem",
                backgroundColor: "var(--bg-surface)",
                borderRadius: "var(--radius-lg)",
                border: "1px solid var(--border-subtle)",
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
                fontSize: "0.875rem",
                color: "var(--text-secondary)",
              }}
            >
              <span
                style={{
                  width: "0.875rem",
                  height: "0.875rem",
                  border: "2px solid var(--accent-primary)",
                  borderRightColor: "transparent",
                  borderRadius: "50%",
                  animation: "spin 0.6s linear infinite",
                }}
              />
              <span>Retrieving passages & synthesizing grounded answer...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Box Footer */}
      <form
        onSubmit={handleSubmit}
        style={{
          padding: "1rem 1.25rem",
          backgroundColor: "var(--bg-surface)",
          borderTop: "1px solid var(--border-subtle)",
          display: "flex",
          gap: "0.75rem",
          alignItems: "flex-end",
        }}
      >
        <div style={{ flex: 1, position: "relative" }}>
          <textarea
            ref={textareaRef}
            rows={2}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              documents.length === 0
                ? "Upload a document above to begin asking questions..."
                : selectedDoc
                ? `Ask a question about ${selectedDoc.filename}... (Press Enter to send)`
                : "Ask a question about your indexed documents... (Press Enter to send)"
            }
            maxLength={1000}
            disabled={isSubmitting}
            style={{
              width: "100%",
              backgroundColor: "var(--bg-primary)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "var(--radius-md)",
              padding: "0.75rem 1rem",
              fontSize: "0.875rem",
              color: "var(--text-primary)",
              resize: "none",
              outline: "none",
              transition: "border-color var(--transition-fast)",
            }}
            onFocus={(e) => (e.target.style.borderColor = "var(--border-focus)")}
            onBlur={(e) => (e.target.style.borderColor = "var(--border-subtle)")}
          />
        </div>

        <Button
          type="submit"
          variant="primary"
          size="md"
          disabled={!inputValue.trim() || isSubmitting}
          isLoading={isSubmitting}
          rightIcon={<Send size={15} />}
        >
          Send
        </Button>
      </form>
    </div>
  );
};
