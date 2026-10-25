import React, { useState } from "react";
import { Check, Copy } from "lucide-react";
import { CitationSource } from "../../types";

interface MarkdownMessageProps {
  content: string;
  sources?: CitationSource[];
  onOpenCitation?: (source: CitationSource) => void;
}

interface BlockElement {
  type: "heading" | "paragraph" | "ul" | "ol" | "codeblock";
  level?: number;
  text?: string;
  items?: string[];
  lang?: string;
  code?: string;
}

export const MarkdownMessage: React.FC<MarkdownMessageProps> = ({
  content,
  sources = [],
  onOpenCitation,
}) => {
  const [copiedCodeIdx, setCopiedCodeIdx] = useState<number | null>(null);

  const handleCopyCode = async (code: string, idx: number) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedCodeIdx(idx);
      setTimeout(() => setCopiedCodeIdx(null), 2000);
    } catch {
      // Fallback if clipboard API is restricted
    }
  };

  /** Parse inline Markdown tokens: bold, italic, code, and inline citations */
  const parseInlineTokens = (rawText: string): React.ReactNode[] => {
    if (!rawText) return [];

    // Split by inline code first: `...`
    const codeParts = rawText.split(/(`[^`]+`)/g);

    return codeParts.flatMap((part, partIdx) => {
      // Inline code
      if (part.startsWith("`") && part.endsWith("`") && part.length >= 2) {
        return (
          <code
            key={`code-${partIdx}`}
            style={{
              padding: "0.15rem 0.4rem",
              margin: "0 0.15rem",
              borderRadius: "var(--radius-sm)",
              backgroundColor: "var(--bg-primary)",
              border: "1px solid var(--border-subtle)",
              fontFamily: "var(--font-mono)",
              fontSize: "0.85em",
              color: "var(--accent-primary)",
            }}
          >
            {part.slice(1, -1)}
          </code>
        );
      }

      // Regex to detect bold, italic, and inline citations: [1], [2], [Source 1], etc.
      const inlineRegex = /(\*\*[^*]+\*\*|\*[^*]+\*|\[(?:Source\s*)?(\d+)\])/gi;
      const elements: React.ReactNode[] = [];
      let lastIndex = 0;
      let match: RegExpExecArray | null;

      while ((match = inlineRegex.exec(part)) !== null) {
        if (match.index > lastIndex) {
          elements.push(part.slice(lastIndex, match.index));
        }

        const matchedStr = match[0];

        // Bold: **text**
        if (matchedStr.startsWith("**") && matchedStr.endsWith("**") && matchedStr.length > 4) {
          elements.push(
            <strong key={`bold-${partIdx}-${match.index}`} style={{ fontWeight: 600, color: "var(--text-primary)" }}>
              {matchedStr.slice(2, -2)}
            </strong>
          );
        }
        // Italic: *text*
        else if (matchedStr.startsWith("*") && matchedStr.endsWith("*") && matchedStr.length > 2) {
          elements.push(
            <em key={`italic-${partIdx}-${match.index}`} style={{ fontStyle: "italic", color: "var(--text-secondary)" }}>
              {matchedStr.slice(1, -1)}
            </em>
          );
        }
        // Citation: [1], [Source 1], etc.
        else if (match[2]) {
          const sourceNum = parseInt(match[2], 10);
          const foundSource = sources.find((s) => s.source_id === sourceNum);

          if (foundSource && onOpenCitation) {
            elements.push(
              <button
                key={`citation-${partIdx}-${match.index}`}
                type="button"
                onClick={() => onOpenCitation(foundSource)}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  padding: "0.1rem 0.45rem",
                  margin: "0 0.15rem",
                  borderRadius: "var(--radius-full)",
                  backgroundColor: "rgba(56, 189, 248, 0.15)",
                  border: "1px solid var(--border-focus)",
                  color: "var(--accent-primary)",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  cursor: "pointer",
                  lineHeight: "1.2",
                  verticalAlign: "baseline",
                  transition: "all var(--transition-fast)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = "rgba(56, 189, 248, 0.28)";
                  e.currentTarget.style.boxShadow = "var(--shadow-glow)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "rgba(56, 189, 248, 0.15)";
                  e.currentTarget.style.boxShadow = "none";
                }}
                title={`View citation from ${foundSource.filename} (Page ${foundSource.page_numbers.join(", ")})`}
                aria-label={`View citation [${sourceNum}] from ${foundSource.filename}, page ${foundSource.page_numbers.join(", ")}`}
              >
                {matchedStr}
              </button>
            );
          } else {
            // Unmatched citation: render as ordinary text
            elements.push(matchedStr);
          }
        } else {
          elements.push(matchedStr);
        }

        lastIndex = match.index + matchedStr.length;
      }

      if (lastIndex < part.length) {
        elements.push(part.slice(lastIndex));
      }

      return elements;
    });
  };

  /** Parse block-level Markdown elements (codeblocks, headings, lists, paragraphs) */
  const parseBlocks = (markdown: string): BlockElement[] => {
    const lines = markdown.split(/\r?\n/);
    const blocks: BlockElement[] = [];
    let inCodeBlock = false;
    let codeBlockLang = "";
    let codeBlockLines: string[] = [];
    let currentList: { type: "ul" | "ol"; items: string[] } | null = null;

    const flushList = () => {
      if (currentList) {
        blocks.push(currentList);
        currentList = null;
      }
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Code block boundary: ```lang or ```
      const codeFenceMatch = line.match(/^```(\w+)?/);
      if (codeFenceMatch) {
        if (inCodeBlock) {
          blocks.push({
            type: "codeblock",
            lang: codeBlockLang,
            code: codeBlockLines.join("\n"),
          });
          inCodeBlock = false;
          codeBlockLang = "";
          codeBlockLines = [];
        } else {
          flushList();
          inCodeBlock = true;
          codeBlockLang = codeFenceMatch[1] || "";
          codeBlockLines = [];
        }
        continue;
      }

      if (inCodeBlock) {
        codeBlockLines.push(line);
        continue;
      }

      // Empty line separates blocks/paragraphs
      if (!line.trim()) {
        flushList();
        continue;
      }

      // Headings: #, ##, ###, ####
      const headingMatch = line.match(/^(#{1,4})\s+(.+)$/);
      if (headingMatch) {
        flushList();
        blocks.push({
          type: "heading",
          level: headingMatch[1].length,
          text: headingMatch[2].trim(),
        });
        continue;
      }

      // Bulleted list item: - or *
      const bulletMatch = line.match(/^[-*]\s+(.+)$/);
      if (bulletMatch) {
        if (!currentList || currentList.type !== "ul") {
          flushList();
          currentList = { type: "ul", items: [] };
        }
        currentList.items.push(bulletMatch[1].trim());
        continue;
      }

      // Numbered list item: 1. 2. etc.
      const numberMatch = line.match(/^(\d+)\.\s+(.+)$/);
      if (numberMatch) {
        if (!currentList || currentList.type !== "ol") {
          flushList();
          currentList = { type: "ol", items: [] };
        }
        currentList.items.push(numberMatch[2].trim());
        continue;
      }

      // Regular paragraph line
      flushList();
      blocks.push({
        type: "paragraph",
        text: line.trim(),
      });
    }

    if (inCodeBlock) {
      blocks.push({
        type: "codeblock",
        lang: codeBlockLang,
        code: codeBlockLines.join("\n"),
      });
    }
    flushList();

    return blocks;
  };

  const blocks = parseBlocks(content);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "0.65rem",
        fontSize: "0.9375rem",
        lineHeight: 1.6,
        color: "var(--text-primary)",
      }}
    >
      {blocks.map((block, idx) => {
        if (block.type === "codeblock") {
          return (
            <div
              key={`block-${idx}`}
              style={{
                position: "relative",
                borderRadius: "var(--radius-md)",
                backgroundColor: "var(--bg-primary)",
                border: "1px solid var(--border-subtle)",
                overflow: "hidden",
                margin: "0.4rem 0",
              }}
            >
              {/* Header bar with language & copy button */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "0.35rem 0.75rem",
                  backgroundColor: "rgba(255, 255, 255, 0.03)",
                  borderBottom: "1px solid var(--border-subtle)",
                  fontSize: "0.75rem",
                  color: "var(--text-muted)",
                }}
              >
                <span style={{ letterSpacing: "0.05em", fontFamily: "var(--font-mono)" }}>
                  {(block.lang || "code").toUpperCase()}
                </span>
                <button
                  type="button"
                  onClick={() => handleCopyCode(block.code || "", idx)}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "0.3rem",
                    color: copiedCodeIdx === idx ? "var(--status-success)" : "var(--text-secondary)",
                    fontSize: "0.75rem",
                    cursor: "pointer",
                    padding: "0.2rem 0.4rem",
                    borderRadius: "var(--radius-sm)",
                    transition: "color var(--transition-fast)",
                  }}
                  title="Copy code to clipboard"
                  aria-label="Copy code to clipboard"
                >
                  {copiedCodeIdx === idx ? <Check size={13} /> : <Copy size={13} />}
                  <span>{copiedCodeIdx === idx ? "Copied" : "Copy"}</span>
                </button>
              </div>

              {/* Code content */}
              <pre
                style={{
                  margin: 0,
                  padding: "0.85rem 1rem",
                  overflowX: "auto",
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.8125rem",
                  lineHeight: 1.5,
                  color: "#e2e8f0",
                }}
              >
                <code>{block.code}</code>
              </pre>
            </div>
          );
        }

        if (block.type === "heading") {
          const Tag = (
            block.level === 1 ? "h1" : block.level === 2 ? "h2" : block.level === 3 ? "h3" : "h4"
          ) as keyof JSX.IntrinsicElements;

          const size =
            block.level === 1 ? "1.25rem" : block.level === 2 ? "1.125rem" : block.level === 3 ? "1rem" : "0.9375rem";

          return (
            <Tag
              key={`block-${idx}`}
              style={{
                fontSize: size,
                fontWeight: 700,
                color: "var(--text-primary)",
                margin: "0.5rem 0 0.15rem",
                letterSpacing: "-0.01em",
              }}
            >
              {parseInlineTokens(block.text || "")}
            </Tag>
          );
        }

        if (block.type === "ul") {
          return (
            <ul
              key={`block-${idx}`}
              style={{
                paddingLeft: "1.25rem",
                margin: "0.25rem 0",
                display: "flex",
                flexDirection: "column",
                gap: "0.35rem",
              }}
            >
              {block.items?.map((item, itemIdx) => (
                <li key={`li-${itemIdx}`} style={{ color: "var(--text-primary)" }}>
                  {parseInlineTokens(item)}
                </li>
              ))}
            </ul>
          );
        }

        if (block.type === "ol") {
          return (
            <ol
              key={`block-${idx}`}
              style={{
                paddingLeft: "1.25rem",
                margin: "0.25rem 0",
                display: "flex",
                flexDirection: "column",
                gap: "0.35rem",
              }}
            >
              {block.items?.map((item, itemIdx) => (
                <li key={`oli-${itemIdx}`} style={{ color: "var(--text-primary)" }}>
                  {parseInlineTokens(item)}
                </li>
              ))}
            </ol>
          );
        }

        // Paragraph
        return (
          <p key={`block-${idx}`} style={{ margin: 0, wordBreak: "break-word" }}>
            {parseInlineTokens(block.text || "")}
          </p>
        );
      })}
    </div>
  );
};
