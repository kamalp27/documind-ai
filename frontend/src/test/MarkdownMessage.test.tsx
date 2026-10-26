import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MarkdownMessage } from "../components/chat/MarkdownMessage";
import { CitationSource } from "../types";

const mockSources: CitationSource[] = [
  {
    source_id: 1,
    document_id: "doc-alpha",
    filename: "system_architecture.pdf",
    page_numbers: [1],
    excerpt: "System overview of the decoupled RAG architecture.",
    similarity_score: 0.95,
  },
  {
    source_id: 2,
    document_id: "doc-beta",
    filename: "performance_spec.pdf",
    page_numbers: [4, 5],
    excerpt: "Evaluations demonstrated 99.8% precision.",
    similarity_score: 0.88,
  },
];

describe("MarkdownMessage Component", () => {
  it("renders headings at various levels", () => {
    const content = "# Main Heading\n## Subheading Two\n### Subheading Three";
    render(<MarkdownMessage content={content} />);

    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Main Heading");
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent("Subheading Two");
    expect(screen.getByRole("heading", { level: 3 })).toHaveTextContent("Subheading Three");
  });

  it("renders bold and italic formatting", () => {
    const content = "This is **critical information** and this is *emphasized note*.";
    render(<MarkdownMessage content={content} />);

    const boldEl = screen.getByText("critical information");
    expect(boldEl.tagName.toLowerCase()).toBe("strong");

    const italicEl = screen.getByText("emphasized note");
    expect(italicEl.tagName.toLowerCase()).toBe("em");
  });

  it("renders bulleted and numbered lists", () => {
    const content = "- First bullet\n- Second bullet\n\n1. Numbered one\n2. Numbered two";
    render(<MarkdownMessage content={content} />);

    expect(screen.getByText("First bullet")).toBeInTheDocument();
    expect(screen.getByText("Second bullet")).toBeInTheDocument();
    expect(screen.getByText("Numbered one")).toBeInTheDocument();
    expect(screen.getByText("Numbered two")).toBeInTheDocument();
  });

  it("renders inline code and fenced code blocks with language and copy button", () => {
    const content = "Use the `get_settings()` function.\n\n```python\ndef hello():\n    return 'world'\n```";
    render(<MarkdownMessage content={content} />);

    // Inline code
    const inlineCode = screen.getByText("get_settings()");
    expect(inlineCode.tagName.toLowerCase()).toBe("code");

    // Code block
    expect(screen.getByText("PYTHON")).toBeInTheDocument();
    expect(screen.getByText(/def hello\(\):/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Copy code to clipboard/i })).toBeInTheDocument();
  });

  it("makes inline citations interactive when matching actual sources", () => {
    const onOpenCitation = vi.fn();
    const content = "The architecture uses microservices [1] and scales automatically [Source 2].";
    render(
      <MarkdownMessage
        content={content}
        sources={mockSources}
        onOpenCitation={onOpenCitation}
      />
    );

    // [1] button
    const cite1Btn = screen.getByRole("button", { name: /View citation \[1\] from system_architecture.pdf/i });
    expect(cite1Btn).toBeInTheDocument();
    expect(cite1Btn).toHaveTextContent("[1]");

    fireEvent.click(cite1Btn);
    expect(onOpenCitation).toHaveBeenCalledTimes(1);
    expect(onOpenCitation).toHaveBeenCalledWith(mockSources[0]);

    // [Source 2] button
    const cite2Btn = screen.getByRole("button", { name: /View citation \[2\] from performance_spec.pdf/i });
    expect(cite2Btn).toBeInTheDocument();
    expect(cite2Btn).toHaveTextContent("[Source 2]");

    fireEvent.click(cite2Btn);
    expect(onOpenCitation).toHaveBeenCalledTimes(2);
    expect(onOpenCitation).toHaveBeenCalledWith(mockSources[1]);
  });

  it("leaves unmatched citations as ordinary non-clickable text", () => {
    const onOpenCitation = vi.fn();
    const content = "According to an unindexed draft [99], the protocol is deprecated.";
    render(
      <MarkdownMessage
        content={content}
        sources={mockSources}
        onOpenCitation={onOpenCitation}
      />
    );

    // [99] should NOT be a button
    expect(screen.queryByRole("button", { name: /View citation \[99\]/i })).not.toBeInTheDocument();
    expect(screen.getByText(/According to an unindexed draft \[99\], the protocol is deprecated\./i)).toBeInTheDocument();
  });

  it("ensures XSS safety by not evaluating raw HTML tags", () => {
    const content = "Text with <script>alert('xss')</script> and <img src=x onerror=alert(1) />.";
    const { container } = render(<MarkdownMessage content={content} />);

    // Script tag must NOT exist in the DOM as an element
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    // It should be rendered harmlessly as text
    expect(container).toHaveTextContent("<script>alert('xss')</script>");
  });

  it("renders concise specific answer with direct opening and inline citation", () => {
    const onOpen = vi.fn();
    const content = "Kamal S. Patil is a Computer Engineering student [1] specializing in React and Python.";
    render(<MarkdownMessage content={content} sources={mockSources} onOpenCitation={onOpen} />);

    const citeBtn = screen.getByRole("button", { name: /View citation \[1\] from system_architecture.pdf/i });
    expect(citeBtn).toBeInTheDocument();
    expect(screen.getByText(/Kamal S\. Patil is a Computer Engineering student/)).toBeInTheDocument();
  });

  it("renders broad structured answer with headings, bullet lists, and citations", () => {
    const onOpen = vi.fn();
    const content =
      "### Technical Skills\n- Python and TypeScript [1]\n- React and FastAPI [2]\n\n### Education\n- Bachelor of Engineering [1]";
    render(<MarkdownMessage content={content} sources={mockSources} onOpenCitation={onOpen} />);

    expect(screen.getByRole("heading", { level: 3, name: "Technical Skills" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 3, name: "Education" })).toBeInTheDocument();
    expect(screen.getByText(/Python and TypeScript/)).toBeInTheDocument();
    expect(screen.getByText(/React and FastAPI/)).toBeInTheDocument();

    const cite1Btn = screen.getAllByRole("button", { name: /View citation \[1\] from system_architecture.pdf/i });
    expect(cite1Btn.length).toBe(2);

    const cite2Btn = screen.getByRole("button", { name: /View citation \[2\] from performance_spec.pdf/i });
    expect(cite2Btn).toBeInTheDocument();
  });
});
