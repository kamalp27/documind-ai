import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ChatView } from "../components/chat/ChatView";
import { apiClient } from "../services/api";
import { DocumentMetadata } from "../types";

const sampleDocs: DocumentMetadata[] = [
  {
    document_id: "doc-1",
    filename: "spec.pdf",
    file_size_bytes: 1024,
    total_pages: 2,
    total_chunks: 4,
    created_at: "2026-10-25T12:00:00Z",
  },
];

describe("ChatView", () => {
  it("renders welcome prompt when messages array is empty", () => {
    render(
      <ChatView
        documents={sampleDocs}
        selectedDocId={null}
        onSelectDoc={vi.fn()}
        onOpenCitation={vi.fn()}
      />
    );
    expect(screen.getByText(/Ask anything about your documents/i)).toBeInTheDocument();
  });

  it("renders versatile starter prompt suggestions and populates input on click", () => {
    render(
      <ChatView
        documents={sampleDocs}
        selectedDocId={null}
        onSelectDoc={vi.fn()}
        onOpenCitation={vi.fn()}
      />
    );
    const suggestion1 = screen.getByText("Summarize the key information in this document.");
    expect(suggestion1).toBeInTheDocument();
    expect(screen.getByText("What are the main topics and details covered?")).toBeInTheDocument();
    expect(screen.getByText("What skills, facts, or important details are mentioned?")).toBeInTheDocument();

    const textarea = screen.getByPlaceholderText(/Ask a question about your indexed documents/i) as HTMLTextAreaElement;
    expect(textarea.value).toBe("");

    fireEvent.click(suggestion1);
    expect(textarea.value).toBe("Summarize the key information in this document.");
  });

  it("submits query, renders assistant response and citation badges", async () => {
    const mockQueryResponse = {
      answer: "The specification details a decoupled RAG architecture [1].",
      sources: [
        {
          source_id: 1,
          document_id: "doc-1",
          filename: "spec.pdf",
          page_numbers: [2],
          excerpt: "Decoupled FastAPI backend and React frontend.",
          similarity_score: 0.92,
        },
      ],
      insufficient_evidence: false,
      model_used: "gemini-2.5-flash",
      request_id: "req-123",
    };

    vi.spyOn(apiClient, "queryChat").mockResolvedValueOnce(mockQueryResponse);
    const citationSpy = vi.fn();

    render(
      <ChatView
        documents={sampleDocs}
        selectedDocId={null}
        onSelectDoc={vi.fn()}
        onOpenCitation={citationSpy}
      />
    );

    const textarea = screen.getByPlaceholderText(/Ask a question about your indexed documents/i);
    fireEvent.change(textarea, { target: { value: "What is the system architecture?" } });

    const sendBtn = screen.getByRole("button", { name: /Send/i });
    fireEvent.click(sendBtn);

    await waitFor(() => {
      expect(screen.getByText(/The specification details a decoupled RAG architecture/i)).toBeInTheDocument();
    });

    // Check citation badge
    const citationBtn = screen.getByText(/\[1\] spec.pdf \(p. 2\)/i);
    expect(citationBtn).toBeInTheDocument();

    fireEvent.click(citationBtn);
    expect(citationSpy).toHaveBeenCalledWith(mockQueryResponse.sources[0]);
  });

  it("handles insufficient evidence gracefully without hallucination", async () => {
    const mockInsufficient = {
      answer: "The uploaded documents do not contain sufficient information to answer this question.",
      sources: [],
      insufficient_evidence: true,
      model_used: null,
      request_id: "req-none",
    };

    vi.spyOn(apiClient, "queryChat").mockResolvedValueOnce(mockInsufficient);

    render(
      <ChatView
        documents={sampleDocs}
        selectedDocId={null}
        onSelectDoc={vi.fn()}
        onOpenCitation={vi.fn()}
      />
    );

    const textarea = screen.getByPlaceholderText(/Ask a question/i);
    fireEvent.change(textarea, { target: { value: "Who is the president of Mars?" } });
    fireEvent.click(screen.getByRole("button", { name: /Send/i }));

    await waitFor(() => {
      expect(
        screen.getByText(/The uploaded documents do not contain sufficient information/i)
      ).toBeInTheDocument();
    });
    expect(screen.getByText(/No document passages exceeded the semantic relevance threshold/i)).toBeInTheDocument();
  });

  it("renders inline citations as interactive buttons that open citation drawer", async () => {
    const mockQueryResponse = {
      answer: "Key finding is supported by [1] and [Source 2].",
      sources: [
        {
          source_id: 1,
          document_id: "doc-1",
          filename: "spec.pdf",
          page_numbers: [2],
          excerpt: "Decoupled FastAPI backend and React frontend.",
          similarity_score: 0.92,
        },
        {
          source_id: 2,
          document_id: "doc-2",
          filename: "benchmarks.pdf",
          page_numbers: [1],
          excerpt: "High retrieval accuracy.",
          similarity_score: 0.85,
        },
      ],
      insufficient_evidence: false,
      model_used: "gemini-3.8-flash",
      request_id: "req-inline-test",
    };

    vi.spyOn(apiClient, "queryChat").mockResolvedValueOnce(mockQueryResponse);
    const citationSpy = vi.fn();

    render(
      <ChatView
        documents={sampleDocs}
        selectedDocId={null}
        onSelectDoc={vi.fn()}
        onOpenCitation={citationSpy}
      />
    );

    const textarea = screen.getByPlaceholderText(/Ask a question/i);
    fireEvent.change(textarea, { target: { value: "Where is this documented?" } });
    fireEvent.click(screen.getByRole("button", { name: /Send/i }));

    await waitFor(() => {
      expect(screen.getByText(/Key finding is supported by/i)).toBeInTheDocument();
    });

    // Click inline [1] button
    const inline1 = screen.getByRole("button", { name: /View citation \[1\] from spec.pdf/i });
    expect(inline1).toBeInTheDocument();
    fireEvent.click(inline1);
    expect(citationSpy).toHaveBeenCalledWith(mockQueryResponse.sources[0]);

    // Click inline [Source 2] button
    const inline2 = screen.getByRole("button", { name: /View citation \[2\] from benchmarks.pdf/i });
    expect(inline2).toBeInTheDocument();
    fireEvent.click(inline2);
    expect(citationSpy).toHaveBeenCalledWith(mockQueryResponse.sources[1]);
  });
});
