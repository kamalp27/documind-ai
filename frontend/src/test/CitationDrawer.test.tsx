import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CitationDrawer } from "../components/citations/CitationDrawer";
import { CitationSource } from "../types";

const mockSource: CitationSource = {
  source_id: 1,
  document_id: "doc-999-uuid",
  filename: "annual_report.pdf",
  page_numbers: [4],
  excerpt: "Annual revenue increased by 24% year-over-year according to GAAP auditing standards.",
  similarity_score: 0.88,
};

describe("CitationDrawer", () => {
  it("renders null when source is null", () => {
    const { container } = render(<CitationDrawer source={null} onClose={vi.fn()} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders source metadata, page numbers, similarity score, and excerpt", () => {
    render(<CitationDrawer source={mockSource} onClose={vi.fn()} />);

    expect(screen.getByText("Source [1]")).toBeInTheDocument();
    expect(screen.getByText("annual_report.pdf")).toBeInTheDocument();
    expect(screen.getByText("Page 4")).toBeInTheDocument();
    expect(screen.getByText("Similarity: 0.88")).toBeInTheDocument();
    expect(screen.getByText(/Annual revenue increased by 24%/i)).toBeInTheDocument();
    expect(screen.getByText(/Cosine similarity score:/i)).toBeInTheDocument();
  });

  it("handles source with null or missing similarity score gracefully", () => {
    const sourceWithoutScore: CitationSource = {
      ...mockSource,
      similarity_score: undefined as unknown as number,
    };
    render(<CitationDrawer source={sourceWithoutScore} onClose={vi.fn()} />);

    expect(screen.getByText("Source [1]")).toBeInTheDocument();
    expect(screen.queryByText(/Similarity:/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Cosine similarity score:/i)).not.toBeInTheDocument();
  });

  it("calls onClose when close button is clicked", () => {
    const closeSpy = vi.fn();
    render(<CitationDrawer source={mockSource} onClose={closeSpy} />);

    const closeBtn = screen.getByRole("button", { name: /Close citation drawer/i });
    fireEvent.click(closeBtn);
    expect(closeSpy).toHaveBeenCalled();
  });
});
