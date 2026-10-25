import React, { useCallback, useEffect, useState } from "react";
import { ChatView } from "./components/chat/ChatView";
import { CitationDrawer } from "./components/citations/CitationDrawer";
import { DocumentList } from "./components/documents/DocumentList";
import { DocumentUpload } from "./components/documents/DocumentUpload";
import { Header } from "./components/layout/Header";
import { Sidebar } from "./components/layout/Sidebar";
import { Modal } from "./components/ui/Modal";
import { apiClient } from "./services/api";
import { CitationSource, DocumentMetadata, DocumentUploadResponse } from "./types";

export const App: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState<boolean>(true);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"chat" | "documents">("chat");
  const [activeCitation, setActiveCitation] = useState<CitationSource | null>(null);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState<boolean>(false);
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);
  const [backendVersion, setBackendVersion] = useState<string | null>(null);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);

  // Check health and load documents
  const checkHealth = useCallback(async () => {
    try {
      const health = await apiClient.checkHealth();
      setIsBackendHealthy(health.status === "healthy");
      setBackendVersion(health.version);
    } catch {
      setIsBackendHealthy(false);
      setBackendVersion(null);
    }
  }, []);

  const loadDocuments = useCallback(async () => {
    setIsLoadingDocs(true);
    try {
      const response = await apiClient.listDocuments();
      setDocuments(response.documents || []);
    } catch {
      // In case backend is offline
      setDocuments([]);
    } finally {
      setIsLoadingDocs(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    loadDocuments();
  }, [checkHealth, loadDocuments]);

  const handleUploadSuccess = (newDoc: DocumentUploadResponse) => {
    loadDocuments();
    // Auto-select the newly uploaded document for focused Q&A
    setSelectedDocId(newDoc.document_id);
    setIsUploadModalOpen(false);
    setActiveTab("chat");
  };

  const handleDeleteDocument = async (docId: string) => {
    await apiClient.deleteDocument(docId);
    setDocuments((prev) => prev.filter((d) => d.document_id !== docId));
    if (selectedDocId === docId) {
      setSelectedDocId(null);
    }
  };

  const handleNewChat = () => {
    // Reset any active document filter for a fresh cross-document session
    setSelectedDocId(null);
    setActiveTab("chat");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", backgroundColor: "var(--bg-primary)" }}>
      {/* Top Application Header */}
      <Header
        isBackendHealthy={isBackendHealthy}
        backendVersion={backendVersion}
        documentCount={documents.length}
        onToggleMobileMenu={() => setIsMobileMenuOpen((prev) => !prev)}
      />

      {/* Main Workspace Body */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Navigation Sidebar */}
        <div
          style={{
            display: isMobileMenuOpen ? "flex" : undefined,
          }}
          className={`sidebar-container ${isMobileMenuOpen ? "mobile-open" : ""}`}
        >
          <Sidebar
            currentTab={activeTab}
            onTabChange={(tab) => {
              setActiveTab(tab);
              setIsMobileMenuOpen(false);
            }}
            documents={documents}
            selectedDocId={selectedDocId}
            onSelectDoc={(id) => {
              setSelectedDocId(id);
              setActiveTab("chat");
              setIsMobileMenuOpen(false);
            }}
            onOpenUpload={() => {
              setIsUploadModalOpen(true);
              setIsMobileMenuOpen(false);
            }}
            onNewChat={handleNewChat}
          />
        </div>

        {/* Dynamic Content Pane */}
        <main
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "1.5rem",
            display: "flex",
            flexDirection: "column",
          }}
        >
          {activeTab === "chat" ? (
            <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
              <ChatView
                documents={documents}
                selectedDocId={selectedDocId}
                onSelectDoc={setSelectedDocId}
                onOpenCitation={(source) => setActiveCitation(source)}
              />
            </div>
          ) : (
            <div style={{ maxWidth: "1000px", width: "100%", margin: "0 auto", display: "flex", flexDirection: "column", gap: "2rem" }}>
              <div>
                <h2 style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--text-primary)" }}>
                  Document Management
                </h2>
                <p style={{ fontSize: "0.875rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                  Upload, inspect, and manage PDF documents. Text is automatically chunked and indexed into ChromaDB.
                </p>
              </div>

              {/* Upload Dropzone */}
              <DocumentUpload onUploadSuccess={handleUploadSuccess} />

              {/* Indexed Documents Table */}
              <DocumentList
                documents={documents}
                isLoading={isLoadingDocs}
                onRefresh={loadDocuments}
                onDelete={handleDeleteDocument}
                selectedDocId={selectedDocId}
                onSelectDoc={(id) => {
                  setSelectedDocId(id);
                  setActiveTab("chat");
                }}
              />
            </div>
          )}
        </main>
      </div>

      {/* Upload Dialog Modal */}
      <Modal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        title="Upload PDF Document"
        maxWidth="560px"
      >
        <DocumentUpload onUploadSuccess={handleUploadSuccess} />
      </Modal>

      {/* Citation Inspector Side Drawer */}
      <CitationDrawer
        source={activeCitation}
        onClose={() => setActiveCitation(null)}
      />

      <style>{`
        @media (max-width: 768px) {
          .sidebar-container {
            display: none;
            position: fixed;
            top: 64px;
            bottom: 0;
            left: 0;
            z-index: 40;
            background-color: var(--bg-secondary);
            box-shadow: var(--shadow-lg);
          }
          .sidebar-container.mobile-open {
            display: flex;
          }
        }
      `}</style>
    </div>
  );
};
