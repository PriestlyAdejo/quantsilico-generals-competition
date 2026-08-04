import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { useDataSource } from "../app/DataSourceProvider";
import { DocSection, DocIndex } from "../types/documentation";
import FilterBar from "../components/forms/FilterBar";
import { Copy, Check } from "lucide-react";

export default function DocumentationPage() {
  const ds = useDataSource();
  const navigate = useNavigate();
  const { sectionId } = useParams<{ sectionId?: string }>();
  const [index, setIndex] = useState<DocIndex | null>(null);
  const [activeSection, setActiveSection] = useState<DocSection | null>(null);
  const [search, setSearch] = useState("");
  const [copiedBlock, setCopiedBlock] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    ds.getDocumentationIndex().then(async (idx) => {
      if (cancelled) return;
      setIndex(idx);
      const initial =
        sectionId && idx.sections.some((s) => s.id === sectionId)
          ? sectionId
          : idx.sections[0]?.id;
      if (!initial) return;
      const section = await ds.getDocumentationSection(initial);
      if (cancelled) return;
      setActiveSection(section);
      if (!sectionId) navigate(`/documentation/${initial}`, { replace: true });
    });
    return () => {
      cancelled = true;
    };
  }, [ds, navigate, sectionId]);

  const filteredSections =
    index?.sections.filter((s) => {
      if (!search) return true;
      const q = search.toLowerCase();
      return (
        s.title.toLowerCase().includes(q) ||
        s.id.toLowerCase().includes(q) ||
        s.tags?.some((t) => t.toLowerCase().includes(q))
      );
    }) ?? [];

  const handleSectionClick = (id: string) => {
    navigate(`/documentation/${id}`);
  };

  const copyBlock = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedBlock(code);
    setTimeout(() => setCopiedBlock(null), 1500);
  };

  const renderContent = (content: string) => {
    const parts = content.split(/(```[\s\S]*?```)/g);
    return parts.map((part, i) => {
      if (part.startsWith("```")) {
        const lines = part.split("\n");
        const lang = lines[0].replace("```", "").trim();
        const code = lines.slice(1, -1).join("\n");
        const isCopied = copiedBlock === code;
        return (
          <div key={i} className="my-3 rounded-sm border border-[#1E2630] overflow-hidden">
            <div className="flex items-center justify-between px-3 py-1.5 bg-[#0C1116] border-b border-[#1E2630]">
              <span className="text-[#6F7C89] font-mono text-xs">{lang || "code"}</span>
              <button
                onClick={() => copyBlock(code)}
                className="flex items-center gap-1 text-[#6F7C89] hover:text-[#FFB000] transition-colors font-mono text-xs"
              >
                {isCopied ? <Check size={10} className="text-[#3FB950]" /> : <Copy size={10} />}
                {isCopied ? "COPIED" : "COPY"}
              </button>
            </div>
            <pre
              className="px-3 py-2.5 overflow-x-auto text-[#CDD6DF]"
              style={{ fontFamily: "var(--font-mono)", fontSize: 11, lineHeight: 1.6 }}
            >
              {code}
            </pre>
          </div>
        );
      }
      return (
        <div
          key={i}
          className="text-sm text-[#8593A1] leading-relaxed whitespace-pre-wrap"
          style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}
        >
          {part}
        </div>
      );
    });
  };

  return (
    <div className="flex h-full overflow-hidden">
      <aside
        className="w-56 flex-shrink-0 border-r border-[#1E2630] overflow-y-auto p-3 hidden md:block"
        style={{ minHeight: "calc(100vh - 32px)" }}
      >
        <p className="text-[#6F7C89] font-mono text-xs uppercase tracking-widest mb-3">Sections</p>
        <FilterBar search={search} onSearch={setSearch} placeholder="Search Phase 9Q, PFSP…" />
        <nav className="space-y-0.5">
          {filteredSections.map((s) => (
            <button
              key={s.id}
              onClick={() => handleSectionClick(s.id)}
              className={`w-full text-left px-2 py-1.5 rounded-sm transition-colors ${
                activeSection?.id === s.id
                  ? "bg-[#1E2630] text-[#FFB000]"
                  : "text-[#8593A1] hover:text-[#CDD6DF] hover:bg-[#0C1116]"
              }`}
              style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}
            >
              {s.title}
            </button>
          ))}
        </nav>
      </aside>

      <main className="flex-1 overflow-y-auto p-6">
        <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-4">$ documentation/</p>
        {activeSection ? (
          <div className="max-w-2xl">
            <div className="prose prose-invert">{renderContent(activeSection.content)}</div>
          </div>
        ) : (
          <p className="text-[#4A5568] font-mono text-xs">Select a section.</p>
        )}
      </main>
    </div>
  );
}
