import React, { useEffect, useState } from "react";
import { useDataSource } from "../app/DataSourceProvider";
import { DocSection, DocIndex } from "../types/documentation";
import FilterBar from "../components/forms/FilterBar";
import { Copy, Check } from "lucide-react";

export default function DocumentationPage() {
  const ds = useDataSource();
  const [index, setIndex] = useState<DocIndex | null>(null);
  const [activeSection, setActiveSection] = useState<DocSection | null>(null);
  const [search, setSearch] = useState("");
  const [copiedBlock, setCopiedBlock] = useState<string | null>(null);

  useEffect(() => {
    ds.getDocumentationIndex().then(idx => {
      setIndex(idx);
      if (idx.sections.length > 0) {
        ds.getDocumentationSection(idx.sections[0].id).then(setActiveSection);
      }
    });
  }, [ds]);

  const filteredSections = index?.sections.filter(s =>
    !search || s.title.toLowerCase().includes(search.toLowerCase()) || s.tags?.some(t => t.includes(search.toLowerCase()))
  ) ?? [];

  const handleSectionClick = (id: string) => {
    ds.getDocumentationSection(id).then(setActiveSection);
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
              <button onClick={() => copyBlock(code)} className="flex items-center gap-1 text-[#6F7C89] hover:text-[#FFB000] transition-colors font-mono text-xs">
                {isCopied ? <Check size={10} className="text-[#3FB950]" /> : <Copy size={10} />}
                {isCopied ? "COPIED" : "COPY"}
              </button>
            </div>
            <pre className="px-3 py-2.5 overflow-x-auto text-[#CDD6DF]" style={{ fontFamily: "var(--font-mono)", fontSize: 11, lineHeight: 1.6 }}>{code}</pre>
          </div>
        );
      }
      const md = part
        .replace(/^# (.+)$/gm, '<h1 class="text-xl font-bold text-[#EAF0F6] mt-4 mb-2" style="font-family:var(--font-heading)">$1</h1>')
        .replace(/^## (.+)$/gm, '<h2 class="text-base font-bold text-[#CDD6DF] mt-3 mb-1.5" style="font-family:var(--font-heading)">$1</h2>')
        .replace(/^### (.+)$/gm, '<h3 class="text-sm font-bold text-[#8593A1] mt-2 mb-1">$1</h3>')
        .replace(/\*\*(.+?)\*\*/g, '<strong class="text-[#CDD6DF] font-bold">$1</strong>')
        .replace(/`([^`]+)`/g, '<code class="text-[#22D3EE] bg-[#0C1116] px-1 rounded-sm" style="font-family:var(--font-mono);font-size:0.9em">$1</code>')
        .replace(/^\| (.+) \|$/gm, '<div class="border-b border-[#1E2630] py-1 font-mono text-xs text-[#8593A1]">$1</div>')
        .replace(/^- (.+)$/gm, '<li class="text-[#8593A1] text-sm ml-4 list-disc">$1</li>')
        .replace(/^\d+\. (.+)$/gm, '<li class="text-[#8593A1] text-sm ml-4 list-decimal">$1</li>')
        .replace(/^> (.+)$/gm, '<blockquote class="border-l-2 border-[#FFB000] pl-3 text-[#8593A1] text-sm my-2 italic">$1</blockquote>')
        .replace(/\n\n/g, '<br /><br />')
        .replace(/\n/g, ' ');
      return <div key={i} dangerouslySetInnerHTML={{ __html: md }} className="text-sm text-[#8593A1] leading-relaxed" />;
    });
  };

  return (
    <div className="flex h-full overflow-hidden">
      <aside className="w-56 flex-shrink-0 border-r border-[#1E2630] overflow-y-auto p-3 hidden md:block" style={{ minHeight: "calc(100vh - 32px)" }}>
        <p className="text-[#6F7C89] font-mono text-xs uppercase tracking-widest mb-3">Sections</p>
        <FilterBar search={search} onSearch={setSearch} placeholder="Filter…" />
        <nav className="space-y-0.5">
          {filteredSections.map(s => (
            <button
              key={s.id}
              onClick={() => handleSectionClick(s.id)}
              className={`w-full text-left px-2 py-1.5 rounded-sm transition-colors ${activeSection?.id === s.id ? "bg-[#1E2630] text-[#FFB000]" : "text-[#8593A1] hover:text-[#CDD6DF] hover:bg-[#0C1116]"}`}
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
            {activeSection.content.includes("commands describe") && (
              <div className="mt-4 border border-[#FFB000] border-opacity-40 rounded-sm p-3 bg-[#0C1116]">
                <p className="text-[#FFB000] font-mono text-xs">⚠ These commands describe the eventual integrated QuantSilico project. They may not exist in the standalone Figma Make frontend export.</p>
              </div>
            )}
          </div>
        ) : (
          <p className="text-[#4A5568] font-mono text-xs">Select a section.</p>
        )}
      </main>
    </div>
  );
}
