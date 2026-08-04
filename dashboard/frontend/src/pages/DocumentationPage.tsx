import React, { useEffect, useState, type ComponentType } from "react";
import { useNavigate, useParams } from "react-router";
import { useDataSource } from "../app/DataSourceProvider";
import { DocSection, DocIndex } from "../types/documentation";
import FilterBar from "../components/forms/FilterBar";
import DocMarkdown from "../components/documentation/DocMarkdown";
import { EnvOfficialDoc, GlossaryDoc, OverviewDoc, type MdxSectionId } from "../docs/mdx/registry";
import "../styles/doc-prose.css";

const MDX_PAGES: Record<MdxSectionId, ComponentType> = {
  overview: OverviewDoc,
  glossary: GlossaryDoc,
  "env-official": EnvOfficialDoc,
};

export default function DocumentationPage() {
  const ds = useDataSource();
  const navigate = useNavigate();
  const { sectionId } = useParams<{ sectionId?: string }>();
  const [index, setIndex] = useState<DocIndex | null>(null);
  const [activeSection, setActiveSection] = useState<DocSection | null>(null);
  const [search, setSearch] = useState("");

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

  const MdxPage =
    activeSection && activeSection.id in MDX_PAGES
      ? MDX_PAGES[activeSection.id as MdxSectionId]
      : null;

  return (
    <div className="flex h-full overflow-hidden">
      <aside
        className="w-56 flex-shrink-0 border-r border-[#1E2630] overflow-y-auto p-3 hidden md:block"
        style={{ minHeight: "calc(100vh - 32px)" }}
      >
        <p className="text-[#6F7C89] font-mono text-xs uppercase tracking-widest mb-3">Sections</p>
        <FilterBar search={search} onSearch={setSearch} placeholder="Search docs, PFSP…" />
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
            {MdxPage ? (
              <div className="doc-prose">
                <MdxPage />
              </div>
            ) : (
              <DocMarkdown content={activeSection.content} />
            )}
          </div>
        ) : (
          <p className="text-[#4A5568] font-mono text-xs">Select a section.</p>
        )}
      </main>
    </div>
  );
}
