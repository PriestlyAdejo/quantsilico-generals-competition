import React from "react";
import { Copy, Check } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  content: string;
}

/**
 * Required-scope documentation renderer (headings, paragraphs, lists, links, code).
 * Uses the MDX/remark ecosystem (remark-gfm) so content is semantic HTML, not a raw <pre>.
 */
export default function DocMarkdown({ content }: Props) {
  const [copied, setCopied] = React.useState<string | null>(null);

  const copyBlock = (code: string) => {
    void navigator.clipboard.writeText(code);
    setCopied(code);
    window.setTimeout(() => setCopied(null), 1500);
  };

  return (
    <div className="doc-prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h1>{children}</h1>,
          h2: ({ children }) => <h2>{children}</h2>,
          h3: ({ children }) => <h3>{children}</h3>,
          p: ({ children }) => <p>{children}</p>,
          ul: ({ children }) => <ul>{children}</ul>,
          ol: ({ children }) => <ol>{children}</ol>,
          li: ({ children }) => <li>{children}</li>,
          a: ({ href, children }) => (
            <a href={href} target={href?.startsWith("http") ? "_blank" : undefined} rel="noreferrer">
              {children}
            </a>
          ),
          code: ({ className, children, ...props }) => {
            const text = String(children).replace(/\n$/, "");
            const isBlock = Boolean(className) || text.includes("\n");
            if (!isBlock) {
              return (
                <code className="doc-inline-code" {...props}>
                  {children}
                </code>
              );
            }
            const lang = (className || "").replace("language-", "") || "code";
            const isCopied = copied === text;
            return (
              <div className="doc-code-block">
                <div className="doc-code-header">
                  <span>{lang}</span>
                  <button type="button" onClick={() => copyBlock(text)}>
                    {isCopied ? <Check size={10} className="text-[#3FB950]" /> : <Copy size={10} />}
                    {isCopied ? "COPIED" : "COPY"}
                  </button>
                </div>
                <pre>
                  <code className={className}>{text}</code>
                </pre>
              </div>
            );
          },
          pre: ({ children }) => <>{children}</>,
          strong: ({ children }) => <strong>{children}</strong>,
          em: ({ children }) => <em>{children}</em>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
