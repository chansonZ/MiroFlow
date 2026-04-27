import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Components } from 'react-markdown';

interface MarkdownRendererProps {
  content: string;
}

/**
 * Normalizes markdown table content that may arrive with all rows on a single
 * line (a common LLM output quirk).  When the alignment row (`| :--- |`) is
 * detected inside a longer line, each `| … |` row segment is split onto its
 * own line so that remark-gfm can parse the table correctly.
 *
 * Row boundaries are identified by the pattern `| |` — a closing pipe of one
 * row followed only by whitespace and then the opening pipe of the next row.
 * Normal intra-row cell separators always have non-empty cell content between
 * the two pipes, so they are not matched.
 */
function normalizeTableMarkdown(text: string): string {
  return text
    .split('\n')
    .map((line) => {
      // Only attempt to reformat lines that look like concatenated table rows:
      // they must start with `|` and contain a GFM alignment-row marker
      // (`:---`, `---`, `:--:`, etc.) to limit false positives.
      if (!line.startsWith('|') || !/\|[ \t]*:?-+:?[ \t]*\|/.test(line)) {
        return line;
      }

      // Row boundaries: `|` followed by ONE OR MORE whitespace chars followed
      // immediately by `|`.  Inside a row, each `|` is followed by a cell value
      // (non-`|` text), so intra-row separators never match this pattern.
      const parts = line.split(/\|[ \t]+\|/);
      if (parts.length <= 1) return line;

      // Re-attach the `|` delimiters that were consumed by the split.
      const rows = parts.map((part, i) => {
        const trimmed = part.trim();
        // First fragment already starts with `|`; later ones lost their leading `|`.
        const withLeading = i === 0 ? trimmed : `| ${trimmed}`;
        // All rows must end with `|`.
        return withLeading.endsWith('|') ? withLeading : `${withLeading} |`;
      });

      return rows.join('\n');
    })
    .join('\n');
}

const tableComponents: Components = {
  table: ({ children }) => (
    <div className="overflow-x-auto my-4">
      <table className="min-w-full border-collapse border border-gray-300 text-sm">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-gray-100">{children}</thead>
  ),
  tbody: ({ children }) => (
    <tbody className="divide-y divide-gray-200">{children}</tbody>
  ),
  tr: ({ children }) => (
    <tr className="even:bg-gray-50">{children}</tr>
  ),
  th: ({ children }) => (
    <th className="border border-gray-300 px-3 py-2 text-left font-semibold text-gray-700 whitespace-nowrap">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-gray-300 px-3 py-2 text-gray-700 align-top">
      {children}
    </td>
  ),
};

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  const normalizedContent = normalizeTableMarkdown(content);
  return (
    <div className="markdown-content prose prose-sm max-w-none">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={tableComponents}>
        {normalizedContent}
      </ReactMarkdown>
    </div>
  );
}
