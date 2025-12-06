import React, { useEffect, useMemo, useRef, useState } from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath as RKInlineMath, BlockMath as RKBlockMath } from 'react-katex';
import { MathJax, MathJaxContext } from 'better-react-mathjax';
// Safe, consistent math rendering across the app
// - Uses KaTeX for speed when possible
// - Falls back to MathJax v3 (CDN) only if KaTeX reports an error
// - Sanitizes common characters (°, en-dash, mid-dot)
// - Provides readable fallback when both renderers fail

// Using better-react-mathjax for tolerant fallback rendering; no manual loader required.

// Sanitizer fixes common authoring issues and JS-escape damage
function sanitizeMath(input: string): string {
  if (!input) return input;
  let s = input;
  // 1) Fix common symbol issues
  s = s.replace(/(\d)°/g, '$1^{\\circ}'); // 45° -> 45^{\circ}
  s = s.replace(/[–—]/g, '-');               // en/em dash -> minus
  s = s.replace(/([a-zA-Z0-9])·([a-zA-Z0-9])/g, '$1 \\cdot $2'); // mid-dot
  s = s.replace(/[×✕]/g, '\\times');       // Unicode times -> \\times
  s = s.replace(/π/g, '\\pi');             // unicode pi
  s = s.replace(/ℏ/g, '\\hbar');           // unicode hbar
  s = s.replace(/⟨/g, '\\langle');         // bra-ket symbols
  s = s.replace(/⟩/g, '\\rangle');

  // 1b) Repair common authoring shorthand and ASCII bra-ket
  s = s.replace(/\bmathcal\s*([A-Za-z])\b/g, '\\mathcal{$1}'); // mathcalL -> \mathcal{L}
  s = s.replace(/<([^|>]+)\|/g, '\\langle $1|');                 // <v| -> \langle v|
  s = s.replace(/\|([^<>|]+)>/g, '|$1\\rangle');                 // |v> -> |v\\rangle

  // 2) Repair JS-escape-damaged LaTeX sequences (control chars)
  s = s.replace(/[\x0A]abla/g, '\\nabla');     // \\nabla -> newline + abla
  s = s.replace(/[\x0D]ice/g, '\\nice');       // rare, safety
  s = s.replace(/[\x0D]rightarrow/g, '\\rightarrow');
  s = s.replace(/[\x09]imes/g, '\\times');     // tab + 'imes'
  s = s.replace(/[\x0C]rac/g, '\\frac');       // form feed + 'rac'
  s = s.replace(/[\x0B]ec/g, '\\vec');         // vertical tab + 'ec'
  s = s.replace(/[\x08]egin/g, '\\begin');     // backspace + 'egin'

  // 3) If authors typed commands without a leading backslash, fix conservatively
  // Allow punctuation like | ] ) , . ; : ! after the command
  const ensureCmd = (name: string) => {
    const re = new RegExp(`(?<!\\\\)${name}(?=\\s|[({\\[|,.;:!?\\)\\]\\\\]|$)`, 'g');
    s = s.replace(re, `\\\\${name}`);
  };
  [
    'nabla','rightarrow','leftarrow','times','frac','vec','partial','hbar','sigma','Omega','mathbf','mathrm','mathbb','mathcal','langle','rangle','pi',
    'alpha','beta','gamma','delta','theta','phi','psi','sum'
  ].forEach(ensureCmd);

  return s;
}

type InlineProps = React.ComponentProps<typeof RKInlineMath> & { math: string };
export const InlineMath: React.FC<InlineProps> = ({ math, ...rest }) => {
  const safe = sanitizeMath(math);
  const [fallback, setFallback] = useState(false);
  const katexContainerRef = useRef<HTMLSpanElement | null>(null);
  useEffect(() => {
    // Detect KaTeX error marker after render
    if (!fallback && katexContainerRef.current?.querySelector('[data-katex-error="1"]')) {
      setFallback(true);
    }
  }, [safe, fallback]);


  if (fallback) {
    return (
      <MathJaxContext config={{ tex: { inlineMath: [["\\(", "\\)"]] } }}>
        <MathJax inline dynamic>{`\\(${safe}\\)`}</MathJax>
      </MathJaxContext>
    );
  }

  return (
    <span ref={katexContainerRef}>
      <RKInlineMath
        math={safe}
        errorColor="hsl(var(--destructive))"
        renderError={(err) => {
          console.warn('[KaTeX InlineMath error]', err?.message, 'for:', safe);
          return (
            <span data-katex-error="1" className="font-mono text-[11px] text-destructive/80" title={err.message}>{safe}</span>
          );
        }}
        {...rest}
      />
    </span>
  );
};

type BlockProps = React.ComponentProps<typeof RKBlockMath> & { math: string };
export const BlockMath: React.FC<BlockProps> = ({ math, ...rest }) => {
  const safe = sanitizeMath(math);
  const [fallback, setFallback] = useState(false);
  const katexContainerRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!fallback && katexContainerRef.current?.querySelector('[data-katex-error="1"]')) {
      setFallback(true);
    }
  }, [safe, fallback]);


  if (fallback) {
    return (
      <MathJaxContext config={{ tex: { displayMath: [["\\[", "\\]"]] } }}>
        <MathJax dynamic>{`\\[${safe}\\]`}</MathJax>
      </MathJaxContext>
    );
  }

  return (
  <div ref={katexContainerRef}>
      <RKBlockMath
        math={safe}
        errorColor="hsl(var(--destructive))"
        renderError={(err) => (
          <pre data-katex-error="1" className="font-mono text-xs text-destructive/80 bg-destructive/10 p-2 rounded" title={err.message}>{safe}</pre>
        )}
        {...rest}
      />
    </div>
  );
};

export default { InlineMath, BlockMath };
