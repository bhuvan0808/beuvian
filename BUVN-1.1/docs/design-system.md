# Beuvian Design System

> Line Art + Neo-Brutalist + Minimalist

```
═══════════════════════════════════════════════════════════════
  BEUVIAN DESIGN SYSTEM v1.0
  Raw. Honest. No-nonsense.
═══════════════════════════════════════════════════════════════
```

---

## 1 | Brand Identity

| Key        | Value                                         |
|------------|-----------------------------------------------|
| Ecosystem  | **Beuvian**                                   |
| Foundation | **BUVN** (BhUVaN)                             |
| Code Agent | **SRVN** (SaRVaN)                             |
| Finance    | **MNI** (MoNItor)                             |
| Tagline    | *One Foundation. Three Intelligences.*        |
| Philosophy | Raw, honest, no-nonsense AI -- built from scratch, nothing hidden |

### Voice

- Dense over fluffy
- Numbers over adjectives
- Code over prose
- Show the work, always

---

## 2 | Color Palette

### Primary

```
╔══════════════╦═══════════╦═══════════════════════════════════════╗
║ Name         ║ Hex       ║ Usage                                 ║
╠══════════════╬═══════════╬═══════════════════════════════════════╣
║ Void         ║ #0A0A0A   ║ Primary background (near-black)       ║
║ Bone         ║ #F5F0EB   ║ Primary text (off-white, warm)        ║
║ Signal       ║ #FF3E00   ║ Accent — highlights, links, callouts  ║
╚══════════════╩═══════════╩═══════════════════════════════════════╝
```

### Model Accents

```
╔══════════════╦═══════════╦═══════════════════════════════════════╗
║ Model        ║ Hex       ║ Name                                  ║
╠══════════════╬═══════════╬═══════════════════════════════════════╣
║ BUVN         ║ #F5F0EB   ║ Bone — foundation is neutral          ║
║ SRVN         ║ #00FF88   ║ Terminal Green — code agent            ║
║ MNI          ║ #FFD600   ║ Gold — finance model                  ║
╚══════════════╩═══════════╩═══════════════════════════════════════╝
```

### Utility

| Name    | Hex       | Usage                  |
|---------|-----------|------------------------|
| Muted   | `#6B6B6B` | Secondary text, borders|
| Success | `#00FF88` | Positive states        |
| Warning | `#FFD600` | Caution states         |
| Error   | `#FF3E00` | Failures, danger       |

### Rules

- 90% of any screen is Void + Bone
- Color is used ONLY to convey meaning (status, model identity, emphasis)
- No gradients. Ever. Flat fills only.
- If in doubt, use Bone on Void.

---

## 3 | Typography

```
╔══════════════════╦═══════════════════════════════════════════════╗
║ Role             ║ Font Stack                                    ║
╠══════════════════╬═══════════════════════════════════════════════╣
║ Display/Headings ║ Space Mono, monospace                         ║
║ Primary/Body     ║ JetBrains Mono, Fira Code, monospace          ║
║ Fallback         ║ Consolas, Monaco, Courier New, monospace      ║
╚══════════════════╩═══════════════════════════════════════════════╝
```

### Absolute Rules

- **NO** serif fonts. Anywhere.
- **NO** sans-serif fonts. Anywhere.
- Monospace is the brand. Code IS the aesthetic.
- Heading sizes: H1 = bold uppercase, H2 = bold, H3 = regular bold
- Line height: 1.5 for body, 1.2 for headings

---

## 4 | Design Principles

```
 1. RAW OVER POLISHED        ASCII art over SVG illustrations
 2. MONOSPACE EVERYTHING     Code IS the aesthetic
 3. HIGH CONTRAST            Black background, bright text
 4. THICK BORDERS            Box-drawing chars: ═ ║ ╔ ╗ ╚ ╝
 5. NO GRADIENTS             Flat colors only
 6. NO ROUNDED CORNERS       Sharp edges everywhere
 7. MINIMAL COLOR            90% black/white, color for meaning only
 8. INFORMATION DENSE        Tables over paragraphs
 9. SHOW THE WORK            Real numbers, real code, no marketing fluff
```

---

## 5 | GitHub README Styling Rules

### Badges

All badges use `style=flat-square` with `labelColor=0A0A0A`:

```
![label](https://img.shields.io/badge/LABEL-VALUE-F5F0EB?style=flat-square&labelColor=0A0A0A)
```

Model-specific badge colors:

```
BUVN:  F5F0EB (Bone)
SRVN:  00FF88 (Terminal Green)
MNI:   FFD600 (Gold)
```

### Section Dividers

Thin divider (between subsections):

```
───────────────────────────────────────────────────────────
```

Thick divider (between major sections):

```
═══════════════════════════════════════════════════════════
```

### Headings

- NO emoji in headings
- Use ASCII symbols: `>`, `/`, `|`, `#`, `*`, `-->`
- Examples:
  - `## > Architecture`
  - `## | Benchmarks`
  - `## * Roadmap`

### Code Blocks

Use `diff` syntax for emphasis:

```diff
+ BUVN-2.0  PPL 29.19  -- beat GPT-2 Small
- GPT-2     PPL 29.41  -- trained on 20,000x more data
```

### HTML Usage

- `<pre>` blocks for ASCII art that must preserve spacing
- `bgcolor` on table cells for colored blocks (sparingly)
- `<b>` and `<code>` for inline emphasis in HTML tables
- NO `<div align="center">` — left-aligned is the default, raw is the style

---

## 6 | ASCII Art Style

### Box Format

Use double-line box-drawing characters for primary containers:

```
╔══════════════════════════════════╗
║  BUVN -- Foundation Model       ║
║  109.5M params | PPL 29.19      ║
╠══════════════════════════════════╣
║  -> 12 layers, 768 dim, 12 heads║
║  -> 32K vocab, 1024 context     ║
║  -> Trained on 2B tokens (C4)   ║
╚══════════════════════════════════╝
```

Single-line for secondary/nested containers:

```
┌──────────────────────────────────┐
│  Sub-component or detail         │
└──────────────────────────────────┘
```

### Tree Diagrams

```
BEUVIAN
├── BUVN [Foundation]
│   ├── 12 layers, 768 dim
│   ├── 109.5M params
│   └── PPL 29.19
├── SRVN [Code Agent]
│   ├── Fine-tuned from BUVN
│   └── Planned
└── MNI  [Finance]
    ├── Domain-trained from BUVN
    └── Planned
```

### Flow Diagrams

Use arrows and boxes — never mermaid:

```
Raw Text --> [Tokenizer] --> [Transformer x12] --> [LM Head] --> Next Token
```

Or vertical:

```
  Input Tokens
      |
      v
  ┌──────────┐
  │ Embedding │
  └────┬─────┘
       |
  ┌────v─────┐
  │ Block x12│
  └────┬─────┘
       |
  ┌────v─────┐
  │ LM Head  │
  └────┬─────┘
       |
      v
  Output Logits
```

---

## 7 | Component Examples

### Header Block

```
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   B E U V I A N                                       ║
║   One Foundation. Three Intelligences.                ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
```

### Metric Display

```
╔══════════╦══════════╦══════════╦══════════╗
║ PPL      ║ Top-1    ║ Top-5    ║ Rank     ║
║ 29.19    ║ 37.88%   ║ 60.34%   ║ #8 / 11  ║
╚══════════╩══════════╩══════════╩══════════╝
```

### Status Table

| Item                  | Status |
|-----------------------|--------|
| BUVN-1.1 (CPU test)  | `[DONE]` |
| BUVN-2.0 (H100 prod) | `[DONE]` |
| Scale to 120M params  | `[WIP]`  |
| SRVN code fine-tune   | `[PLANNED]` |
| MNI finance training  | `[PLANNED]` |

### Callout Box

```
┌─── NOTE ─────────────────────────────────────────────┐
│                                                       │
│  BUVN-2.0 beat GPT-2 Small with 20,000x less data.  │
│  The architecture is competitive. The gap to higher-  │
│  ranked models is purely about scale.                 │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### Section Divider (in context)

```
═══════════════════════════════════════════════════════════
 > SECTION TITLE
═══════════════════════════════════════════════════════════
```

### Badge Row (Markdown)

```markdown
![ecosystem](https://img.shields.io/badge/Ecosystem-3_Models-F5F0EB?style=flat-square&labelColor=0A0A0A)
![buvn](https://img.shields.io/badge/BUVN-Foundation-F5F0EB?style=flat-square&labelColor=0A0A0A)
![srvn](https://img.shields.io/badge/SRVN-Code_Agent-00FF88?style=flat-square&labelColor=0A0A0A)
![mni](https://img.shields.io/badge/MNI-Finance-FFD600?style=flat-square&labelColor=0A0A0A)
```

---

## 8 | Anti-Patterns (Do NOT)

| DO NOT                              | DO INSTEAD                          |
|-------------------------------------|-------------------------------------|
| Use emoji in headings               | Use ASCII: `>`, `|`, `*`, `-->`     |
| Use gradients or shadows            | Flat colors, hard edges             |
| Use rounded corners                 | Sharp corners, box-drawing chars    |
| Use serif or sans-serif fonts       | Monospace only                      |
| Write fluffy marketing copy         | Dense, factual, show numbers        |
| Use mermaid diagrams                | ASCII box-drawing diagrams          |
| Use animated GIFs as dividers       | Use `═══` or `───` text dividers    |
| Use `for-the-badge` style badges    | Use `flat-square` only              |
| Center everything                   | Left-align by default               |
| Use capsule-render or typing SVGs   | Use `<pre>` ASCII art               |

---

```
═══════════════════════════════════════════════════════════
  END OF DESIGN SYSTEM
  Beuvian -- Raw. Honest. Built from scratch.
═══════════════════════════════════════════════════════════
```
