---
description: Convert a scientific paper (PDF or .tex) into a QuantEcon-style MyST markdown lecture with Python code and exercises. Use when writing a QuantEcon lecture, converting a paper to MyST markdown, or creating a lecture from a PDF or .tex file.
argument-hint: [path-to-paper.pdf-or-.tex]
---

You are helping Thomas Sargent (co-founder of QuantEcon) convert a scientific paper into a QuantEcon lecture.

## Your Task

Read the paper at `$1` (PDF or LaTeX file) and produce two outputs:

1. A QuantEcon lecture written in **MyST-flavoured Markdown** (`.md`)
2. A **BibTeX `.bib` file** for any references not already in `quant-econ.bib`

---

## Step 1 — Read the Paper

Use available tools to read and understand the paper at `$1`. Extract:
- The core economic/mathematical ideas and results
- All notation, equations, and model definitions
- Any empirical or computational results worth replicating
- All references cited

---

## Step 2 — Study the Example Lecture

Read the file `likelihood_ratio_process.md` in the current directory. Use it as a **style and structure template** — including how sections, equations, figures, code cells, and exercises are formatted.

Also examine `quant-econ.bib` (if present) to identify which references are already covered.

---

## Step 3 — Write the MyST Lecture

Produce a `.md` file named after the paper topic (snake_case). Follow these rules strictly:

### MyST Formatting Rules

- Use MyST-flavoured Markdown compatible with Jupyter Book
- Begin with a Jupyter Book YAML front-matter block:
  ```yaml
  ---
  jupytext:
    text_representation:
      extension: .md
      format_name: myst
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  ---
  ```
- Section headings: `#`, `##`, `###`
- **Equations**: display math must have a **blank line before the opening `$$` and a blank line after the closing `$$`**. Never place `$$` immediately adjacent to text or other lines without a blank line separator.
- Inline math: `$...$`
- Label display equations with `{eq}` directives when they are referenced in the text
- Code cells: use MyST code-cell fences:
  ````
  ```{code-cell} ipython3
  # Python code here
  ```
  ````
- Notes, warnings, tips: use admonition directives (`{note}`, `{warning}`, `{tip}`)
- Citations: use `{cite}` role, e.g. `{cite}SargentWallace1981`

### Content Requirements

1. **Introduction** — motivate the paper's contribution in accessible terms
2. **Model Setup** — define notation and the economic/mathematical environment
3. **Key Results** — present theorems, propositions, or main findings with equations
4. **Python Implementation** — self-contained, well-commented code cells that:
   - Import only standard QuantEcon-compatible packages (`numpy`, `scipy`, `matplotlib`, `quantecon`, etc.)
   - Reproduce or illustrate the paper's main results computationally
   - Produce clearly labelled plots
5. **Exercises** — at least 3 exercises of increasing difficulty at the end, followed by **Solutions** in collapsed `{dropdown}` admonition blocks:
   ```
   ````{dropdown} Solution
   ```{code-cell} ipython3
   # solution code
   ```
   ````
   ```
6. **References** — a `{bibliography}` directive at the end

---

## Step 4 — Write the BibTeX File

- Check `quant-econ.bib` for existing entries
- Create a new `.bib` file (named `<lecture_name>.bib`) containing only references **not** already in `quant-econ.bib`
- Use standard BibTeX format with clean, consistent cite keys (AuthorYYYY style)

---

## Quality Checklist

Before finishing, verify:
- [ ] Every display equation has a blank line before `$$` and after `$$`
- [ ] All code cells are self-contained and would run without errors
- [ ] Exercises have solution dropdowns
- [ ] All citations have BibTeX entries somewhere
- [ ] The lecture reads as a coherent pedagogical document, not just a paper summary
