---
name: "QuantEcon Lecture from Paper"
description: "Convert a scientific paper (PDF or .tex) into a QuantEcon lecture in MyST markdown. Attach the paper file before invoking. Produces a .md lecture file and a supplementary .bib file."
argument-hint: "Attach the paper PDF or .tex file, then optionally specify the desired output filename (e.g. 'my_topic.md')"
agent: "agent"
---

You are helping Thomas Sargent convert a scientific paper into a QuantEcon lecture written in the MyST dialect of markdown, following the style and conventions of [lectures/likelihood_ratio_process.md](../lectures/likelihood_ratio_process.md).

## Your Task

1. **Read the attached paper** (PDF or .tex). Understand its core economic/mathematical content, key results, key intuitions, and analytical techniques.

2. **Draft a complete QuantEcon lecture** as a `.md` file in `lectures/`. The lecture should:
   - Explain the paper's ideas accessibly to a graduate student audience
   - Lead the reader through the theory step by step, not just summarize
   - Include substantial Python code cells that illustrate, compute, and visualize the paper's key results
   - End with exercises (with full solutions in dropdown blocks)

3. **Produce a supplementary `.bib` file** for any references not already in [lectures/_static/quant-econ.bib](../lectures/_static/quant-econ.bib).

---

## MyST / Jupyter Book Format Rules

Follow these rules exactly. Study [lectures/likelihood_ratio_process.md](../lectures/likelihood_ratio_process.md) as the canonical example.

### File Frontmatter (required, verbatim structure)

```
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
```

### Required Header Block

Immediately after the frontmatter, add a cross-reference label and the QuantEcon logo block:

```
(my_lecture_label)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Lecture Title

```{contents} Contents
:depth: 2
```
```

### Equations — CRITICAL SPACING RULE

Every display equation block **must** have a blank line before the opening `$$` and a blank line after the closing `$$`. This is mandatory.

**Correct:**

```
some text before

$$
E[x] = \mu
$$

some text after
```

**Wrong (will break the build):**

```
some text before
$$
E[x] = \mu
$$
some text after
```

Inline math uses single dollars: $\mu$, $\sigma^2$.

Multi-line aligned equations use:

```
$$
\begin{aligned}
a &= b \\
c &= d
\end{aligned}
$$
```

### Code Cells

Use ` ```{code-cell} ipython3 ` for all executable Python. For the `pip install` cell at the top (if needed):

```
```{code-cell} ipython3
:tags: [hide-output]
!pip install --upgrade quantecon
```
```

### Citations

Use `{cite}` with the BibTeX key: `{cite}` `` `Author_Year` ``. Example: `{cite}` `` `Neyman_Pearson` ``.

Check [lectures/_static/quant-econ.bib](../lectures/_static/quant-econ.bib) first. Add only truly missing references to the new `.bib` file.

### Cross-references

- Link to other lectures: `{doc}` `` `likelihood_ratio_process` ``
- Label a section: `(my_label)=` on the line before the heading
- Reference a label: `{ref}` `` `my_label` ``

### Admonitions

```
```{note}
...
```

```{warning}
...
```
```

### Exercises with Solutions

```
```{exercise}
:label: ex_label1

Exercise text here.
```

```{solution-start} ex_label1
:class: dropdown
```

Full solution here, including code cells if needed.

```{solution-end}
```
```

---

## Lecture Structure Template

Follow this section order:

1. **Overview** — What is this lecture about? What will the reader learn? List bullets.
2. **Setup** — Imports code cell (all needed libraries). If non-standard packages are needed, add the `pip install` cell first.
3. **Theory sections** — Walk through mathematical content. Alternate prose, equations, and code cells. Each major concept gets its own `##` section.
4. **Computational/Simulation sections** — Python code that replicates or extends the paper's numerical results.
5. **Exercises** — 2–4 exercises ranging from straightforward to challenging, each with a full solution.
6. **References** — at the end, just add: `` ```{bibliography} `` on its own if references were cited (the global bib handles this automatically via `_config.yml`).

---

## Python Code Guidelines

- Use `numpy`, `scipy`, `matplotlib`, `quantecon` as the default stack
- Prefer `jax.numpy` / JAX for computationally intensive sections (this repo already has JAX installed)
- Every figure should call `plt.show()` or `plt.tight_layout(); plt.show()`
- Write clean, readable code with short docstrings on functions
- Simulate and plot the paper's key theoretical results rather than just describing them

---

## Supplementary BibTeX File

Name it `lectures/_static/<lecture_name>_extra.bib`. Format example:

```bibtex
@article{Author_Year,
  author  = {Last, First and Last2, First2},
  title   = {Full Title of the Paper},
  journal = {Journal Name},
  volume  = {XX},
  number  = {Y},
  pages   = {1--30},
  year    = {YYYY}
}
```

Only include references **not already found** in `lectures/_static/quant-econ.bib`.

---

## Output

Produce the complete lecture as a single MyST markdown file. After completing it, also report:
- The name and path of the output file (e.g. `lectures/my_topic.md`)
- The name and path of the supplementary bib file (if any new references were needed)
- A brief (3–5 bullet) summary of what the lecture covers
