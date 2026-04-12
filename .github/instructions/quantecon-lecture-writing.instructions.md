---
applyTo: "lectures/**/*.md"
description: "MyST markdown and QuantEcon lecture writing conventions. Applied when editing or creating files in the lectures/ directory."
---

# QuantEcon Lecture Writing Conventions

## Equation Spacing (Critical)

Display equations **must** have a blank line before `$$` and after `$$`:

```
text before

$$
equation here
$$

text after
```

Never place `$$` immediately adjacent to text lines.

## File Frontmatter

Every lecture `.md` file starts with:

```yaml
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

## Cross-Reference Label

Immediately after frontmatter, before the title:

```
(lecture_label)=
```{raw} jupyter
<div id="qe-notebook-header" ...>...</div>
```

# Title
```

## Code Cells

All executable Python uses `` ```{code-cell} ipython3 ``.
Non-executable code uses `` ```python ``.

## Citations and References

- Cite with `{cite}` `` `BibKey` ``
- Check `lectures/_static/quant-econ.bib` for existing keys before adding new ones
- New references go in a separate `_extra.bib` file alongside the lecture

## Exercises

Use the paired directives:

```
```{exercise}
:label: label_ex1
...
```

```{solution-start} label_ex1
:class: dropdown
```
...
```{solution-end}
```
```

## Preferred Python Libraries

`numpy`, `scipy`, `matplotlib`, `quantecon`, `jax` (for computationally intensive work), `numba`
