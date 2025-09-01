# QuantEcon Intermediate Lecture Series Review Instructions

## How to Use This Document

When asked to review a lecture file, Claude Code will:
1. Read this instruction document first
2. Read the lecture file to be reviewed
3. Systematically review the lecture against all rules in this document
4. Propose changes one at a time with detailed reasoning
5. Wait for your approval before implementing each change
6. Commit and push each approved change with detailed commit messages
7. Perform a second review pass to ensure consistency

**Example usage**: "Please read the instruction in quantecon_review_instructions.md and review the xxx.md lecture accordingly"

## Overview
This document provides comprehensive instructions for reviewing QuantEcon Intermediate Lecture series. 

**CRITICAL REQUIREMENT**: When proposing ANY change, you MUST:
1. State the specific rule being applied (e.g., "According to Title Rule #2...")
2. Quote the exact text from the rule
3. Explain why the current text violates this rule
4. Show the proposed correction

Example format:
```
According to [Rule Category] Rule #[Number]: "[exact rule text]"
Current: "The Algorithm" 
Proposed: "The algorithm"
Reason: "Algorithm" is not a proper noun, so it should not be capitalized in section headings.
```

**REVIEW WORKFLOW**: You MUST follow this iterative workflow:

### Phase 1: Initial Review and Changes
1. **Read and analyze** the entire lecture file first
2. **Provide an issue summary** listing all violations found:
   ```
   Issues Found:
   1. [Rule Category] Rule #X: [Brief description of violation]
   2. [Rule Category] Rule #Y: [Brief description of violation]
   ... (continue for all issues)
   ```
3. **Begin iterative changes** - For EACH change proposal, you MUST include:
   - **Issue reference**: "Addressing Issue #X from the summary"
   - **Rule violated**: State the specific rule (e.g., "Title Rule #2")
   - **Quote the exact rule text**: Include the full rule description
   - **Current text**: Show the exact current text that violates the rule
   - **Proposed correction**: Show the exact proposed new text
   - **Detailed explanation**: Explain why this violates the rule and why the change is necessary
   
   Example format for each proposal:
   ```
   **Addressing Issue #1 from the summary**
   
   According to Title Rule #2: "Capitalize ONLY the first word and proper nouns in all other headings (sections, subsections, etc.)"
   
   Current: "The Algorithm for Value Function Iteration"
   Proposed: "The algorithm for value function iteration"
   
   Explanation: This is a section heading, not a lecture title. According to Title Rule #2, only the first word should be capitalized. "Algorithm", "Value", "Function", and "Iteration" are not proper nouns, so they should be lowercase.
   ```
4. **Wait for user approval** before implementing
5. **After approval**, implement the change
6. **Commit and push** immediately with detailed message:
   ```bash
   git add [file]
   git commit -m "Fix: [Description] (Rule #X: [rule name])
   
   Applied [Rule Category] Rule #[Number] which states:
   \"[exact rule text]\"
   
   Changed: [what was changed]
   Reason: [why it violated the rule]"
   
   git push
   ```
6. **Proceed to next change** and repeat until all issues are addressed

### Phase 2: Second Pass Review
After all initial changes are complete:
1. **Perform a comprehensive second review** to check for:
   - Any rules missed in the first pass
   - Compatibility issues introduced by changes (e.g., after JAX conversion, text may need updates)
   - Consistency across the entire document
   - Verify all mathematical notation is correct
   - Ensure code examples still work with changes
2. **Propose any additional changes** found, following the same approval process
3. **Confirm completion** when no more issues are found

### Important Notes:
- **NEVER** implement changes without user approval
- **ALWAYS** propose changes one at a time for clarity
- **ALWAYS** wait for explicit approval before making each change
- **ALWAYS** reference back to the issue summary** when proposing each change
- **ALWAYS** include the FULL detailed reasoning** for each proposal, even if it was mentioned in the summary
- Each commit should address exactly ONE rule violation or issue
- Push after each commit to maintain clear history in the pull request
- Do not skip the detailed explanation even if the issue seems obvious

**FUNDAMENTAL PRINCIPLE**: Only make changes when something is incorrect. Do not modify material unless there are actual errors (grammar, spelling, technical inaccuracies, or style guide violations).

---

## PART A: GENERAL WRITING RULES

### Writing Rule #1: Clarity and Brevity
Keep writing clear and short. The value of the lecture = (importance and clarity of information) ÷ (number of words).

### Writing Rule #2: One-Sentence Paragraphs
**MANDATORY**: Use one sentence paragraphs only. Each paragraph must contain exactly one sentence.

### Writing Rule #3: Short Sentences
Keep those one-sentence paragraphs short and clear. Avoid complex, multi-clause sentences when possible.

### Writing Rule #4: Logical Flow
Ensure logical flow without jumps. Choose carefully what to pay attention to and minimize distractions.

### Writing Rule #5: Simplicity Preference
If you have a choice between two reasonable options, always pick the simplest one.

### Writing Rule #6: Capitalization - General
Don't capitalize words unless you need to (proper nouns, beginning of sentences, or as specified in rules below).

### Writing Rule #7: Mathematical Symbol Choice
Use 𝒫 instead of P when you have the option to choose freely in mathematical notation.

### Writing Rule #8: Visual Presentation
Good lectures look good and use colors and layout to emphasize ideas. Ensure proper formatting for readability.

---

## PART B: TITLES AND HEADINGS RULES

### Title Rule #1: Lecture Titles
Capitalize ALL words in lecture titles.
- Example: "How it Works: Data, Variables and Names" ✓
- Not: "How it works: data, variables and names" ✗

### Title Rule #2: Section/Subsection Headings
Capitalize ONLY the first word and proper nouns in all other headings (sections, subsections, etc.).
- Example: "Binary packages with Python frontends" ✓
- Example: "Adding a new reference to QuantEcon" ✓
- Not: "Binary Packages with Python Frontends" ✗

---

## PART C: EMPHASIS AND FORMATTING RULES

### Format Rule #1: Definitions
Use **bold** for definitions.
- Example: "A **closed set** is a set whose complement is open."

### Format Rule #2: Emphasis
Use *italic* for emphasis.
- Example: "All consumers have *identical* endowments."

### Format Rule #3: Jupyter Book Theme
Lectures are powered by Jupyter Book with the theme quantecon-book-theme. Adhere to conventions for best results.

---

## PART D: MATHEMATICAL NOTATION RULES

### Math Rule #1: Transpose Notation
Use `\top` (renders as ⊤) to represent the transpose of a vector or matrix.
- Correct: A⊤ using `\top`
- Wrong: A' or AT or A^T

### Math Rule #2: Vectors/Matrices of Ones
Use `\mathbb{1}` (renders as 𝟙) to represent a vector or matrix of ones. Always explain it in the lecture (e.g., "Let 𝟙 be an n × 1 vector of ones...").

### Math Rule #3: Matrix Brackets
Matrices ALWAYS use square brackets with `\begin{bmatrix}...\end{bmatrix}`.
- Never use parentheses `\begin{pmatrix}` for matrices

### Math Rule #4: No Bold Face
Do NOT use bold face for either matrices or vectors.
- Wrong: **A**, **x**
- Correct: A, x

### Math Rule #5: Sequence Notation
Sequences use curly brackets: `\{x_t\}_{t=0}^{\infty}`
- Not: (x_t) or [x_t]

### Math Rule #6: Aligned Environment
Use `\begin{aligned}...\end{aligned}` when inside a `$$` math environment.
- Critical for PDF builds
- Never use `\begin{align}` inside `$$` (causes LaTeX build failure)

### Math Rule #7: Equation Numbering
Do NOT use `\tag` for manual equation numbering. Use built-in equation numbering:
```
$$
x_y = 2
$$ (label)
```
Then reference with `{eq}\`label\``

---

## PART E: CODE STYLE RULES

### Code Rule #1: PEP8 Compliance
Follow PEP8 unless there's a good reason to do otherwise (e.g., to match mathematical notation).

### Code Rule #2: Matrix Capitalization
It's acceptable to use capitals for matrices to match mathematical notation.

### Code Rule #3: Operator Spacing
Operators are typically surrounded by spaces: `a * b`, `a + b`
- Exception: Write `a**b` for exponentiation (no spaces)

### Code Rule #4: Unicode Greek Letters
**MANDATORY**: Prefer Unicode symbols for Greek letters commonly used in economics:
- Use `α` instead of `alpha`
- Use `β` instead of `beta`  
- Use `γ` instead of `gamma`
- Use `δ` instead of `delta`
- Use `ε` instead of `epsilon`
- Use `σ` instead of `sigma`
- Use `θ` instead of `theta`
- Use `ρ` instead of `rho`

### Code Rule #5: Package Installation
- QuantEcon lectures should run in a base Anaconda installation
- Any non-Anaconda packages must be installed at the top of the lecture
- Use `tags: [hide-output]` when output is not central
- Example:
```python
!pip install quantecon yfinance --quiet
```

### Code Rule #6: Performance Timing
Use modern `qe.Timer()` context manager, NOT manual timing patterns.
- Correct: `with qe.Timer(): result = computation()`
- Wrong: Using `time.time()` manually or `tic/tac/toc` functions

---

## PART F: JAX-SPECIFIC RULES

### JAX Rule #1: No JAX Installation
Do NOT install JAX at the top of lectures. It may install `jax[cpu]` which is not optimal.

### JAX Rule #2: GPU Admonition
When using JAX with GPU, include the standard GPU admonition warning about hardware acceleration.

### JAX Rule #3: Functional Programming - No Mutation
Functions should NOT modify their inputs. Return new data instead.
- Wrong: `state[0] += shock`
- Correct: `state.at[0].add(shock)`

### JAX Rule #4: Pure Functions
Functions should be deterministic with no side effects.

### JAX Rule #5: Model Structure Pattern
Replace classes with:
1. `NamedTuple` for storing primitives
2. Factory functions for creating instances
3. Collections of pure functions for computations

### JAX Rule #6: No jitclass
Eliminate `jitclass` - use simple `NamedTuple` instead.

### JAX Rule #7: NumPy to JAX Conversion
- Import: `import jax.numpy as jnp` (not `numpy as np`)
- Array creation: `jnp.zeros(10)` (not `np.zeros(10)`)
- Functional updates: `arr.at[0].set(5)` (not `arr[0] = 5`)

### JAX Rule #8: Loop Patterns
Replace explicit loops with JAX constructs:
- Use `jax.lax.scan` when collecting intermediate results
- Use `jax.lax.fori_loop` for simple fixed-iteration loops
- Use `jax.lax.while_loop` for conditional loops

### JAX Rule #9: Random Number Generation
Use JAX random with explicit key management:
```python
import jax.random as jr
key = jr.PRNGKey(42)
shocks = jr.normal(key, (100,))
```

### JAX Rule #10: JAX Transformations
Leverage JAX transformations:
- Use `@jax.jit` for compilation
- Use `vmap` for vectorization
- Use `grad` for automatic differentiation

---

## PART G: EXERCISE AND PROOF RULES

### Exercise Rule #1: Gated Syntax Usage
Use gated syntax (`exercise-start`/`exercise-end`) whenever exercise uses:
- Executable code cells
- Any nested directives (math, note, etc.)

### Exercise Rule #2: Solution Dropdown
Use `:class: dropdown` for solutions by default.

### Exercise Rule #3: Exercise-Solution Pairing
Each exercise admonition MUST be paired with a solution admonition.

### Exercise Rule #4: PDF Compatibility
For PDF builds, use `image` directive (not `figure` directive) when inside another directive like exercise.

### Exercise Rule #5: Nested Directives - Tick Count
When using tick count management (for non-exercise directives):
- Inner directive: 3 ticks (```)
- Outer directive: 4 ticks (````)

### Exercise Rule #6: Directive Support
- `exercise` and `solution`: Support both tick count AND gated syntax
- `prf:` directives (proof, theorem, etc.): Only tick count management
- Standard MyST directives: Only tick count management

---

## PART H: REFERENCE AND CITATION RULES

### Reference Rule #1: Citations
Use the cite role: `{cite}\`bibtex-label\``
- Example: `{cite}\`StokeyLucas1989\``

### Reference Rule #2: Adding References
New references must be added to `<repo>/lectures/_static/quant-econ.bib`

### Reference Rule #3: Internal Links
For same lecture series: Use standard markdown links `[text](filename)`
- Leave text blank to use page title: `[](figures)`

### Reference Rule #4: Cross-Series Links
For different lecture series: Use `{doc}` links with intersphinx
- Example: `{doc}\`intro:linear_equations\``

---

## PART I: INDEX ENTRY RULES

### Index Rule #1: Inline Index Entries
Use `:index:` for unchanged keywords: `{index}\`bellman equation\``

### Index Rule #2: Index Directives
Use directives for complex arrangements or nested entries:
```
```{index} single: Dynamic Programming; Bellman Equation
```
```

### Index Rule #3: Case Sensitivity
Index items are case sensitive - maintain consistency.

---

## PART J: BINARY PACKAGE RULES

### Binary Rule #1: graphviz Package
If using graphviz:
1. Install with `pip install graphviz` at lecture top
2. Check `.github/workflows/ci.yml` for preview builds
3. Add note admonition about local installation requirements

---

## REVIEW CHECKLIST WITH RULE REFERENCES

When reviewing, check each item and reference the specific rule when proposing changes:

### Writing and Formatting
- [ ] One-sentence paragraphs (Writing Rule #2)
- [ ] Short, clear sentences (Writing Rule #3)
- [ ] Logical flow without jumps (Writing Rule #4)
- [ ] Minimal capitalization (Writing Rule #6)
- [ ] Bold for definitions (Format Rule #1)
- [ ] Italic for emphasis (Format Rule #2)

### Titles and Headings
- [ ] Lecture titles: all words capitalized (Title Rule #1)
- [ ] Section headings: only first word capitalized (Title Rule #2)

### Mathematical Notation
- [ ] Transpose uses `\top` (Math Rule #1)
- [ ] Vectors of ones use `\mathbb{1}` (Math Rule #2)
- [ ] Matrices use square brackets (Math Rule #3)
- [ ] No bold face for matrices/vectors (Math Rule #4)
- [ ] Sequences use curly brackets (Math Rule #5)
- [ ] `aligned` inside `$$` environments (Math Rule #6)
- [ ] No manual equation numbering with `\tag` (Math Rule #7)

### Code Style
- [ ] PEP8 compliance (Code Rule #1)
- [ ] Unicode Greek letters (Code Rule #4)
- [ ] Proper operator spacing (Code Rule #3)
- [ ] Package installation at top (Code Rule #5)
- [ ] `qe.Timer()` for timing (Code Rule #6)

### JAX-Specific (if applicable)
- [ ] No JAX installation command (JAX Rule #1)
- [ ] Pure functions, no mutation (JAX Rules #3, #4)
- [ ] NamedTuple instead of jitclass (JAX Rules #5, #6)
- [ ] JAX arrays and operations (JAX Rule #7)
- [ ] JAX loop constructs (JAX Rule #8)
- [ ] Explicit PRNG keys (JAX Rule #9)

### Exercises and Solutions
- [ ] Gated syntax when needed (Exercise Rule #1)
- [ ] Dropdown solutions (Exercise Rule #2)
- [ ] All exercises have solutions (Exercise Rule #3)
- [ ] Proper tick count for nesting (Exercise Rule #5)

---

## HOW TO REFERENCE RULES

When proposing changes, use this format:

"According to [Rule Category] Rule #[Number], [current text/code] should be changed to [proposed text/code] because [brief explanation]."

Examples:
- "According to Writing Rule #2, this multi-sentence paragraph should be split into separate one-sentence paragraphs."
- "According to Code Rule #4, `alpha` should be changed to `α` to use Unicode Greek letters."
- "According to Math Rule #1, `A'` should be changed to `A^\top` for proper transpose notation."
- "According to Title Rule #2, 'Binary Packages With Python Frontends' should be 'Binary packages with Python frontends' (only first word capitalized)."

---

## IMPORTANT REMINDERS

1. **Only fix actual errors** - Don't change stylistic preferences unless they violate rules
2. **Preserve author intent** - Don't add or remove content unless fixing errors
3. **Document all changes** - Reference specific rule numbers for transparency
4. **Check technical accuracy** - Ensure mathematical and code correctness
5. **Maintain consistency** - Apply rules uniformly throughout the lecture

---

## GIT COMMIT MESSAGE FORMAT

When committing changes after review, use this format for the commit message:

```
Fix: [Brief description] per QuantEcon style guide

Applied rules:
- Title Rule #2: Section headings capitalization
- Writing Rule #2: One-sentence paragraphs
- Code Rule #4: Unicode Greek letters

Details:
- Changed "The Algorithm" to "The algorithm" (Title Rule #2)
- Split multi-sentence paragraphs (Writing Rule #2)
- Replaced "alpha" with "α" in code (Code Rule #4)
```

For single-rule fixes, use:
```
Fix: [Description] (Title Rule #2: section heading capitalization)
```

Example:
```
Fix: Correct section heading capitalizations (Title Rule #2)

Changed 3 section headings to follow Title Rule #2 which states:
"Capitalize ONLY the first word and proper nouns in all other headings"

Changes:
- "The Algorithm" → "The algorithm"
- "Value Function Iteration" → "Value function iteration"  
- "Fitted Value Function Iteration" → "Fitted value function iteration"
```