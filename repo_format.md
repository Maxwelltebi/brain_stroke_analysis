# Reusable Project Showcase README Prompt

Use the following prompt whenever creating or polishing a project's README.

---

Create or polish this repository's `README.md` as a professional project showcase. Follow the visual structure of Mintahandrews' Football Predictor introduction: https://github.com/Mintahandrews/football-predictor. The required layout is specified below so this prompt works without access to the reference.

First inspect the repository's code, configuration, scripts, existing documentation, and available images. Establish what the project actually does, its technology stack, how to run it, and what results are supported by evidence. Then write the README directly. Preserve useful, accurate existing information.

## Mandatory header and overview format

Always use this exact structure and ordering. Replace bracketed placeholders with project-specific content; preserve the HTML alignment, heading levels, spacing between blocks, badge row, separator, and two-paragraph overview. Do not place a table of contents, demo link, screenshot, or other section before or between these elements.

```markdown
<h1 align="center">[Project Name]</h1>

<h3 align="center">[One-line tagline describing the core capability and its defining technology or approach]</h3>

<p align="center">
  [One concise sentence explaining what users can accomplish and highlighting the most important outputs or features.]
</p>

<p align="center">
  <img src="[Language badge URL]" alt="[Language and verified version]" />
  <img src="[Framework badge URL]" alt="[Framework and verified version]" />
  <img src="[Other core technology badge URL]" alt="[Technology and verified version]" />
  <a href="LICENSE"><img src="[License badge URL]" alt="[Actual license]" /></a>
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat" alt="PRs welcome" />
</p>

---

## Overview

**[Project Name]** is a [type of application or tool] that uses **[core technology, model, or approach]** to [solve a specific problem for its intended users]. It [explains the main workflow or distinguishing capability], providing **[key benefit or output]**.

[Explain how the core implementation works in two or three lines, emphasizing **the central algorithm, workflow, or architectural choice**. Connect the inputs and processing to the outputs, adding useful technical detail without repeating the first paragraph.]
```

The visual contract is:

- A centered project title rendered as H1, with GitHub's normal heading underline.
- A centered, bold H3 tagline beneath it.
- A centered plain-text summary beneath the tagline, normally one or two rendered lines.
- One centered row of compact, flat Shields.io badges, ordered by language, framework, other core technologies, license, and contribution status.
- A horizontal separator after the badges.
- A left-aligned H2 titled exactly `Overview`.
- Two left-aligned paragraphs with selective bold emphasis: the first explains the product and its value; the second explains the implementation.

Use GitHub-compatible HTML and Markdown; do not add custom CSS, decorative banners, emojis in these headings, or alternate header layouts. Exact line wrapping depends on screen width. Keep the structure identical across projects while adapting all wording and technologies to the actual repository.

Use about four to seven relevant badges when supported. Use Shields.io's `flat` style and appropriate technology logos where available. Verify versions from repository manifests or lockfiles; omit version labels when unknown. Include a license badge only when the repository has that license, and link to its actual file. Include `PRs welcome` only when contributions are welcome. These factual omissions are permitted; do not invent badges to fill the row.

## Remaining showcase structure

After the overview, use the following order, adapting sections to the project's scope:

1. **Demo and Screenshots** — Show the actual interface or representative output with concise captions. Link to a verified live demo if available. Use repository-relative image paths where possible.
2. **Features** — Describe concrete user capabilities and distinguishing behavior. Use short bullets or a feature/description table.
3. **Tech Stack** — Use a concern/technology/purpose table so readers understand each technology's role.
4. **Project Structure** — Include a concise, annotated directory tree based on the actual repository. Explain the main components and data flow where useful.
5. **Quick Start** — List prerequisites, then give ordered, copyable setup and run commands, required configuration, and the expected successful outcome. Identify working directories and operating-system-specific commands when relevant.
6. **How It Works** — Explain the core workflow, architecture, or model. For machine learning projects, cover data, preprocessing, training or inference, and outputs as applicable. Define equations and parameters when they help the reader.
7. **Results and Limitations** — Present measured results, evaluation conditions, representative outputs, and known constraints. Distinguish completed evaluation from planned evaluation.
8. **Usage / API Reference** — Give practical examples. Include endpoint or command tables only if the project exposes them.
9. **Configuration** — Document actual settings, required versus optional values, defaults, and an example environment file if one exists. Never include secrets.
10. **Development and Deployment** — Include supported build, test, and deployment commands. Link to a separate deployment guide when details are extensive.
11. **Roadmap** — Use unchecked tasks for genuine planned work, clearly separated from existing features.
12. **Contributing, License, and Acknowledgements** — State the actual contribution process and license, and credit relevant datasets, research, tools, or assets.

Omit sections that genuinely do not apply. Keep the header and overview structure mandatory. Put extensive operational instructions or specifications into separate documents when helpful, and link them from the README.

## Writing and evidence standards

- Make the project understandable at a glance, then provide enough detail to run and assess it.
- Use clear, specific language, short paragraphs, purposeful bold emphasis, readable tables, and fenced code blocks with language labels.
- Present implemented behavior accurately. Do not describe planned features as completed or make unsupported claims about accuracy, speed, reliability, or production readiness.
- Never fabricate screenshots, demos, benchmark numbers, licenses, tests, links, commands, or technology versions.
- Do not leave placeholder text or empty screenshot sections in the finished README. Report missing assets or evidence in your completion message instead.
- Keep every command and file reference consistent with the repository. Verify setup steps where practical, and state which steps could not be tested.
- For each project, emphasize its own problem, implementation decisions, and demonstrated results. Reuse this presentation format, not generic promotional wording.

## Completion checks

Check that the opening matches the mandatory layout, all placeholders are replaced, links and image paths are valid, badge labels are accurate, and code blocks and tables render correctly. Preview the rendered README if a suitable preview is available. Confirm that the README clearly answers: what is it, why is it useful, what does it look like, how does it work, and how can someone run it?

Finish with a short summary of the documentation changes and any missing screenshots, results, or verification that still require attention.
