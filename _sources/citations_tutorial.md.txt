# Citation Management Tutorial

This tutorial shows how to manage `.bib` citations in Annolid from chat, GUI, and CLI.

## 1. Save Citation From Active Paper

Use Annolid Bot chat input:

- `save citation`
- `save citation from pdf`
- `save citation from web`
- `save citation from pdf as mypaper2024 to references.bib`

Behavior:

- Annolid extracts metadata from the open PDF/web page.
- It validates metadata online (Google Scholar first, then fallback providers).
- It writes/updates the BibTeX entry in your target `.bib` file.

## 2. Add Raw BibTeX Directly

You can paste full BibTeX in chat:

```text
add citation @article{yang2024annolid, title={Annolid: Annotate, Segment, and Track Anything You Need}, author={Yang, Chen and Cleland, Thomas A}, journal={arXiv preprint arXiv:2403.18690}, year={2024}}
```

Optional target file:

```text
add citation @article{yang2024annolid, ...} to references.bib
```

## 3. List Existing Citations

From chat:

- `list citations`
- `list citations from references.bib`
- `list citations from references.bib for annolid`

## 4. Use Citation Manager UI

Open Annolid Bot panel and click the citation button:

- Browse/select `.bib` file.
- Save from PDF/Web/Auto.
- Turn on/off `Auto validate before save`.
- Use `Strict` to reject weak metadata matches.
- Edit rows inline and click `Save Row Edits`.
- Use `Source` column to store URL or local PDF path.
- Remove selected entries.

## 5. CLI Workflow

```bash
annolid-run citations-list --bib-file references.bib
annolid-run citations-upsert --bib-file references.bib --key yang2024annolid --title "Annolid: Annotate, Segment, and Track Anything You Need" --author "Yang, Chen and Cleland, Thomas A" --year 2024
annolid-run citations-remove --bib-file references.bib --key yang2024annolid
annolid-run citations-format --bib-file references.bib
```

## 6. Tips

- Keep citation keys stable when possible.
- Prefer DOI-backed entries for best metadata quality.
- If strict validation blocks a save, disable strict mode and retry.
