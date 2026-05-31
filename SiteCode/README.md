# SiteCode Local Development

This folder now contains a minimal Vite + Vue setup so you can run the experiment sites locally with npm.

## Prerequisites

You need Node.js, which includes npm.

Install it using one of these options:

1. Official installer: download the LTS version from https://nodejs.org
2. Homebrew on macOS:

```bash
brew install node
```

After installation, verify it:

```bash
node -v
npm -v
```

## Install dependencies

From this folder:

```bash
npm install
```

## Run the local dev server

From this folder:

```bash
npm run dev
```

Vite will print a localhost URL, usually:

```text
http://localhost:5173/
```

## Experiment routes

Use these pages in the browser:

- Home page: `http://localhost:5173/`
- Static experiment entry: `http://localhost:5173/staticExplain`
- Static experiment study page: `http://localhost:5173/staticExplain/study`
- Interactive experiment entry: `http://localhost:5173/baseExplain`
- Interactive experiment study page: `http://localhost:5173/baseExplain/study`

## Model and visualization pipeline

The experiment pages are built around a BERT-style masked language model trained on Wikipedia and books. The explanatory content and interactive examples use a fill-in-the-blank workflow: a sentence is converted into subword tokens with a WordPiece-style tokenizer, lower-cased, and wrapped with the standard BERT boundary markers `[` `CLS` `]` and `[` `SEP` `]` before being sent through the completion pipeline. For masked-token probes, the site requests the top candidate token predictions and associated probabilities, then renders those ranked completions back into the UI. The visualizations are drawn client-side with JavaScript and D3, combining HTML controls with SVG charts to show token completions, paired sentence comparisons, difference plots, and the gender-over-time views. In practice, the Vue pages load the explanatory markdown, then sequentially load the legacy experiment scripts in `source/`, `source-static/`, and `third_party/`, which transform the model outputs into the interactive graphics shown to participants.

## Optional API configuration

The Vue pages submit responses to `/api/...` by default.

- If you already have a backend running on `http://localhost:3001`, the Vite dev server proxies `/api` requests there automatically.
- If you want to use a different backend URL, start Vite with `VITE_API_URL` set:

```bash
VITE_API_URL=http://localhost:4000 npm run dev
```

## Production build

```bash
npm run build
npm run preview
```