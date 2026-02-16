# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kalshi data analysis repository focused on evaluating LLM performance on structured prediction tasks with quantifiable ground truth. The project compares Claude (Haiku, Sonnet, Opus) and Gemini models across three domains: Golden Globes predictions, box office revenue estimation, and VES Awards predictions.

## Tech Stack

- **Python 3** with Jupyter notebooks
- **Data**: pandas, polars
- **Visualization**: matplotlib, seaborn
- **Environment**: Google Colab with Google Drive for data storage
- No formal build/test/lint pipeline or dependency management

## Architecture

The entire analysis lives in `GradStudentEval.ipynb`. The notebook follows this flow:

1. **Data cleaning** — `clean_award_data()` normalizes titles, categories, and award names
2. **Golden Globes analysis** — Compares LLM predictions against canonical Best Picture winners (2000–2024)
3. **Box office projections** — Evaluates revenue predictions using rounded accuracy and closeness thresholds (1%, 5%)
4. **VES Awards analysis** — Tests nomination count and outstanding award prediction accuracy

Data sources are TSV files stored in Google Drive under `/content/gdrive/My Drive/Colab Notebooks/data/grad_student_eval/` with subdirectories `picture/`, `boxoffice/`, and `ves_awards/`. Data is not committed to the repo.

## Running

Open `GradStudentEval.ipynb` in Google Colab. The notebook mounts Google Drive to access input data files.

## License

GNU General Public License v3.
