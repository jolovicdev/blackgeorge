# AGENTS.md Guide

This document describes the `AGENTS.md` file in the project root, which serves as the source of truth for AI coding agents working in this repository.

## What is AGENTS.md?

`AGENTS.md` is an emerging standard file (similar to `README.md`) specifically designed to be read by AI coding agents. While a `README.md` explains the project to humans, an `AGENTS.md` explains the project to AIs—covering folder structure, coding conventions, specific commands, and architectural rules.

## Location

The `AGENTS.md` file is located at the project root: `/AGENTS.md`

## Contents

The `AGENTS.md` file contains:

1. **Project Overview** — High-level summary, key features, core primitives (Desk, Worker, Workforce, Workflow), and design philosophy

2. **Tech Stack & Dependencies** — Python 3.12+, uv, Pydantic v2, Instructor, LiteLLM, ChromaDB, MCP, and development tools (ruff, mypy, pytest, mkdocs)

3. **Architecture & Directory Structure** — Complete folder structure explaining what goes where, and the design patterns used (composition, adapters, repositories, event-driven, state machine)

4. **Coding Standards & Conventions** — Naming conventions, strict typing rules, code style (no comments/docstrings in code), patterns to follow and avoid

5. **Development Workflow** — Exact commands for installing, testing, linting, formatting, type checking, and serving docs

6. **Critical Rules** — Repository hygiene, code quality requirements, commit message conventions (Conventional Commits), and API design principles

## For AI Agents

When working on this codebase:

1. **Read AGENTS.md first** — It contains the context needed to write consistent, high-quality code
2. **Follow the pre-flight checklist** — Run ruff, mypy, and pytest before committing
3. **Respect the coding standards** — No comments/docstrings, strict typing, clear naming
4. **Use uv for dependencies** — Never use pip directly
5. **Update docs when changing behavior** — The docs/ folder contains MkDocs markdown files

## Updates

When the project structure, tech stack, or conventions change, update `AGENTS.md` to keep it accurate for future agents.

## External Resources

- [agents.md](https://agents.md/) — The AGENTS.md standard website
