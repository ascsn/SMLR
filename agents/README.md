# Agent Library

This folder collects reusable agent roles for scientific coding and software development.
Each agent card defines what the role is for, when to use it, what inputs help most, and what output to ask for.

## Categories

- `core-development`: implementation, debugging, and code review.
- `scientific-numerical`: numerical reliability, physics validation, and data provenance.
- `hpc-workflow`: cluster scripts, reproducibility, and workflow robustness.
- `documentation-communication`: docs and notebooks.
- `project-hygiene`: dependencies and release management.
- `software-package-development`: package architecture, APIs, tests, CLIs, artifacts, and packaging audits.

## Usage Pattern

Use one agent for one clear job. Give it the relevant files, expected behavior, constraints, and the exact output you want. For larger work, run agents in stages: inspect, implement, verify, then document.
