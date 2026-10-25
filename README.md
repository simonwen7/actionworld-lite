# ActionWorld Lite

**A Lightweight Interactive World Model for Physical AI**

## Overview

ActionWorld Lite is an independent research and engineering project exploring how small, action-conditioned AI models can learn and predict physical interactions in an interactive virtual environment.

The long-term goal is to build a lightweight world simulator where an agent can perform actions, observe model-predicted outcomes, and compare those predictions with a physics-based reference environment.

This project investigates a small part of the broader research problem behind interactive world models for embodied intelligence.

## Project Goals

- Develop a reproducible physical interaction environment.
- Collect and validate action-conditioned training data.
- Train a lightweight learned world model.
- Evaluate predictive accuracy and multi-step rollout behavior.
- Investigate one measurable modeling or evaluation improvement.
- Build an interactive demonstration of the trained model.
- Document experiments, results, limitations, and technical decisions.

## Research

Research is a first-class component of this project.

The [research directory](research/README.md) will contain research questions, hypotheses, experimental methods, evidence, results, and limitations.

Experimental claims will be supported by reproducible measurements wherever possible.

## Development Roadmap

| Milestone | Focus |
| --- | --- |
| M1 | Project Setup & Research Foundation |
| M2 | Physical Simulation Environment |
| M3 | Dataset Generation & Baseline |
| M4 | AI World Model Training |
| M5 | Interactive Prediction & Evaluation |
| M6 | Research Innovation & Experiments |
| M7 | Interactive Demo & Final Release |

Each milestone follows three stages:

1. READ / CHECK
2. WRITE / IMPLEMENT
3. FINAL TEST / ACCEPTANCE

## Project Principles

- AI-based prediction must be a genuine part of the system.
- All technical claims must distinguish intended functionality from verified results.
- Failed experiments and limitations must be documented honestly.
- Implementation decisions must be evidence-driven.
- The initial project must remain feasible without paid cloud GPU infrastructure or physical robot hardware.
- Research code, simulation, evaluation, and demo components will be maintained in one repository.

## Current Status

**M1 — Project Setup & Research Foundation**

The repository is being initialized.

The simulation framework, model architecture, dataset strategy, and research innovation are not yet finalized.

No model training or performance results are claimed at this stage.

## Development Workflow

ChatGPT authors project source code, file contents, and implementation instructions.

Cursor is used as a controlled file operator and test executor. It must not independently generate implementation code or make unapproved technical decisions.

Git commits and pushes are controlled manually by the repository owner.

## License

A project license has not yet been selected.
