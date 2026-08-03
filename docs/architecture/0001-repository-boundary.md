# 0001 — Repository boundary

## Status

Accepted (bootstrap)

## Context

QuantSilico maintains multiple repositories. This competition effort must stay
isolated from the public portfolio and the main QuantSilico product.

## Decision

`quantsilico-generals-competition` is a **private**, standalone repository for:

- private bots and tactics;
- evaluation harnesses;
- training systems and experiment logs;
- model weights (gitignored until deliberately exported);
- submission packaging (not uploaded during bootstrap).

It must remain separate from:

- `priestlyadejo-portfolio`
- `quantsilico`
- `quantsilico-quanthack-competition`
- other QuantSilico competition archives

## Consequences

- Do not copy active tactics into the portfolio.
- Do not modify the main QuantSilico product from this tree.
- Do not upload competition submissions unless a later task explicitly requires it.
- Treat this repository as confidential competition IP.
