# SecretMafia

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/SecretMafia)

## Description

**SecretMafia** is an environment for evaluating agents on social deduction, persuasion, and strategic deception in a Mafia-style game. This environment wraps the SecretMafia implementation from [TextArena](https://github.com/LeonGuertler/TextArena), a framework for text-based game environments.

## Capabilities

- Social deduction and role inference
- Persuasive argumentation and defense
- Strategic voting and coalition building
- Deception detection and counter-deception

## Compute Requirements

SecretMafia does not require a sandbox. It has minimal compute requirements.

## License

[MIT](https://github.com/LeonGuertler/TextArena/blob/main/LICENSE).

## Tasks

There are two splits: train (150 tasks) and test (150 tasks). Each split contains 50 tasks across each of 3 variants:

- **SecretMafia-v0**
- **SecretMafia-v0-train**
- **SecretMafia-v0-raw**

Each task is seeded for reproducibility.

## Reward Structure

This is a sparse reward environment. Rewards are mapped from TextArena's native range of {-1, 0, 1} to {0.0, 0.5, 1.0} via `(raw + 1) / 2`.

We do not use LLM graders for this environment; reward is determined programmatically.

## Data

Game state is generated procedurally by the TextArena engine using seeded randomness. No external data files are required.

## Tools

Agents are given a single tool:

- `send_message(message)`: Send a message, vote [vote X], or night action [kill X].

## Time Horizon

SecretMafia is a multi-turn environment.

## Environment Difficulty

High. Agents must navigate complex social dynamics with hidden information, persuading others and identifying opponents while concealing their own role, across alternating day and night phases with 6 players.

## Other Environment Requirements

This environment requires an OpenAI API key (passed via secrets) to power the LLM opponents.

## Safety

Agents in SecretMafia interact only with a social deduction game and have no access to external systems, the internet, or sensitive data. The environment does not present safety risks.

## Citations

```bibtex
@software{textarena2024,
  author    = {Guertler, Leon and Banting, Wilfried and Pignatelli, Eduardo},
  title     = {TextArena},
  year      = {2024},
  publisher = {GitHub},
  url       = {https://github.com/LeonGuertler/TextArena}
}
```
