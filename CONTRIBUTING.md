# Contributing to Anvil

Thank you for your interest in contributing to Anvil. This project is maintained by a solo developer with limited time, so contributions need to be thoughtful, well-scoped, and considerate of the broader design of the addon.

## Core Philosophy

Before contributing, ask yourself:

* Do I fully understand the problem space?
* Have I considered alternative approaches and trade-offs?
* Am I adding meaningful value beyond what AI tools alone could generate?

Noting your considerations in a PR will make it more likely to get merged.

## Before You Start

For any non-trivial feature, you should open an issue and discuss it before starting work.

If you are unsure whether something should be implemented, assume it shouldn’t be and ask first.

Pull requests that introduce significant changes without prior discussion are likely to be rejected, even if they are technically correct and seem to align with the core values of Anvil.

## Types of Contributions

### 1. Bug Fixes (Welcome)

Bug fixes are always welcome.

A bug is defined as behavior that does not match the expected functionality of the addon. These are typically straightforward to validate and do not require extensive discussion.

### 2. Minor Changes (Generally Not Accepted)

Minor changes include:

* Changing defaults
* Modifying hotkeys
* Small behavior tweaks

These often involve subjective trade-offs. Allowing these changes would invite back and forth, so they are usually rejected.

### 3. Complex Features (Conditionally Accepted)

These are features where:

* The problem is well-defined
* The implementation is non-trivial (AI cannot one-shot it)

Examples include complex geometry operations (e.g., something like a cylinder cut).

These contributions are welcome, but:

* Expect detailed review
* Ensure the feature aligns with the project’s direction by discussing beforehand

### 4. Vague or Exploratory Features (Use Caution)

These are features where:

* The scope is unclear
* The design requires exploration

Examples include introducing new material workflows (e.g. supporting PBR materials).

Even if the idea seems obviously useful:

* Approval is not guaranteed
* Review may take significant time
* Additional iteration will likely be required

## AI Usage

AI tools are allowed and encouraged—but only as a tool.

If your contribution consists of:

* Copying feedback into AI
* Returning the generated result without deeper reasoning

Then you are not adding meaningful value.

Good contributions should demonstrate:

* Understanding of the problem
* Consideration of alternatives
* Awareness of trade-offs

## Responsibility & Maintenance

If you introduce a feature, you are the primary owner of this feature.

* If your change introduces a bug, you must acknowledge your intention to fix or not within 48 hours
* Failure to do so may result in being banned from contributing

This policy exists to prevent contributors from creating ongoing maintenance burden for the project.

## Testing Guidelines

Tests are required when appropriate, especially for:

* Geometry-related functionality
* Texture application
* Areas of the codebase that frequently change

Guidelines:

* Aim to simulate real-world usage as closely as possible
* Avoid brittle tests (e.g., mouse interaction tests)
* Focus on stability and meaningful coverage; there are many cases where tests are not required

## Code Style

All contributions must match the existing code style and patterns used in the repository.

Do not introduce new stylistic conventions without discussion.

## Documentation Requirements

If your change affects users, you must update the README.

This includes:

* New features
* New or changed hotkeys
* Behavioral changes

**Important:**

* Do NOT use AI to update the README
* Documentation should reflect real understanding of the feature

## Hotkeys

If your feature introduces functionality that benefits from a hotkey:

* You must include one where appropriate

## Review Expectations

* Reviews may be slow
* Pull requests may be rejected even if they are technically correct
* PRs that were not discussed beforehand are especially likely to be rejected
* Alignment with Anvil’s design philosophy is more important than correctness alone

## Summary Checklist

Before submitting a PR:

* [ ] Is this a bug fix or a meaningful feature?
* [ ] Have I opened an issue for non-trivial work?
* [ ] Have I considered trade-offs and alternatives?
* [ ] Am I adding value beyond AI-generated code?
* [ ] Does my code match the existing style?
* [ ] Have I added tests where appropriate?
* [ ] Have I updated the README (without AI)?
* [ ] Have I added hotkeys if needed?
* [ ] Am I prepared to maintain this feature?

For your first PR - please include this checklist in the PR description
