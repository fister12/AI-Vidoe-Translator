---
name: feature-integration-architecture-review
description: 'Analyze a proposed feature, technology, or approach for integration into an existing project. Use for system design trade-off analysis, integration planning, architecture impact assessment, risk mitigation, and decision-ready technical recommendations.'
argument-hint: 'Provide: current system context, proposed feature/tech, constraints, and success criteria.'
user-invocable: true
disable-model-invocation: false
---
# Feature Integration Architecture Review

Provide an engineering-focused, practical integration assessment from a senior software architect perspective.

## When to Use
- Evaluating whether a new feature or technology should be adopted.
- Planning migration or integration into an existing codebase.
- Producing an architecture decision brief with explicit trade-offs.
- Comparing candidate approaches before implementation.

## Required Inputs
Collect these before writing the analysis. If any are missing, state assumptions explicitly.
- Existing system summary: architecture, modules, data flow, deployment model, scale.
- Tech stack and constraints: language/runtime, infrastructure, compliance, budget, timeline.
- Proposed change: feature/technology scope and intended outcomes.
- Non-functional goals: latency, throughput, reliability, security, maintainability.

## Procedure
1. Understand the Existing System
- Summarize current architecture, stack, and operational constraints.
- Identify integration boundaries and ownership (backend, frontend, data, infra).
- List assumptions for missing details and label confidence level.

2. Break Down the Proposed Feature/Technology
- Explain what it is, how it works internally, and why teams adopt it.
- Map the concrete problems it solves in this project context.
- State prerequisites and fit criteria.

3. Define Integration Strategy
- Provide a step-by-step rollout path (MVP, pilot, phased adoption, full rollout).
- Specify required changes by layer:
  - Architecture and service boundaries
  - Database schema, migrations, and data lifecycle
  - API contracts, versioning, and compatibility strategy
  - Frontend or client changes and UX implications
  - CI/CD, observability, and operational runbooks
- Identify dependencies, tooling, and environment changes.

4. Evaluate Pros and Cons
- Advantages: performance, scalability, delivery speed, developer experience, resilience.
- Disadvantages: complexity, operational burden, vendor lock-in, cost, migration risk.
- Tie each point to project constraints instead of generic claims.

5. Compare Alternatives
- Propose 1 to 2 realistic alternatives (including a conservative baseline option).
- Compare options using a simple decision matrix:
  - Implementation effort
  - Time to value
  - Risk profile
  - Total cost of ownership
  - Long-term flexibility
- Explain why the preferred option wins under current constraints.

6. Assess Scalability and Future Impact
- Analyze effects on horizontal/vertical scaling, team autonomy, and architecture evolution.
- Describe impact on maintainability, testability, and onboarding complexity.
- Note whether this enables or blocks likely future roadmap items.

7. Identify Risks and Mitigations
- List top failure modes across technical, operational, and organizational dimensions.
- For each risk, provide mitigation, owner, and early warning signal.
- Include rollback and incident-response considerations.

8. Give a Final Verdict
- Decide: adopt now, adopt with conditions, pilot first, or reject.
- Justify with explicit trade-offs and assumptions.
- Provide immediate next steps (first 2 to 5 engineering actions).

## Decision Branching
Use this logic while writing recommendations.
- If core constraints are unknown, continue with clearly labeled assumptions and request missing data.
- If migration risk is high and reversibility is low, recommend pilot-first with exit criteria.
- If benefits are marginal versus complexity, recommend simpler alternative or deferment.
- If compliance/security constraints are strict, prioritize approaches with stronger control and auditability.

## Quality Checks Before Finalizing
- The response contains all 8 required sections in order.
- Every recommendation is tied to project-specific constraints.
- Trade-offs are explicit and quantified when possible.
- Risks include practical mitigations and rollback thinking.
- Final verdict is clear, not ambiguous.
- The output is actionable, with concrete next steps.

## Output Format
Use this section structure exactly:
1. Understanding of the Existing System
2. Feature/Technology Breakdown
3. Integration Strategy
4. Pros and Cons of This Approach
5. Alternative Approaches
6. Scalability and Future Impact
7. Risks and Mitigation
8. Final Verdict

Writing style requirements:
- Practical and engineering-focused.
- Avoid vague claims; use real-world reasoning and trade-offs.
- Prefer concise depth over verbosity.
