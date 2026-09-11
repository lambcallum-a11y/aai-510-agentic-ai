# Elite Athletic AI Advisor

Final project for **AAI 510: Agentic AI Systems** in the University of San Diego MS in Applied Artificial Intelligence program.

## Project

Elite Athletic AI Advisor is a coach- and parent-supervised planning assistant for youth basketball athletes. It uses athlete profiles, fitness benchmarks, exercise data, nutrition guidance, and safety rules to draft weekly training and recovery plans.

The course implementation includes a Databricks data pipeline, agent tools, model-serving calls, MLflow traces, safety checks, and evaluation across two language-model endpoints.

## Repository map

- Root notebooks, Python files, CSVs, and the commentary document preserve the submitted team project.
- `portfolio_extension/` contains **Callum Lamb's post-submission portfolio extension**. It organizes the pipeline, agent, evaluation, and business case into a clearer runnable sequence.

Run the extension in this order:

1. `portfolio_extension/01_data_pipeline.py`
2. `portfolio_extension/02_agent_definition.py`
3. `portfolio_extension/03_evaluate_multi_llm.py`

The extension supports mock-model execution for development. Live execution requires Databricks, MLflow, configured model-serving endpoints, and appropriate workspace permissions.

## Safety scope

The prototype supplies educational training, recovery, and general food guidance. It does not provide medical advice, supplement plans, fasting plans, extreme diets, or weight-cutting guidance. Injury-related recommendations require human review.

## Provenance and reuse

This is a provenance-preserving fork of the team repository, with original contributions associated with GitHub users `Beakal-23` and `J2NM`. The `portfolio_extension/` work is attributed to **Callum Lamb**. No license was supplied with the upstream project, so public visibility does not grant reuse rights; see `NOTICE.md`.
