# Portfolio extension

This directory contains Callum Lamb's post-submission extension to the Elite Athletic AI Advisor course project. It is separated from the submitted team artifacts at the repository root.

## Run order

1. `01_data_pipeline.py` builds Bronze, Silver, and Gold Databricks tables.
2. `02_agent_definition.py` defines the tools, intake flow, safety checks, and two-model review pattern.
3. `03_evaluate_multi_llm.py` compares model endpoints across evaluation cases and records MLflow metrics.
4. `demo_and_business_case.md` documents the demo, human-review policy, ROI variables, and deployment considerations.

## Environment

Run the scripts in Databricks with Unity Catalog, MLflow, and model-serving permissions. The evaluation supports `allow_mock_llm=true` for local classroom development. Live runs require configured endpoints and current provider pricing inputs.

The prototype supports youth basketball planning with coach or parent oversight. It provides educational guidance only and rejects medical, supplement, fasting, extreme-diet, and weight-cutting requests.
