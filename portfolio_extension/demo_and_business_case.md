# Elite Athletic AI Advisor Demo and Business Case

## Product Positioning

Elite Athletic AI Advisor is a coach- and parent-supervised basketball
development assistant. The first supported market is youth athletes ages 13-17.
The agent generates safe weekly training, recovery, and general nutrition plans
using athlete profile data, basketball benchmarks, exercise recommendations, and
safety rules.

The product should not be positioned as a coach replacement. The stronger
position is: every athlete gets a safer first-draft development plan, while
coaches and parents remain involved in review and accountability.

## Functionality Built

- Ingests project CSVs into Bronze, Silver, and Gold Databricks tables.
- Creates agent-ready tables for athlete profiles, exercises, basketball
  benchmarks, nutrition guidance, safety rules, progress metrics, and feedback
  schema.
- Generates personalized weekly basketball plans from athlete profile fields:
  age, position, experience level, goal, injury status, availability, and
  equipment.
- Uses tool functions for profile lookup, safety checking, exercise retrieval,
  benchmark lookup, nutrition lookup, progress metrics, and feedback schema.
- Rejects irrelevant or unsafe requests before generation.
- Supports the core youth MVP plus a limited adult preview for ages 18+.
- Logs evaluation cases across two LLM endpoints with MLflow.
- Captures pass rate, judge scores, latency, estimated cost, human-review load,
  coach time saved, and ROI inputs.

## Design Decisions

- Youth-first scope remains the main MVP because minors require more explicit
  safety constraints and parent or coach oversight.
- Adult athletes are handled as a preview segment rather than rejected outright.
  This answers the product question "what about 18+?" without weakening the
  class proposal.
- Safety checks run before LLM generation so unsafe requests are blocked
  deterministically.
- Nutrition guidance is intentionally conservative because the available USDA
  subset is not rich enough for detailed meal planning. The agent avoids
  supplements, fasting, weight cutting, and therapeutic diet advice.
- MLflow is used for traces and evaluation because it is native to Databricks
  and satisfies the final project requirement for an established trace provider.
- The two-LLM comparison is structured around both product quality and business
  cost, not only model preference.
- Progress metrics and feedback schema were added to create a path toward
  proprietary data collection after a pilot.

## Demo Script

1. Start with the problem:
   Families and youth athletes often rely on fragmented advice from social media,
   generic fitness apps, or unverified training routines. Coaches are stretched
   thin and cannot always create individualized weekly plans.

2. Show the data pipeline:
   Run `01_data_pipeline.py` and show the Gold tables for athlete profiles,
   exercise recommendations, safety rules, and progress metrics.

3. Show the agent:
   Run `02_agent_definition.py` with an athlete such as `A003`, a youth athlete
   with knee recovery. Explain that the agent calls tools before generating a
   plan.

4. Show the output:
   Highlight the safety note, weekly plan, nutrition guidance, recovery guidance,
   and progress metrics.

5. Show a rejection:
   Run the supplement or homework case from `03_evaluate_multi_llm.py`. Explain
   that the agent rejects unsafe or irrelevant requests before LLM generation.

6. Show the two-LLM comparison:
   Use the `gold_llm_comparison_summary` table to compare pass rate, safety
   score, latency, estimated cost, coach time saved, and ROI.

7. Close with business value:
   The POC demonstrates a scalable advisor for academies and clubs. A coach can
   review and refine AI-generated first drafts instead of starting from scratch.

## Evaluation Cases

- Youth agility plan with shoulder pain.
- Youth conditioning plan with knee recovery.
- Youth strength plan with safe nutrition guidance.
- Irrelevant homework rejection.
- Unsafe supplement, fasting, and weight-cutting rejection.
- Adult athlete preview for a 19-year-old club basketball player.

## ROI Story

The evaluation notebook exposes these business variables as widgets:

- `pilot_athlete_count`
- `subscription_price_per_athlete`
- `plans_generated_per_month`
- `coach_minutes_saved_per_plan`
- `value_per_successful_plan`
- model input and output costs per 1,000 tokens

Recommended video framing:

If an academy has 250 athletes at $12 per athlete per month, monthly recurring
revenue is $3,000 and annual recurring revenue is $36,000 per academy. The model
comparison should show whether the more expensive LLM produces enough additional
successful plans, safer responses, or lower human-review burden to justify its
cost.

## Human Review

Human review is required for:

- Failed judge cases.
- Any injury-related plan.
- Unsafe or irrelevant rejection cases.
- Adult-preview responses, until the adult segment has its own validated policy.
- Any low safety score.

Human reviewers should confirm age appropriateness, injury caution, nutrition
safety, clarity, and whether the plan is practical for a parent, coach, or adult
athlete to review.

## Investor Readiness

The current POC uses public data, so the initial data moat is limited. The
investable version should collect proprietary operating data during academy
pilots:

- plan completion
- pain or soreness flags
- coach ratings
- athlete ratings
- goal progress
- model version and plan quality
- human edits to generated plans

That feedback loop can improve recommendations over time and create a defensible
dataset that generic fitness chatbots do not have.

## Deployment Recommendation

- Use Databricks Git folders for team development.
- Use Databricks Jobs to refresh the data pipeline.
- Use Databricks Model Serving for the agent's LLM endpoints.
- Use MLflow traces and evaluation tables for quality monitoring.
- Use Streamlit or a simple web app for the pilot user interface.
- Route injury flags, failed evaluations, and low-confidence responses to human
  review before showing them broadly.
