# Databricks notebook source
# MAGIC %md
# MAGIC # Elite Athletic AI Advisor: Data Pipeline
# MAGIC
# MAGIC This notebook builds the Bronze, Silver, and Gold tables used by the agent.
# MAGIC It is designed for Databricks Git folders, but the `source_dir` widget can
# MAGIC point to any DBFS or workspace-file location containing the project CSVs.

# COMMAND ----------

from pyspark.sql import Row
from pyspark.sql import functions as F


def _default_source_dir():
    """Prefer the repo root when this notebook is run from a Databricks Git folder."""
    try:
        context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        notebook_path = context.notebookPath().get()
        workspace_path = notebook_path if notebook_path.startswith("/Workspace/") else f"/Workspace{notebook_path}"
        notebook_dir = "/".join(workspace_path.split("/")[:-1])
        repo_root = "/".join(notebook_dir.split("/")[:-1]) if notebook_dir.endswith("/notebooks") else notebook_dir
        return f"file:{repo_root}"
    except Exception:
        return "dbfs:/FileStore/elite_athletic_ai_advisor"


def _widget(name, default):
    try:
        dbutils.widgets.text(name, default)
        return dbutils.widgets.get(name)
    except Exception:
        return default


CATALOG = _widget("catalog", "main")
SCHEMA = _widget("schema", "default")
SOURCE_DIR = _widget("source_dir", _default_source_dir()).rstrip("/")


def q(table_name):
    return f"`{CATALOG}`.`{SCHEMA}`.`{table_name}`"


def read_project_csv(file_name):
    path = f"{SOURCE_DIR}/{file_name}"
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(path)
    )


def write_table(df, table_name):
    (
        df.write
        .mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(q(table_name))
    )


spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
spark.sql(f"USE CATALOG `{CATALOG}`")
spark.sql(f"USE SCHEMA `{SCHEMA}`")

print(f"Catalog/schema: {CATALOG}.{SCHEMA}")
print(f"Source directory: {SOURCE_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Tables

# COMMAND ----------

BRONZE_FILES = {
    "bronze_athlete_profiles": "athlete_test_profiles.csv",
    "bronze_player_data": "player_data.csv",
    "bronze_players": "Players.csv",
    "bronze_nba_stats": "Seasons_Stats.csv",
    "bronze_gym_exercises": "megaGymDataset.csv",
    "bronze_food_sample": "sub_sample_food.csv",
    "bronze_food_nutrients": "sub_sample_result.csv",
    "bronze_food_categories": "food_category.csv",
    "bronze_food_portions": "food_portion.csv",
}

for table_name, file_name in BRONZE_FILES.items():
    df = read_project_csv(file_name)
    write_table(df, table_name)
    print(f"Created {q(table_name)} from {file_name}: {df.count()} rows, {len(df.columns)} columns")

# COMMAND ----------

def profile_table(table_name):
    df = spark.table(q(table_name))
    null_counts = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])

    print("=" * 80)
    print(f"TABLE: {q(table_name)}")
    print(f"Rows: {df.count()}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duplicates: {df.count() - df.dropDuplicates().count()}")
    df.printSchema()
    display(null_counts)
    display(df.limit(5))


for table_name in BRONZE_FILES:
    profile_table(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver Tables

# COMMAND ----------

athlete = spark.table(q("bronze_athlete_profiles"))

silver_athlete_profiles = (
    athlete
    .dropDuplicates(["athlete_id"])
    .select(
        F.trim(F.col("athlete_id")).alias("athlete_id"),
        F.col("age").cast("int").alias("age"),
        F.lower(F.trim(F.col("sport"))).alias("sport"),
        F.trim(F.col("position")).alias("position"),
        F.lower(F.trim(F.col("experience_level"))).alias("experience_level"),
        F.lower(F.trim(F.col("goal"))).alias("goal"),
        F.lower(F.trim(F.coalesce(F.col("injury_status"), F.lit("none")))).alias("injury_status"),
        F.col("available_days").cast("int").alias("available_days"),
        F.lower(F.trim(F.col("equipment"))).alias("equipment"),
        F.trim(F.col("notes")).alias("notes"),
    )
    .filter(F.col("age") >= 13)
    .withColumn(
        "age_group",
        F.when((F.col("age") >= 13) & (F.col("age") <= 17), F.lit("youth"))
        .when(F.col("age") >= 18, F.lit("adult"))
        .otherwise(F.lit("unsupported")),
    )
    .withColumn(
        "oversight_model",
        F.when(F.col("age_group") == "youth", F.lit("parent_or_coach_required"))
        .when(F.col("age_group") == "adult", F.lit("self_directed_or_coach_supported"))
        .otherwise(F.lit("not_supported")),
    )
    .withColumn(
        "safe_training_days",
        F.when(
            F.col("age_group") == "adult",
            F.least(F.greatest(F.coalesce(F.col("available_days"), F.lit(3)), F.lit(1)), F.lit(6)),
        ).otherwise(
            F.least(F.greatest(F.coalesce(F.col("available_days"), F.lit(3)), F.lit(1)), F.lit(5))
        ),
    )
)

adult_preview_profile = spark.createDataFrame([
    Row(
        athlete_id="A901",
        age=19,
        sport="basketball",
        position="Point Guard",
        experience_level="intermediate",
        goal="improve conditioning",
        injury_status="none",
        available_days=5,
        equipment="basketball, cones, gym access",
        notes="Adult preview profile for product expansion testing",
        age_group="adult",
        oversight_model="self_directed_or_coach_supported",
        safe_training_days=5,
    )
])

silver_athlete_profiles = silver_athlete_profiles.unionByName(adult_preview_profile)

write_table(silver_athlete_profiles, "silver_athlete_profiles")
display(silver_athlete_profiles)

# COMMAND ----------

nba = spark.table(q("bronze_nba_stats"))

silver_nba_stats = (
    nba
    .dropDuplicates()
    .select(
        F.col("Year").cast("int").alias("season_year"),
        F.trim(F.col("Player")).alias("player_name"),
        F.upper(F.trim(F.col("Pos"))).alias("position"),
        F.col("Age").cast("int").alias("age"),
        F.col("G").cast("int").alias("games"),
        F.col("MP").cast("double").alias("minutes_played"),
        F.col("PTS").cast("double").alias("points"),
        F.col("AST").cast("double").alias("assists"),
        F.col("TRB").cast("double").alias("rebounds"),
        F.col("STL").cast("double").alias("steals"),
        F.col("BLK").cast("double").alias("blocks"),
    )
    .filter(F.col("player_name").isNotNull())
)

write_table(silver_nba_stats, "silver_nba_stats")
display(silver_nba_stats.limit(10))

# COMMAND ----------

gym = spark.table(q("bronze_gym_exercises"))

silver_gym_exercises = (
    gym
    .dropDuplicates()
    .select(
        F.trim(F.col("Title")).alias("exercise_name"),
        F.trim(F.coalesce(F.col("Desc"), F.lit("No description available"))).alias("description"),
        F.lower(F.trim(F.col("Type"))).alias("exercise_type"),
        F.lower(F.trim(F.col("BodyPart"))).alias("body_part"),
        F.lower(F.trim(F.coalesce(F.col("Equipment"), F.lit("body only")))).alias("equipment"),
        F.lower(F.trim(F.col("Level"))).alias("difficulty_level"),
    )
    .filter(F.col("exercise_name").isNotNull())
)

write_table(silver_gym_exercises, "silver_gym_exercises")
display(silver_gym_exercises.limit(10))

# COMMAND ----------

silver_food_nutrients = spark.table(q("bronze_food_nutrients")).dropDuplicates()
silver_food_portions = spark.table(q("bronze_food_portions")).dropDuplicates()
silver_food_categories = spark.table(q("bronze_food_categories")).dropDuplicates()

write_table(silver_food_nutrients, "silver_food_nutrients")
write_table(silver_food_portions, "silver_food_portions")
write_table(silver_food_categories, "silver_food_categories")

display(silver_food_nutrients.limit(10))
display(silver_food_portions.limit(10))
display(silver_food_categories.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Agent-Ready Tables

# COMMAND ----------

exercise_text = F.lower(
    F.concat_ws(
        " ",
        F.col("exercise_name"),
        F.col("description"),
        F.col("exercise_type"),
        F.col("body_part"),
        F.col("equipment"),
        F.col("difficulty_level"),
    )
)

gold_exercise_recommendations = (
    silver_gym_exercises
    .withColumn(
        "goal_tag",
        F.when(exercise_text.rlike("plyometric|jump|vertical|calves|quadriceps|hamstrings|glutes"), "vertical_jump")
        .when(exercise_text.rlike("conditioning|cardio|endurance"), "conditioning")
        .when(exercise_text.rlike("agility|speed|legs|footwork"), "speed_agility")
        .when(exercise_text.rlike("shoulders|chest|back|arms|strength|barbell|dumbbell"), "strength")
        .when(exercise_text.rlike("abdominals|core|plank"), "core")
        .otherwise("general_fitness"),
    )
    .withColumn(
        "agent_text",
        F.concat_ws(
            " | ",
            F.col("exercise_name"),
            F.col("exercise_type"),
            F.col("body_part"),
            F.col("equipment"),
            F.col("difficulty_level"),
            F.col("description"),
        ),
    )
)

write_table(gold_exercise_recommendations, "gold_exercise_recommendations")
display(gold_exercise_recommendations.limit(10))

# COMMAND ----------

gold_basketball_benchmarks = (
    silver_nba_stats
    .filter(F.col("season_year") >= 2000)
    .filter(F.col("position").isNotNull())
    .groupBy("position")
    .agg(
        F.avg("points").alias("avg_points"),
        F.avg("assists").alias("avg_assists"),
        F.avg("rebounds").alias("avg_rebounds"),
        F.avg("steals").alias("avg_steals"),
        F.avg("blocks").alias("avg_blocks"),
        F.avg("minutes_played").alias("avg_minutes_played"),
        F.count("*").alias("sample_size"),
    )
)

write_table(gold_basketball_benchmarks, "gold_basketball_benchmarks")
display(gold_basketball_benchmarks)

# COMMAND ----------

nutrition_rows = [
    Row(goal_tag="general_fitness", timing="daily", guidance="Build meals around carbohydrates, lean protein, colorful produce, water, and age-appropriate portions.", avoid="No supplements, fasting, extreme dieting, or medical nutrition treatment."),
    Row(goal_tag="speed_agility", timing="pre-workout", guidance="Choose easy carbohydrates and hydration before training, such as fruit, toast, rice, oats, or a balanced snack.", avoid="Avoid energy drinks and stimulant supplements."),
    Row(goal_tag="vertical_jump", timing="post-workout", guidance="Pair carbohydrates with protein after jumping or strength sessions to support recovery, such as yogurt and fruit or a balanced meal.", avoid="Avoid high-dose protein supplements unless supervised by a clinician."),
    Row(goal_tag="strength", timing="daily", guidance="Use regular meals with protein-rich foods, whole grains, fruits, vegetables, and dairy or fortified alternatives.", avoid="Avoid bulking plans, weight-cutting, or supplement stacks for youth athletes."),
    Row(goal_tag="conditioning", timing="recovery", guidance="Emphasize hydration, regular meals, and sleep after conditioning days. Add rest-day snacks if appetite is high.", avoid="Avoid fasted training plans for minors."),
    Row(goal_tag="injury_recovery", timing="recovery", guidance="Recommend consistent meals, hydration, sleep, and guardian or clinician input while training volume is reduced.", avoid="Do not prescribe therapeutic diets or supplement protocols."),
]

gold_nutrition_guidance = spark.createDataFrame(nutrition_rows)
write_table(gold_nutrition_guidance, "gold_nutrition_guidance")
display(gold_nutrition_guidance)

# COMMAND ----------

safety_rows = [
    Row(rule_id="R001", rule_category="age", severity="required", rule_text="Youth athletes ages 13-17 require parent or coach oversight."),
    Row(rule_id="R002", rule_category="injury", severity="required", rule_text="If injury status is not none, avoid high-impact training and recommend coach or medical review."),
    Row(rule_id="R003", rule_category="workload", severity="required", rule_text="Youth athletes should not receive intense training plans for more than 5 days per week."),
    Row(rule_id="R004", rule_category="nutrition", severity="required", rule_text="Do not recommend supplements, extreme diets, fasting, weight cutting, or medical nutrition treatment."),
    Row(rule_id="R005", rule_category="recovery", severity="required", rule_text="Every weekly plan must include recovery, stretching, hydration, and sleep guidance."),
    Row(rule_id="R006", rule_category="scope", severity="required", rule_text="The agent provides educational basketball development guidance only, not medical advice."),
    Row(rule_id="R007", rule_category="scope", severity="required", rule_text="Reject requests unrelated to youth basketball training, recovery, or general nutrition."),
    Row(rule_id="R008", rule_category="age", severity="preview", rule_text="Adult athletes ages 18+ can receive a limited adult basketball development preview with self-directed or coach-supported language."),
]

gold_safety_rules = spark.createDataFrame(safety_rows)
write_table(gold_safety_rules, "gold_safety_rules")
display(gold_safety_rules)

# COMMAND ----------

progress_metric_rows = [
    Row(metric_id="P001", metric_name="sessions_completed", metric_type="integer", owner="athlete", description="Number of planned weekly sessions completed."),
    Row(metric_id="P002", metric_name="pain_or_soreness_flag", metric_type="boolean", owner="athlete", description="Flag for pain escalation or unusual soreness after a session."),
    Row(metric_id="P003", metric_name="sleep_quality", metric_type="1_to_5_rating", owner="athlete", description="Self-reported sleep quality for recovery tracking."),
    Row(metric_id="P004", metric_name="hydration_check", metric_type="boolean", owner="athlete", description="Whether hydration guidance was followed on training days."),
    Row(metric_id="P005", metric_name="skill_confidence", metric_type="1_to_5_rating", owner="athlete", description="Self-reported confidence in the primary skill focus."),
    Row(metric_id="P006", metric_name="coach_parent_reviewed", metric_type="boolean", owner="coach_or_guardian", description="Whether a coach, parent, or adult reviewer checked the plan and progress."),
]

gold_progress_metric_definitions = spark.createDataFrame(progress_metric_rows)
write_table(gold_progress_metric_definitions, "gold_progress_metric_definitions")
display(gold_progress_metric_definitions)

# COMMAND ----------

feedback_schema_rows = [
    Row(field_name="plan_id", data_type="string", purpose="Unique identifier for a generated weekly plan."),
    Row(field_name="athlete_id", data_type="string", purpose="Links feedback to an athlete profile."),
    Row(field_name="generated_at", data_type="timestamp", purpose="Tracks when the plan was created."),
    Row(field_name="model_endpoint", data_type="string", purpose="Supports model performance and ROI comparison over time."),
    Row(field_name="goal_tag", data_type="string", purpose="Groups plans by athlete development goal."),
    Row(field_name="sessions_completed", data_type="integer", purpose="Measures adherence to the weekly plan."),
    Row(field_name="pain_or_soreness_flag", data_type="boolean", purpose="Triggers human review for injury or recovery concerns."),
    Row(field_name="coach_rating", data_type="integer_1_to_5", purpose="Captures human quality feedback from coach or parent review."),
    Row(field_name="athlete_rating", data_type="integer_1_to_5", purpose="Captures whether the athlete found the plan useful and motivating."),
    Row(field_name="free_text_feedback", data_type="string", purpose="Collects qualitative feedback for future recommendation improvement."),
]

gold_feedback_event_schema = spark.createDataFrame(feedback_schema_rows)
write_table(gold_feedback_event_schema, "gold_feedback_event_schema")
display(gold_feedback_event_schema)

# COMMAND ----------

write_table(silver_athlete_profiles, "gold_athlete_profiles")
write_table(spark.table(q("silver_food_nutrients")).dropDuplicates(), "gold_nutrition_lookup")
write_table(spark.table(q("silver_food_portions")).dropDuplicates(), "gold_food_portions")

GOLD_TABLES = [
    "gold_athlete_profiles",
    "gold_exercise_recommendations",
    "gold_basketball_benchmarks",
    "gold_nutrition_guidance",
    "gold_nutrition_lookup",
    "gold_food_portions",
    "gold_safety_rules",
    "gold_progress_metric_definitions",
    "gold_feedback_event_schema",
]

summary_rows = []
for table_name in GOLD_TABLES:
    df = spark.table(q(table_name))
    summary_rows.append(Row(table_name=table_name, row_count=df.count(), column_count=len(df.columns)))

gold_summary = spark.createDataFrame(summary_rows)
write_table(gold_summary, "gold_table_summary")
display(gold_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Notes
# MAGIC
# MAGIC The nutrition source files in this POC are useful for showing USDA-derived
# MAGIC nutrient and portion data, but they do not include a rich food-description
# MAGIC table. The `gold_nutrition_guidance` table adds a small, safety-reviewed
# MAGIC guidance layer so the agent can produce grounded youth-appropriate nutrition
# MAGIC advice without inventing supplement or diet recommendations.