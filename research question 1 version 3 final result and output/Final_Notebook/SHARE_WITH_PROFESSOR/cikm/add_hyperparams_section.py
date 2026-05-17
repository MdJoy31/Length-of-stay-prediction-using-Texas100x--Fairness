"""
Insert a new section right after Section 5.2 (model training) that
documents every hyperparameter for every model in a clean table.
The section produces:
  - A markdown header
  - A code cell that builds T_HYPERPARAMS.csv and displays it inline.
"""
import json
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


# Find cell 13 (model training) and insert new cells right after it
training_idx = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "5.2 · Train 12 models" in src and "Stacking Ensemble" in src:
        training_idx = i
        break

if training_idx is None:
    print("ERROR: training cell not found")
    raise SystemExit(1)
print(f"Training cell at index {training_idx}")

# Don't re-insert if section already present
already = False
for c in nb["cells"]:
    if c["cell_type"] == "markdown" and "5.3 · Hyperparameters reference table" in "".join(c.get("source", [])):
        already = True
        break
if already:
    print("Hyperparameters section already exists — replacing in place")


MD_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 5.3 · Hyperparameters reference table (all 12 models)\n",
        "\n",
        "Complete set of hyperparameters used to fit every classifier in this study.\n",
        "All models are trained on the same scaled training matrix `X_train_sc` (740,102 rows, 14 features) with `random_state=42` for reproducibility. The table is saved to `output_final/tables/T_HYPERPARAMS.csv`.\n",
    ],
}

CODE_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ──────────────────────────────────────────────────────────────\n",
        "# 5.3 · Hyperparameters reference table\n",
        "# Programmatically extract every parameter from every fitted\n",
        "# classifier so the table can never drift from the actual code.\n",
        "# ──────────────────────────────────────────────────────────────\n",
        "def _params_one_line(mdl, drop_keys=(\"random_state\", \"seed\", \"n_jobs\",\n",
        "                                      \"verbose\", \"verbosity\", \"thread_count\",\n",
        "                                      \"silent\", \"warm_start\", \"missing\",\n",
        "                                      \"validate_parameters\", \"early_stopping\",\n",
        "                                      \"tree_method\", \"objective\", \"booster\",\n",
        "                                      \"interaction_constraints\", \"monotone_constraints\",\n",
        "                                      \"sampling_method\", \"feature_types\",\n",
        "                                      \"enable_categorical\", \"importance_type\",\n",
        "                                      \"device\", \"max_bin\", \"grow_policy\",\n",
        "                                      \"max_leaves\", \"max_cat_threshold\",\n",
        "                                      \"max_cat_to_onehot\", \"multi_strategy\",\n",
        "                                      \"num_parallel_tree\", \"sample_type\",\n",
        "                                      \"normalize_type\", \"rate_drop\", \"one_drop\",\n",
        "                                      \"skip_drop\", \"updater\", \"refresh_leaf\",\n",
        "                                      \"process_type\", \"max_delta_step\",\n",
        "                                      \"scale_pos_weight\", \"base_score\", \"seed_per_iteration\",\n",
        "                                      \"colsample_bylevel\", \"colsample_bynode\",\n",
        "                                      \"gamma\", \"feature_weights\", \"callbacks\",\n",
        "                                      \"eval_metric\", \"disable_default_eval_metric\",\n",
        "                                      \"verbose_eval\", \"eval_set\")):\n",
        "    \"\"\"Return a clean comma-separated list of non-default user-set params.\"\"\"\n",
        "    p = mdl.get_params(deep=False)\n",
        "    items = []\n",
        "    for k, v in p.items():\n",
        "        if k in drop_keys: continue\n",
        "        if v is None: continue\n",
        "        if isinstance(v, (list, tuple)) and len(v) == 0: continue\n",
        "        # Skip estimator references (Stacking) — handled separately below\n",
        "        if hasattr(v, 'get_params'): continue\n",
        "        if isinstance(v, list) and any(hasattr(item, 'get_params') for item in v): continue\n",
        "        if isinstance(v, str): items.append(f\"{k}='{v}'\")\n",
        "        else: items.append(f\"{k}={v}\")\n",
        "    return \", \".join(items)\n",
        "\n",
        "hp_rows = []\n",
        "model_family = {\n",
        "    \"Logistic Regression\":    \"Linear (regularised)\",\n",
        "    \"Decision Tree\":          \"Single tree\",\n",
        "    \"Random Forest\":          \"Bagging of trees\",\n",
        "    \"Gradient Boosting\":      \"Boosting (sklearn)\",\n",
        "    \"AdaBoost\":               \"Boosting (adaptive)\",\n",
        "    \"XGBoost\":                \"GBM (XGBoost) - CANONICAL\",\n",
        "    \"LightGBM\":               \"GBM (LightGBM)\",\n",
        "    \"CatBoost\":               \"GBM (CatBoost)\",\n",
        "    \"HistGradientBoosting\":   \"Histogram GBM (sklearn)\",\n",
        "    \"Bagging\":                \"Bagging of decision trees\",\n",
        "    \"Extra Trees\":            \"Randomised trees\",\n",
        "    \"Stacking Ensemble\":      \"Meta: LR + RF + XGB -> LR\",\n",
        "}\n",
        "for name, mdl in trained_models.items():\n",
        "    hp_rows.append({\n",
        "        \"Model\": name,\n",
        "        \"Family\": model_family.get(name, \"\"),\n",
        "        \"Hyperparameters\": _params_one_line(mdl),\n",
        "    })\n",
        "\n",
        "# For Stacking, also expand the inner estimators because the row above\n",
        "# only shows top-level Stacking params (cv, passthrough, etc.).\n",
        "if \"Stacking Ensemble\" in trained_models:\n",
        "    stack = trained_models[\"Stacking Ensemble\"]\n",
        "    for sub_name, sub_est in stack.estimators:\n",
        "        hp_rows.append({\n",
        "            \"Model\": f\"  └─ Stacking.{sub_name}\",\n",
        "            \"Family\": f\"base estimator ({type(sub_est).__name__})\",\n",
        "            \"Hyperparameters\": _params_one_line(sub_est),\n",
        "        })\n",
        "    hp_rows.append({\n",
        "        \"Model\": \"  └─ Stacking.final_estimator\",\n",
        "        \"Family\": f\"meta-learner ({type(stack.final_estimator).__name__})\",\n",
        "        \"Hyperparameters\": _params_one_line(stack.final_estimator),\n",
        "    })\n",
        "\n",
        "T_HP = pd.DataFrame(hp_rows)\n",
        "T_HP.to_csv(f\"{TABLES_DIR}/T_HYPERPARAMS.csv\", index=False)\n",
        "print(f\"Wrote {TABLES_DIR}/T_HYPERPARAMS.csv  ({T_HP.shape[0]} rows)\")\n",
        "print(\"\\nFixed seeds: random_state=42 (or seed=42 where applicable). Other reproducibility-only\")\n",
        "print(\"flags (n_jobs, verbosity, thread_count, etc.) are omitted from the table.\\n\")\n",
        "with pd.option_context('display.max_colwidth', 220, 'display.width', 220):\n",
        "    display(T_HP)\n",
    ],
}


# If section already exists, replace; otherwise insert
if already:
    # Find and replace existing markdown + code pair
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown" and "5.3 · Hyperparameters reference table" in "".join(c.get("source", [])):
            nb["cells"][i] = MD_CELL
            # Code cell follows
            if i + 1 < len(nb["cells"]) and nb["cells"][i + 1]["cell_type"] == "code":
                nb["cells"][i + 1] = CODE_CELL
            break
    print("Replaced existing hyperparameters section")
else:
    # Insert after training cell
    nb["cells"].insert(training_idx + 1, MD_CELL)
    nb["cells"].insert(training_idx + 2, CODE_CELL)
    print(f"Inserted hyperparameters section at indices {training_idx+1}, {training_idx+2}")


with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote notebook ({len(nb['cells'])} cells total)")
