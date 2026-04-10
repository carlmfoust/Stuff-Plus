# functions/models.R
load_model_xgb <- function(path) {
  xgb.load(file.path("~/Code/Stuff-Plus/models", path))
}

# Load all models and assign to environment
models <- list(
  # Pitch Quality
  .rhp_ff_loc_whiff_model = load_model_xgb("rhp_ff_loc_whiff_XGBoost_2026-04-09_12-58.json"),
  .rhp_ff_loc_foul_model = load_model_xgb("rhp_ff_loc_foul_XGBoost_2026-04-09_13-35.json"),
  .rhp_ff_loc_bip_model = load_model_xgb("rhp_ff_loc_bip_XGBoost_2026-04-09_13-54.json"),
  .rhp_ff_loc_cs_model = load_model_xgb("rhp_ff_loc_cs_XGBoost_2026-04-09_14-50.json"),
  .rhp_ff_loc_hbp_model = load_model_xgb("rhp_ff_loc_hbp_XGBoost_2026-04-09_15-33.json"),
  .rhp_ff_loc_swing_model = load_model_xgb("rhp_ff_loc_swing_XGBoost_2026-04-09_16-17.json"),
  
  .lhp_ff_loc_whiff_model = load_model_xgb("lhp_ff_loc_whiff_XGBoost_2026-04-09_14-28.json"),
  .lhp_ff_loc_foul_model = load_model_xgb("lhp_ff_loc_foul_XGBoost_2026-04-09_14-35.json"),
  .lhp_ff_loc_bip_model = load_model_xgb("lhp_ff_loc_bip_XGBoost_2026-04-09_14-40.json"),
  .lhp_ff_loc_cs_model = load_model_xgb("lhp_ff_loc_cs_XGBoost_2026-04-09_16-01.json"),
  .lhp_ff_loc_hbp_model = load_model_xgb("lhp_ff_loc_hbp_XGBoost_2026-04-09_16-09.json"),
  .lhp_ff_loc_swing_model = load_model_xgb("lhp_ff_loc_swing_XGBoost_2026-04-09_17-41.json")
)
