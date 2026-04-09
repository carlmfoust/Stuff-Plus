library(tidyverse)
library(tidymodels)
library(future)
library(probably)
library(vip)
library(xgboost)
library(bonsai)
library(lightgbm)

swing_list <- c(
  'hit_into_play',
  'foul',
  'foul_tip',
  'swinging_strike',
  'swinging_strike_blocked',
  'foul_bunt',
  'missed_bunt',
  'bunt_foul_tip'
)

whiff_list <- c('swinging_strike', 'swinging_strike_blocked', 'foul_tip', 'bunt_foul_tip')

foul_list <- c('foul', 'foul_bunt')

strike_list <- c('called_strike')

contact_list <- c('hit_into_play', 'foul', 'foul_tip')

vaa_features <- c("release_speed", "release_pos_z", "a_vert", "plate_z")
haa_features <- c("release_speed", "release_pos_x", "a_horz", "plate_x")
arm_angle_features <- c("release_pos_x", "release_pos_z", "a_horz", "a_vert", "spin_axis")
exp_ff_features <- c("release_pos_x", "release_pos_z", "arm_angle", "release_extension")
exp_ff_x_features <- c("release_pos_x", "release_pos_z", "arm_angle", "release_extension", "is_lhp")

load_model_xgb <- function(path) {
  xgb.load(file.path("~/Code/Stuff-Plus/models/", path))
}

generate_model_preds <- function(model, data, features, output_name = "pred") {
  
  # 1. Select only the features needed for this specific model
  # 'all_of' ensures it throws an error if a column is missing
  prediction_data <- data %>% 
    select(all_of(features))
  
  # 2. Generate predictions
  preds <- predict(model, prediction_data) %>% 
    as.data.frame()
  
  # 3. Rename the first column to your custom name
  # We use the := operator (walrus operator) for dynamic naming in dplyr
  preds <- preds %>% 
    rename(!!output_name := 1)
  
  return(preds)
}

create_advance_metrics <- function(df) {
  df_prep <- df |>
    mutate(
      pitch_name = ifelse(pitch_name == "Split-Finger", "Splitter", pitch_name),
      pitch_group = case_when(
        pitch_name %in% c("4-Seam Fastball", "Sinker", "Cutter") ~ "Fastball",
        pitch_name %in%
          c(
            "Slider",
            "Sweeper",
            "Curveball",
            "Slurve",
            "Knuckle Curve",
            "Slow Curve",
            "Screwball"
          ) ~ "BreakingBall",
        pitch_name %in% c("Changeup", "Splitter", "Forkball") ~ "Offspeed",
        pitch_name %in%
          c("Eephus", "Knuckleball", "Other", "Pitch Out") ~ "Other",
        TRUE ~ NA
      )
    ) |>
    mutate(
      pitch_uid = paste(
        game_date,
        pitcher,
        at_bat_number,
        pitch_number,
        sep = "_"
      )
    ) |>
    select(
      pitcher,
      batter,
      game_type,
      pitch_name,
      pitch_group,
      at_bat_number,
      pitch_uid,
      pitch_number,
      description,
      events,
      plate_x,
      plate_z,
      launch_speed_angle,
      release_speed,
      release_spin_rate,
      release_extension,
      spin_axis,
      release_pos_x,
      release_pos_z,
      pfx_x,
      pfx_z,
      p_throws,
      stand,
      game_year,
      game_date,
      delta_run_exp,
      vx0,
      vy0,
      vz0,
      ax,
      ay,
      az,
      arm_angle,
      sz_top,
      sz_bot,
      balls,
      strikes,
      hc_x,
      hc_y,
      launch_speed,
      launch_angle,
      bat_speed,
      swing_length,
      hit_distance_sc,
      estimated_woba_using_speedangle,
      estimated_ba_using_speedangle,
      swing_length,
      type
    )
  
  df_prep <- df_prep |>
    mutate(
      vy_f = -sqrt((vy0^2 - (2 * ay * (50 - (17 / 12))))),
      t = (vy_f - vy0) / ay,
      vz_f = vz0 + (az * t),
      vx_f = vx0 + (ax * t),
      vaa = -atan((vz_f / vy_f)) * (180 / pi),
      haa = -atan((vx_f / vy_f)) * (180 / pi),
      v_mag = sqrt(vx0^2 + vy0^2 + vz0^2),
      tang_x = vx0 / v_mag,
      tang_y = vy0 / v_mag,
      tang_z = vz0 / v_mag,
      handedness_factor = if_else(p_throws == "R", 1, -1),
      temp_x = 0,
      temp_y = 0,
      temp_z = 1,
      horz_x_raw = temp_y * tang_z - temp_z * tang_y,
      horz_y_raw = temp_z * tang_x - temp_x * tang_z,
      horz_z_raw = temp_x * tang_y - temp_y * tang_x,
      horz_mag = sqrt(horz_x_raw^2 + horz_y_raw^2 + horz_z_raw^2),
      horz_x = horz_x_raw / horz_mag,
      horz_y = horz_y_raw / horz_mag,
      horz_z = horz_z_raw / horz_mag,
      vert_x = tang_y * horz_z - tang_z * horz_y,
      vert_y = tang_z * horz_x - tang_x * horz_z,
      vert_z = tang_x * horz_y - tang_y * horz_x,
      a_tang = ax * tang_x + ay * tang_y + az * tang_z,
      a_horz = ax * horz_x + ay * horz_y + az * horz_z,
      a_vert = ax * vert_x + ay * vert_y + az * vert_z
    ) |>
    select(-c(vx0:az, vy_f:vx_f, v_mag:a_tang))
  
  df_prep <- df_prep |>
    mutate(
      pfx_x = pfx_x * 12,
      pfx_z = pfx_z * 12,
      stand = as.numeric(factor(stand)) - 1,
      plate_x_centered = 0,
      plate_z_centered = 2.5,
      sz_top = sz_top * 12,
      sz_bot = sz_bot * 12,
      sz_height = (sz_top - sz_bot),
      sz_mid = (sz_height / 2) + sz_bot,
      sz_mid_z_diff = (plate_z * 12) - sz_mid,
      dist_from_mid = ((sz_mid_z_diff)^2 + (plate_x * 12)^2)^(1 / 2),
      p_throws = ifelse(p_throws == 'R', 1, 0),
      # plate_x_og = plate_x,
      # plate_z_og = plate_z,
      # SHH = ifelse(p_throws == stand, 1, 0),
      is_shh = ifelse(p_throws == stand, 1, 0),
      is_lhp = ifelse(p_throws == 0, 1, 0),
      is_lhh = ifelse(stand == 0, 1, 0),
      is_3_0 = ifelse(balls == 3 & strikes == 0, 1, 0),
      is_called_strike = ifelse(description %in% strike_list, 1, 0),
      is_batter_swing = ifelse(description %in% swing_list, 1, 0)
    )
  
  fb_usage <- df_prep %>% 
    group_by(pitcher, game_year)%>%
    summarise(
      n=n(),
      fastball_pct = round(sum(pitch_name %in% c("4-Seam Fastball", "Sinker"), na.rm = TRUE)/n()*100,1),
      fastball_velo = round(mean(release_speed[pitch_name %in% c("4-Seam Fastball", "Sinker")], na.rm = TRUE),1),
      cutter_pct   = round(sum(pitch_name == "Cutter", na.rm = TRUE)/n()*100,1),
      cutter_velo = round(mean(release_speed[pitch_name == "Cutter"], na.rm = TRUE),1)
    ) %>% 
    mutate(ct_prim = if_else((fastball_pct) < 20 | cutter_velo > fastball_velo*0.98, 1, 0)) %>% 
    select(pitcher, game_year, ct_prim)
  
  df_typed <- df_prep |>
    left_join(fb_usage, by = c("pitcher", "game_year")) |>
    mutate(pitch_group_new = ifelse(ct_prim == 1, "Fastball", "BreakingBall"),
           pitch_group = ifelse(pitch_name == "Cutter", pitch_group_new, pitch_group))
  
  pitcher_prim_fb <- df_typed %>% 
    filter(pitch_group == "Fastball") |> 
    group_by(pitcher, game_year)%>%
    summarise(
      n=n(),
      fastball_pct = round(sum(pitch_name == "4-Seam Fastball", na.rm = TRUE)/n()*100,1),
      sinker_pct = round(sum(pitch_name == "Sinker", na.rm = TRUE)/n()*100,1),
      cutter_pct   = round(sum(pitch_name == "Cutter", na.rm = TRUE)/n()*100,1)
    ) %>% 
    rowwise() %>%
    mutate(primary_fb= c("4-Seam Fastball", "Sinker", "Cutter")[which.max(c(fastball_pct, sinker_pct, cutter_pct))]) %>% 
    ungroup() %>% 
    select(pitcher, game_year, primary_fb)
  
  df_fb_avg <- df_typed %>% 
    left_join(pitcher_prim_fb) %>% 
    mutate(is_primary = ifelse(primary_fb == pitch_name, 1, 0)) %>% 
    filter(is_primary == 1) |> 
    group_by(pitcher, game_year) |> 
    summarise(
      n = n(),
      fb_velo = mean(release_speed, na.rm = TRUE),
      fb_ivb = mean(pfx_z, na.rm = TRUE),
      fb_hb = mean(pfx_x, na.rm = TRUE),
      fb_a_vert = mean(a_vert, na.rm = TRUE),
      fb_a_horz = mean(a_horz, na.rm = TRUE),
      .groups = "drop"
    ) 
  
  
  df_test <- df_typed |>
    left_join(df_fb_avg) %>% 
    left_join(pitcher_prim_fb) %>% 
    mutate(
      rel_speed_fb_diff = release_speed - fb_velo,
      a_horz_fb_diff = a_horz - fb_a_horz,
      a_vert_fb_diff = a_vert - fb_a_vert,
      hb_fb_diff = pfx_x - fb_hb,
      ivb_fb_diff = pfx_z - fb_ivb
    )
  
  return(df_test)
}

train_classification_model <- function(
    df,
    model_name,
    tune_initial = 20,
    tune_iter = 100,
    tune_no_improve = 5,
    set_seed = 1015
) {
  set.seed(set_seed)
  
  plan(multisession, workers = 4)
  gc()
  
  if (!dir.exists("/Users/carlfoust/Code/Stuff-Plus/models")) {
    message("Directory 'models/' not found. Exiting early.")
    return(NULL)
  }
  
  full_file_name <- paste0(
    "/Users/carlfoust/Code/Stuff-Plus/",
    "models/",
    model_name,
    "_XGBoost_",
    Sys.Date(),
    "_",
    format(Sys.time(), "%H-%M")
  )
  
  data_split <- initial_validation_split(df, prop = c(0.8, 0.1), strata = label)
  train <- training(data_split)
  test <- testing(data_split)
  val_set <- validation_set(data_split)
  val <- validation(data_split)
  
  spec <- boost_tree(
    trees = tune(),
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    sample_size = tune(),
    loss_reduction = tune()
  ) |>
    set_mode("classification") |>
    set_engine("xgboost", 
               nthread = 1,
               monotone_constraints = c(release_speed = 1))
  
  rec <- recipe(label ~ ., data = train)
  
  wf <- workflow() |>
    add_recipe(rec) |>
    add_model(spec)
  
  print(paste0("Starting Tuning: ", Sys.time()))
  
  tune_res <- wf |>
    tune_bayes(
      resamples = val_set,
      initial = tune_initial,
      iter = tune_iter,
      control = control_bayes(
        verbose = TRUE,
        no_improve = tune_no_improve,
        seed = set_seed,
        parallel_over = "everything"
      ),
      metrics = metric_set(roc_auc)
    )
  plan(sequential)
  
  print(paste0("Ended Tuning: ", Sys.time()))
  
  model_metrics <- collect_metrics(tune_res)
  best_params <- select_best(tune_res, metric = "roc_auc")
  
  write.csv(
    model_metrics,
    paste0(full_file_name, "_Model_Metrics.csv"),
    row.names = FALSE
  )
  
  print(paste0("Starting Fitting: ", Sys.time()))
  
  final_model <- wf |>
    finalize_workflow(best_params) |>
    fit(data = train)
  
  print(paste0("Ended Fitting: ", Sys.time()))
  
  saveRDS(final_model, paste0(full_file_name, ".rds"))
  xgb.save(
    parsnip::extract_fit_engine(final_model),
    paste0(full_file_name, ".json")
  )
  
  preds <- predict(final_model, test, type = "prob") |>
    bind_cols(test)
  
  auc_res <- roc_auc(preds, label, .pred_0)
  log_loss <- mn_log_loss(preds, label, .pred_0)
  
  metrics_df <- bind_rows(auc_res, log_loss)
  
  write.csv(metrics_df, paste0(full_file_name, "_Error.csv"), row.names = FALSE)
  
  auc_res_plot <- autoplot(roc_curve(preds, label, .pred_0))
  
  ggsave(
    filename = paste0(full_file_name, "_AUC.png"),
    plot = auc_res_plot,
    width = 10,
    height = 8,
    dpi = 300
  )
  
  cal_plot <- cal_plot_breaks(preds, truth = label, estimate = .pred_0) +
    labs(title = paste0(model_name, " Calibration Plot"))
  
  ggsave(
    filename = paste0(full_file_name, "_Calibration.png"),
    plot = cal_plot,
    width = 10,
    height = 8,
    dpi = 300
  )
  
  vip_plot = final_model |>
    extract_fit_parsnip() |>
    vip(num_features = 40) +
    ggtitle(paste0("Feature Importance"))
  
  ggsave(
    filename = paste0(full_file_name, "_VIP.png"),
    plot = vip_plot,
    width = 10,
    height = 8,
    dpi = 300
  )
  
  list(
    train = train,
    test = test,
    val_set = val_set,
    val = val,
    model_metrics = model_metrics,
    final_model = final_model
  )
}

train_classification_lightGBM <- function(
    df,
    model_name,
    tune_initial = 20,
    tune_iter = 100,
    tune_no_improve = 5,
    set_seed = 1015
) {
  set.seed(set_seed)
  
  plan(sequential)
  plan(multisession, workers = 2)
  gc()
  
  if (!dir.exists("/Users/carlfoust/Code/Stuff-Plus/models")) {
    message("Directory 'models/' not found. Exiting early.")
    return(NULL)
  }
  
  full_file_name <- paste0(
    "/Users/carlfoust/Code/Stuff-Plus/",
    "models/",
    model_name,
    "_lightGBM_",
    Sys.Date(),
    "_",
    format(Sys.time(), "%H-%M")
  )
  
  data_split <- initial_validation_split(df, prop = c(0.8, 0.1), strata = label)
  train <- training(data_split)
  test <- testing(data_split)
  val_set <- validation_set(data_split)
  
  # --- UPDATED SECTION: CONSTRAINT HANDLING ---
  # Get the names of the predictors (everything except 'label')
  predictor_names <- train |> select(-label) |> colnames()
  
  # Create a vector of 0s (no constraint) named by the predictors
  mono_vec <- rep(0, length(predictor_names))
  names(mono_vec) <- predictor_names
  
  # Set the specific constraint for release_speed
  if("release_speed" %in% names(mono_vec)) {
    mono_vec["release_speed"] <- 1
  }
  
  spec <- boost_tree(
    trees = tune(),
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    sample_size = tune(),
    loss_reduction = tune(),
    mtry = tune() # Keep mtry as we discussed to help with xVAA dominance
  ) |>
    set_mode("classification") |>
    set_engine("lightgbm", 
               nthread = 1,
               # LightGBM prefers the numeric vector 
               monotone_constraints = mono_vec)
  
  # --- END UPDATED SECTION ---
  
  rec <- recipe(label ~ ., data = train)
  
  wf <- workflow() |>
    add_recipe(rec) |>
    add_model(spec)
  
  # Finalize mtry based on data size
  model_params <- extract_parameter_set_dials(spec) |> 
    update(mtry = mtry(c(1, length(predictor_names))))
  
  print(paste0("Starting Tuning: ", Sys.time()))
  
  tune_res <- wf |>
    tune_bayes(
      resamples = val_set,
      initial = tune_initial,
      iter = tune_iter,
      param_info = model_params,
      control = control_bayes(
        verbose = TRUE,
        no_improve = tune_no_improve,
        seed = set_seed,
        parallel_over = "everything"
      ),
      metrics = metric_set(roc_auc)
    )
  
  print(paste0("Ended Tuning: ", Sys.time()))
  
  # ... [Rest of the fitting and plotting code remains the same] ...
  # (Including best_params, finalize_workflow, and ggsave calls)
  
  model_metrics <- collect_metrics(tune_res)
  best_params <- select_best(tune_res, metric = "roc_auc")
  
  write.csv(model_metrics, paste0(full_file_name, "_Model_Metrics.csv"), row.names = FALSE)
  
  final_model <- wf |>
    finalize_workflow(best_params) |>
    fit(data = train)
  
  saveRDS(final_model, paste0(full_file_name, ".rds"))
  
  preds <- predict(final_model, test, type = "prob") |>
    bind_cols(test)
  
  auc_res <- roc_auc(preds, label, .pred_0)
  log_loss <- mn_log_loss(preds, label, .pred_0)
  metrics_df <- bind_rows(auc_res, log_loss)
  write.csv(metrics_df, paste0(full_file_name, "_Error.csv"), row.names = FALSE)
  
  # VIP Plot
  vip_plot = final_model |>
    extract_fit_parsnip() |>
    vip(num_features = 40) +
    ggtitle(paste0("Feature Importance (LightGBM)"))
  
  ggsave(filename = paste0(full_file_name, "_VIP.png"), plot = vip_plot, width = 10, height = 8)
  
  list(
    train = train,
    test = test,
    final_model = final_model
  )
}

train_regression_model <- function(
  df,
  model_name,
  tune_initial = 20,
  tune_iter = 100,
  tune_no_improve = 20,
  set_seed = 1015
) {
  set.seed(set_seed)

  dir.exists(file.path("models/"))
  full_file_name <- paste0(
    "models/",
    model_name,
    "_XGBoost_",
    Sys.Date(),
    "_",
    format(Sys.time(), "%H-%M")
  )

  data_split <- initial_validation_split(df, prop = c(0.8, 0.1), strata = label)
  train <- training(data_split)
  test <- testing(data_split)
  val_set <- validation_set(data_split)
  val <- validation(data_split)

  spec <- boost_tree(
    trees = tune(),
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    sample_size = tune(),
    loss_reduction = tune()
  ) |>
    set_mode("regression") |>
    set_engine("xgboost", nthread = 8)

  rec <- recipe(label ~ ., data = train)

  wf <- workflow() |>
    add_recipe(rec) |>
    add_model(spec)

  plan(multisession, workers = 2)

  print(paste0("Starting Tuning: ", Sys.time()))

  tune_res <- wf |>
    tune_bayes(
      resamples = val_set,
      initial = tune_initial,
      iter = tune_iter,
      control = control_bayes(
        verbose = TRUE,
        no_improve = tune_no_improve,
        seed = set_seed,
        parallel_over = "everything"
      ),
      metrics = metric_set(rmse)
    )

  print(paste0("Ended Tuning: ", Sys.time()))

  model_metrics <- collect_metrics(tune_res)
  best_params <- select_best(tune_res, metric = "rmse")

  print(paste0("Starting Fitting: ", Sys.time()))

  final_model <- wf |>
    finalize_workflow(best_params) |>
    fit(data = train)

  print(paste0("Ended Fitting: ", Sys.time()))

  saveRDS(final_model, paste0(full_file_name, ".rds"))
  # xgb.save(final_model$fit, paste0(full_file_name, ".json"))
  xgb.save(parsnip::extract_fit_engine(final_model), paste0(full_file_name, ".json"))

  preds <- predict(final_model, test, type = "numeric") |>
    bind_cols(test)

  preds_rmse <- rmse(preds, label, .pred)
  preds_rsq <- rsq(preds, label, .pred)
  preds_rsq_trad <- rsq_trad(preds, label, .pred)
  preds_mae <- mae(preds, label, .pred)

  metrics_df <- bind_rows(
    preds_rmse,
    preds_rsq,
    preds_rsq_trad,
    preds_mae
  )

  write.csv(metrics_df, paste0(full_file_name, "_Error.csv"), row.names = FALSE)

  cal_plot <- cal_plot_regression(preds, truth = label, estimate = .pred) +
    labs(title = paste0(model_name, " Calibration Plot"))

  vip_plot = final_model |>
    extract_fit_parsnip() |>
    vip(num_features = 40) +
    ggtitle(paste0("Feature Importance"))

  write.csv(
    model_metrics,
    paste0(full_file_name, "_Model_Metrics.csv"),
    row.names = FALSE
  )

  ggsave(
    filename = paste0(full_file_name, "_Calibration.png"),
    plot = cal_plot,
    width = 10,
    height = 8,
    dpi = 300
  )

  ggsave(
    filename = paste0(full_file_name, "_VIP.png"),
    plot = vip_plot,
    width = 10,
    height = 8,
    dpi = 300
  )

  list(
    train = train,
    test = test,
    val_set = val_set,
    val = val,
    model_metrics = model_metrics,
    final_model = final_model
  )
}

train_regression_lightGBM <- function(
  df,
  model_name,
  tune_initial = 20,
  tune_iter = 100,
  tune_no_improve = 20,
  set_seed = 1015
) {
  set.seed(set_seed)

  dir.exists(file.path("models/"))
  full_file_name <- paste0(
    "models/",
    model_name,
    "_LightGBM_",
    Sys.Date(),
    "_",
    format(Sys.time(), "%H-%M")
  )

  data_split <- initial_validation_split(df, prop = c(0.8, 0.1), strata = label)
  train <- training(data_split)
  test <- testing(data_split)
  val_set <- validation_set(data_split)
  val <- validation(data_split)

  spec <- boost_tree(
    trees = tune(),
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    sample_size = tune(),
    loss_reduction = tune()
  ) |>
    set_mode("regression") |>
    set_engine("lightgbm", nthread = 8)

  rec <- recipe(label ~ ., data = train)

  wf <- workflow() |>
    add_recipe(rec) |>
    add_model(spec)

  plan(multisession, workers = 2)

  print(paste0("Starting Tuning: ", Sys.time()))

  tune_res <- wf |>
    tune_bayes(
      resamples = val_set,
      initial = tune_initial,
      iter = tune_iter,
      control = control_bayes(
        verbose = TRUE,
        no_improve = tune_no_improve,
        seed = set_seed,
        parallel_over = "everything"
      ),
      metrics = metric_set(rmse)
    )

  print(paste0("Ended Tuning: ", Sys.time()))

  model_metrics <- collect_metrics(tune_res)
  best_params <- select_best(tune_res, metric = "rmse")

  print(paste0("Starting Fitting: ", Sys.time()))

  final_model <- wf |>
    finalize_workflow(best_params) |>
    fit(data = train)

  print(paste0("Ended Fitting: ", Sys.time()))

  saveRDS(final_model, paste0(full_file_name, ".rds"))
  # xgb.save(final_model$fit, paste0(full_file_name, ".json"))

  preds <- predict(final_model, test, type = "numeric") |>
    bind_cols(test)

  preds_rmse <- rmse(preds, label, .pred)
  preds_rsq <- rsq(preds, label, .pred)
  preds_rsq_trad <- rsq_trad(preds, label, .pred)
  preds_mae <- mae(preds, label, .pred)

  metrics_df <- bind_rows(
    preds_rmse,
    preds_rsq,
    preds_rsq_trad,
    preds_mae
  )

  write.csv(metrics_df, paste0(full_file_name, "_Error.csv"), row.names = FALSE)

  cal_plot <- cal_plot_regression(preds, truth = label, estimate = .pred) +
    labs(title = paste0(model_name, " Calibration Plot"))

  vip_plot = final_model |>
    extract_fit_parsnip() |>
    vip(num_features = 40) +
    ggtitle(paste0("Feature Importance"))

  write.csv(
    model_metrics,
    paste0(full_file_name, "_Model_Metrics.csv"),
    row.names = FALSE
  )

  ggsave(
    filename = paste0(full_file_name, "_Calibration.png"),
    plot = cal_plot,
    width = 10,
    height = 8,
    dpi = 300
  )

  ggsave(
    filename = paste0(full_file_name, "_VIP.png"),
    plot = vip_plot,
    width = 10,
    height = 8,
    dpi = 300
  )

  list(
    train = train,
    test = test,
    val_set = val_set,
    val = val,
    model_metrics = model_metrics,
    final_model = final_model
  )
}

train_multi_classification_model <- function(
  df,
  model_name,
  tune_initial = 20,
  tune_iter = 100,
  tune_no_improve = 5,
  set_seed = 1015
) {
  set.seed(set_seed)

  plan(sequential)
  plan(multisession, workers = 2)
  gc()

  if (!dir.exists("/Users/carlfoust/Code/Stuff-Plus/models")) {
    message("Directory 'models/' not found. Exiting early.")
    return(NULL)
  }

  full_file_name <- paste0(
    "/Users/carlfoust/Code/Stuff-Plus/",
    "models/",
    model_name,
    "_XGBoost_",
    Sys.Date(),
    "_",
    format(Sys.time(), "%H-%M")
  )

  data_split <- initial_validation_split(df, prop = c(0.8, 0.1), strata = label)
  train <- training(data_split)
  test <- testing(data_split)
  val_set <- validation_set(data_split)
  val <- validation(data_split)
  
  # --- UPDATED SECTION START ---
  # Define monotonic constraints. 
  # 1 = Increasing, -1 = Decreasing, 0 = None.
  # We use a named list to ensure it maps correctly to 'release_speed'
  spec <- boost_tree(
    trees = tune(),
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    sample_size = tune(),
    loss_reduction = tune()
  ) |>
    set_mode("classification") |>
    set_engine(
      "xgboost", 
      nthread = 1,
      monotone_constraints = list(release_speed = 1) 
    )
  # --- UPDATED SECTION END ---

  rec <- recipe(label ~ ., data = train)

  wf <- workflow() |>
    add_recipe(rec) |>
    add_model(spec)

  print(paste0("Starting Tuning: ", Sys.time()))

  tune_res <- wf |>
    tune_bayes(
      resamples = val_set,
      initial = tune_initial,
      iter = tune_iter,
      control = control_bayes(
        verbose = TRUE,
        no_improve = tune_no_improve,
        seed = set_seed,
        parallel_over = "everything"
      ),
      metrics = metric_set(roc_auc)
    )

  print(paste0("Ended Tuning: ", Sys.time()))

  model_metrics <- collect_metrics(tune_res)
  best_params <- select_best(tune_res, metric = "roc_auc")

  write.csv(
    model_metrics,
    paste0(full_file_name, "_Model_Metrics.csv"),
    row.names = FALSE
  )

  print(paste0("Starting Fitting: ", Sys.time()))

  final_model <- wf |>
    finalize_workflow(best_params) |>
    fit(data = train)

  print(paste0("Ended Fitting: ", Sys.time()))

  saveRDS(final_model, paste0(full_file_name, ".rds"))
  xgb.save(
    parsnip::extract_fit_engine(final_model),
    paste0(full_file_name, ".json")
  )

  preds <- predict(final_model, test, type = "prob") |>
    bind_cols(test)

  auc_res <- roc_curve(
    data = preds,
    truth = label,
    .pred_0,
    .pred_1,
    .pred_2,
    .pred_3,
    .pred_4,
    .pred_5
  )

  log_loss <- mn_log_loss(
    data = preds,
    truth = label,
    .pred_0,
    .pred_1,
    .pred_2,
    .pred_3,
    .pred_4,
    .pred_5
  )

  metrics_df <- bind_rows(auc_res, log_loss)

  write.csv(metrics_df, paste0(full_file_name, "_Error.csv"), row.names = FALSE)

  auc_res_plot <- autoplot(auc_res) +
    labs(
      title = paste0(model_name, " ROC Curves"),
      subtitle = "One-vs-All Multiclass ROC",
      x = "1 - Specificity",
      y = "Sensitivity"
    )

  ggsave(
    filename = paste0(full_file_name, "_AUC.png"),
    plot = auc_res_plot,
    width = 10,
    height = 8,
    dpi = 300
  )

  cal_plot <- cal_plot_breaks(
    .data = preds,
    truth = label,
    estimate = c(.pred_0, .pred_1, .pred_2, .pred_3, .pred_4, .pred_5)
  ) +
    labs(title = paste0(model_name, " Calibration Plot"))

  ggsave(
    filename = paste0(full_file_name, "_Calibration.png"),
    plot = cal_plot,
    width = 10,
    height = 8,
    dpi = 300
  )

  vip_plot = final_model |>
    extract_fit_parsnip() |>
    vip(num_features = 40) +
    ggtitle(paste0("Feature Importance"))

  ggsave(
    filename = paste0(full_file_name, "_VIP.png"),
    plot = vip_plot,
    width = 10,
    height = 8,
    dpi = 300
  )

  list(
    train = train,
    test = test,
    val_set = val_set,
    val = val,
    model_metrics = model_metrics,
    final_model = final_model
  )
}

train_multi_classification_lightGBM <- function(
  df,
  model_name,
  tune_initial = 20,
  tune_iter = 100,
  tune_no_improve = 5,
  set_seed = 1015
) {
  set.seed(set_seed)

  plan(sequential)
  plan(multisession, workers = 2)
  gc()

  dir.exists(file.path("models/"))
  full_file_name <- paste0(
    "models/",
    model_name,
    "_LightGBM_",
    Sys.Date(),
    "_",
    format(Sys.time(), "%H-%M")
  )

  data_split <- initial_validation_split(df, prop = c(0.8, 0.1), strata = label)
  train <- training(data_split)
  test <- testing(data_split)
  val_set <- validation_set(data_split)
  val <- validation(data_split)

  spec <- boost_tree(
    trees = tune(),
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    sample_size = tune(),
    loss_reduction = tune()
  ) |>
    set_mode("classification") |>
    set_engine("lightgbm", nthread = 1)

  rec <- recipe(label ~ ., data = train)

  wf <- workflow() |>
    add_recipe(rec) |>
    add_model(spec)

  print(paste0("Starting Tuning: ", Sys.time()))

  tune_res <- wf |>
    tune_bayes(
      resamples = val_set,
      initial = tune_initial,
      iter = tune_iter,
      control = control_bayes(
        verbose = TRUE,
        no_improve = tune_no_improve,
        seed = set_seed,
        parallel_over = "everything"
      ),
      metrics = metric_set(roc_auc)
    )

  print(paste0("Ended Tuning: ", Sys.time()))

  model_metrics <- collect_metrics(tune_res)
  best_params <- select_best(tune_res, metric = "roc_auc")

  write.csv(
    model_metrics,
    paste0(full_file_name, "_Model_Metrics.csv"),
    row.names = FALSE
  )

  print(paste0("Starting Fitting: ", Sys.time()))

  final_model <- wf |>
    finalize_workflow(best_params) |>
    fit(data = train)

  print(paste0("Ended Fitting: ", Sys.time()))

  saveRDS(final_model, paste0(full_file_name, ".rds"))
  # xgb.save(final_model$fit, paste0(full_file_name, ".json"))

  preds <- predict(final_model, test, type = "prob") |>
    bind_cols(test)

  roc_res <- roc_curve(
    preds,
    label,
    .pred_1,
    .pred_2,
    .pred_3,
    .pred_4,
    .pred_5,
    .pred_6,
    estimator = "hand_till"
  )
  log_loss <- mn_log_loss(
    data = preds,
    truth = label,
    .pred_1,
    .pred_2,
    .pred_3,
    .pred_4,
    .pred_5,
    .pred_6
  )

  metrics_df <- bind_rows(roc_res, log_loss)

  write.csv(metrics_df, paste0(full_file_name, "_Error.csv"), row.names = FALSE)

  auc_res_plot <- autoplot(roc_res)

  ggsave(
    filename = paste0(full_file_name, "_AUC.png"),
    plot = auc_res_plot,
    width = 10,
    height = 8,
    dpi = 300
  )

  cal_plot <- cal_plot_breaks(
    .data = preds,
    truth = label,
    estimate = c(.pred_1, .pred_2, .pred_3, .pred_4, .pred_5, .pred_6)
  ) +
    labs(title = paste0(model_name, " Calibration Plot"))

  ggsave(
    filename = paste0(full_file_name, "_Calibration.png"),
    plot = cal_plot,
    width = 10,
    height = 8,
    dpi = 300
  )

  vip_plot = final_model |>
    extract_fit_parsnip() |>
    vip(num_features = 40) +
    ggtitle(paste0("Feature Importance"))

  ggsave(
    filename = paste0(full_file_name, "_VIP.png"),
    plot = vip_plot,
    width = 10,
    height = 8,
    dpi = 300
  )

  list(
    train = train,
    test = test,
    val_set = val_set,
    val = val,
    model_metrics = model_metrics,
    final_model = final_model
  )
}
