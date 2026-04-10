generate_model_preds <- function(model, data, features, type = "prob", output_name = "pred") {
  if (nrow(data) == 0) {
    return(stats::setNames(data.frame(numeric()), output_name))
  }
  
  prediction_data <- data %>% 
    select(all_of(features)) %>% 
    xgb.DMatrix()
  
  preds <- predict(model, prediction_data, type = "response") %>% 
    as.data.frame()
  
  preds <- preds %>%
    rename(!!output_name := 1)
  
  return(preds)
}

generate_pitch_quality_outputs <- function(df) {
  
  model_data_rhp <- df |> 
    filter(is_lhp == 0) |> 
    rowid_to_column("ID") |> 
    select(-is_lhp)
  
  model_data_lhp <- df |> 
    filter(is_lhp == 1) |> 
    rowid_to_column("ID") |> 
    select(-is_lhp)
  
  model_rhp_ff_data = model_data_rhp |> filter(pitch_group == 'Fastball')
  model_rhp_bb_data = model_data_rhp |> filter(pitch_group == 'BreakingBall')
  model_rhp_os_data = model_data_rhp |> filter(pitch_group == 'Offspeed')
  
  rhp_ff_loc = xrv_values(model_rhp_ff_data, 
                            models[[".rhp_ff_loc_whiff_model"]], 
                            models[[".rhp_ff_loc_foul_model"]], 
                            models[[".rhp_ff_loc_bip_model"]],
                            models[[".rhp_ff_loc_cs_model"]],
                            models[[".rhp_ff_loc_hbp_model"]],
                            models[[".rhp_ff_loc_swing_model"]],
                            model_type = "loc",
                            pitch_type = "ff")
  
  # rhp_bb_stuff = xrv_values(model_rhp_bb_data,
  #                           models[[".rhp_bb_whiff_model"]],
  #                           models[[".rhp_bb_foul_model"]],
  #                           models[[".rhp_bb_bip_model"]],
  #                           pitch_type = "bb")
  # 
  # rhp_os_stuff = xrv_values(model_rhp_os_data,
  #                           models[[".rhp_os_whiff_model"]],
  #                           models[[".rhp_os_foul_model"]],
  #                           models[[".rhp_os_bip_model"]],
  #                           pitch_type = "os")
  
  model_lhp_ff_data = model_data_lhp |> filter(pitch_group == 'Fastball')
  model_lhp_bb_data = model_data_lhp |> filter(pitch_group == 'BreakingBall')
  model_lhp_os_data = model_data_lhp |> filter(pitch_group == 'Offspeed')
  
  lhp_ff_loc = xrv_values(model_lhp_ff_data, 
                            models[[".lhp_ff_whiff_model"]], 
                            models[[".lhp_ff_foul_model"]], 
                            models[[".lhp_ff_bip_model"]],
                            pitch_type = "ff")
  
  # lhp_bb_stuff = xrv_values(model_lhp_bb_data,
  #                           models[[".lhp_bb_whiff_model"]],
  #                           models[[".lhp_bb_foul_model"]],
  #                           models[[".lhp_bb_bip_model"]],
  #                           pitch_type = "bb")
  # 
  # lhp_os_stuff = xrv_values(model_lhp_os_data,
  #                           models[[".lhp_os_whiff_model"]],
  #                           models[[".lhp_os_foul_model"]],
  #                           models[[".lhp_os_bip_model"]],
  #                           pitch_type = "os")
  
  stuff_combined <- rbind(rhp_ff_loc, 
                          # rhp_bb_stuff,
                          # rhp_os_stuff,
                          lhp_ff_loc, 
                          # lhp_bb_stuff,
                          # lhp_os_stuff
  )
  
  return(stuff_combined)
}

xrv_values <- function(
    df, 
    whiff_model, 
    foul_model, 
    bip_model, 
    hbp_model, 
    cs_model, 
    model_type, 
    handedness, 
    pitch_type
  ) 
{
  if (model_type == "loc") {
    quality_features <- c("is_lhh", "plate_x", "plate_z",
                        "sz_top", "sz_bot", "dist_from_mid")
  } else if (model_type == "stuff") {
    if (pitch_type == "ff") {
      quality_features <- c("is_lhh", "plate_x", "plate_z",
                            "sz_top", "sz_bot", "dist_from_mid")
    } else if (pitch_type == "bb") {
      quality_features <- c("is_lhh", "release_speed", "release_spin_rate", 
                            "release_extension", "release_pos_x", "release_pos_z", 
                            "a_vert", "a_horz",
                            "spin_axis", "arm_angle", "xVAA", "xHAA",
                            "rel_speed_fb_diff", "a_horz_fb_diff", "a_vert_fb_diff")
    } else if (pitch_type == "os") {
      quality_features <- c("is_lhh", "release_speed", "release_spin_rate", 
                            "release_extension", "release_pos_x", "release_pos_z", 
                            "a_vert", "a_horz",
                            "spin_axis", "arm_angle", "xVAA", "xHAA",
                            "rel_speed_fb_diff", "a_horz_fb_diff", "a_vert_fb_diff")
    } else {
      message("Type has to be one of: ff, bb, or os")
      return(NULL)
    }
  } else if (model_type == "quality") {
    if (pitch_type == "ff") {
      quality_features <- c("is_lhh", "plate_x", "plate_z",
                            "sz_top", "sz_bot", "dist_from_mid")
    } else if (pitch_type == "bb") {
      quality_features <- c("is_lhh", "release_speed", "release_spin_rate", 
                            "release_extension", "release_pos_x", "release_pos_z", 
                            "a_vert", "a_horz",
                            "spin_axis", "arm_angle", "xVAA", "xHAA",
                            "rel_speed_fb_diff", "a_horz_fb_diff", "a_vert_fb_diff")
    } else if (pitch_type == "os") {
      quality_features <- c("is_lhh", "release_speed", "release_spin_rate", 
                            "release_extension", "release_pos_x", "release_pos_z", 
                            "a_vert", "a_horz",
                            "spin_axis", "arm_angle", "xVAA", "xHAA",
                            "rel_speed_fb_diff", "a_horz_fb_diff", "a_vert_fb_diff")
    } else {
      message("Type has to be one of: ff, bb, or os")
      return(NULL)
    }
  } else {
    message("Model type has to be one of: loc, stuff, or quality")
    return(NULL)
  }
  
  
  context_neut_rv = structure(
    list(weak_crv = -0.09175214366607343, #1
         topped_crv = -0.11151027302174975, #2
         under_crv = -0.18667004557381425, #3
         flare_burner_crv = 0.24716735722771419, #4
         solid_crv = 0.17268876331232244, #5
         barrel_crv = 0.776270800949034, #6
         foul_crv = -0.034489988876527836, 
         whiff_crv = -0.11376445316237163,
         called_strike_crv = -0.06515756600129853,
         hbp_crv = 0.36601476014760065), 
    class = c("tbl_df",
              "tbl", 
              "data.frame"), 
    row.names = c(NA, -1L)
  )
  
  model_data <- df %>% 
    cross_join(context_neut_rv)
  
  whiff_preds <- generate_model_preds(model = whiff_model,
                                      data = model_data,
                                      features = quality_features,
                                      output_name = "prob_whiff_given_swing") %>% 
    mutate(prob_contact_given_swing = prob_whiff_given_swing,
           prob_whiff_given_swing = 1 - prob_whiff_given_swing)
  
  foul_preds <- generate_model_preds(model = foul_model,
                                     data = model_data,
                                     features = quality_features,
                                     output_name = "prob_foul_given_contact") %>% 
    mutate(prob_bip_given_contact = prob_foul_given_contact,
           prob_foul_given_contact = 1 - prob_foul_given_contact)
  
  bip_preds <- predict(bip_model, model_data %>% 
                         select(all_of(quality_features)), type = "prob")  %>% 
    as.data.frame() |> 
    rename(prob_weak_given_bip_qual = V1, 
           prob_topped_given_bip_qual = V2,
           prob_under_given_bip_qual = V3, 
           prob_flareburner_given_bip_qual = V4,
           prob_solid_given_bip_qual = V5, 
           prob_barrel_given_bip_qual = V6)
  
  hbp_preds <- generate_model_preds(model = hbp_model,
                                     data = model_data,
                                     features = quality_features,
                                     output_name = "prob_hit_by_pitch")
  
  cs_preds <- generate_model_preds(model = cs_model,
                                    data = model_data,
                                    features = quality_features,
                                    output_name = "prob_called_strike")
  
  model_preds <- model_data %>% 
    select(pitch_uid, pitcher, game_year, game_date, game_type, weak_crv:hbp_crv, is_whiff, is_foul,
           is_swing) %>% 
    bind_cols(whiff_preds,
              foul_preds,
              bip_preds,
              hbp_preds,
              cs_preds)
  
  probs <- model_preds %>% 
    mutate(prob_swinging_strike = prob_whiff_given_swing,
           prob_foul_ball = prob_contact_given_swing * prob_foul_given_contact,
           prob_weak = prob_contact_given_swing * prob_bip_given_contact * prob_weak_given_bip_qual,
           prob_topped = prob_contact_given_swing * prob_bip_given_contact * prob_topped_given_bip_qual,
           prob_under = prob_contact_given_swing * prob_bip_given_contact * prob_under_given_bip_qual,
           prob_flareburner = prob_contact_given_swing * prob_bip_given_contact * prob_flareburner_given_bip_qual,
           prob_solid = prob_contact_given_swing * prob_bip_given_contact * prob_solid_given_bip_qual,
           prob_barrel = prob_contact_given_swing * prob_bip_given_contact * prob_barrel_given_bip_qual)
  
  cal_df <- probs %>% filter(is_swing == 1)
  
  whiff_cal <- platt_fit(cal_df$is_whiff, cal_df$prob_swinging_strike)
  whiff_probs_cal <- predict(whiff_cal, probs$prob_swinging_strike)
  
  foul_cal <- platt_fit(cal_df$is_foul, cal_df$prob_foul_ball)
  foul_probs_cal <- predict(foul_cal, probs$prob_foul_ball)
  
  probs_cal <- probs %>% 
    bind_cols(tibble(prob_swinging_strike_cal = whiff_probs_cal),
              tibble(prob_foul_ball_cal = foul_probs_cal))
  
  xrv_qual <- probs_cal %>%
    mutate(xrv_whiff = prob_swinging_strike * whiff_crv,
           xrv_foul = prob_foul_ball * foul_crv,
           xrv_whiff_cal = prob_swinging_strike_cal * whiff_crv,
           xrv_foul_cal = prob_foul_ball_cal * foul_crv,
           xrv_bip = (prob_weak * weak_crv) + (prob_topped * topped_crv) +
             (prob_under * under_crv) + (prob_flareburner * flare_burner_crv) +
             (prob_solid * solid_crv) + (prob_barrel * barrel_crv),
           xrv_cs = prob_called_strike * called_strike_crv,
           xrv_hbp = prob_hit_by_pitch * hbp_crv,
           xrv_stuff = (xrv_whiff) + (xrv_foul) + (xrv_bip) + (xrv_cs) + (xrv_hbp),
           xrv_stuff_cal = (xrv_whiff_cal) + (xrv_foul_cal) + (xrv_bip) + (xrv_cs) + (xrv_hbp)
    ) %>%
    select(pitch_uid, pitcher, game_year, game_date, game_type, 
           prob_swinging_strike:prob_barrel, prob_called_strike, prob_hit_by_pitch,
           xrv_whiff, xrv_foul, xrv_bip, xrv_cs, xrv_hbp, xrv_stuff, 
           prob_swinging_strike_cal, prob_foul_ball_cal, xrv_stuff_cal)
  
  return(xrv_qual)
}

platt_fit <- function(y, p, eps = 1e-15) {
  # y: binary labels (0/1, FALSE/TRUE, or factor with 2 levels)
  # p: predicted probabilities for the POSITIVE class
  # returns: list with predict() and coefficients
  
  # Coerce y to 0/1
  if (is.factor(y)) {
    if (nlevels(y) != 2) stop("y must have exactly 2 levels.")
    y01 <- as.integer(y == levels(y)[2])  # treats 2nd level as positive
  } else if (is.logical(y)) {
    y01 <- as.integer(y)
  } else {
    y01 <- as.integer(y)
  }
  if (!all(y01 %in% c(0L, 1L))) stop("y must be binary (0/1, logical, or 2-level factor).")
  
  # Clamp p to avoid infinite logits
  p <- as.numeric(p)
  if (anyNA(p)) stop("p contains NA.")
  if (any(p < 0 | p > 1)) stop("p must be in [0, 1].")
  p_clamped <- pmin(pmax(p, eps), 1 - eps)
  
  # Platt: fit logistic regression y ~ logit(p)
  logit_p <- qlogis(p_clamped)
  df <- data.frame(y = y01, logit_p = logit_p)
  
  fit <- stats::glm(y ~ logit_p, data = df, family = stats::binomial())
  
  # Predictor function
  predict_fn <- function(p_new) {
    p_new <- as.numeric(p_new)
    if (anyNA(p_new)) stop("p_new contains NA.")
    if (any(p_new < 0 | p_new > 1)) stop("p_new must be in [0, 1].")
    p_new <- pmin(pmax(p_new, eps), 1 - eps)
    lp <- stats::predict(fit, newdata = data.frame(logit_p = qlogis(p_new)), type = "link")
    stats::plogis(lp)
  }
  
  out <- list(
    model = fit,
    intercept = unname(stats::coef(fit)[1]),
    slope = unname(stats::coef(fit)[2]),
    eps = eps,
    predict = predict_fn
  )
  class(out) <- "platt_scaler"
  out
}

predict.platt_scaler <- function(object, p_new, ...) {
  object$predict(p_new)
}
