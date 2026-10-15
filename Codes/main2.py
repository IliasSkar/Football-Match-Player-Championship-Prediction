import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from System import (
    device,
    set_seed,
    get_team_feature_columns,
    Config,
    CsvLoadMergeAndFeatureAdding,
    clean_contaminated_rows,
    teamtable,
    teams_league,
    build_team_features,
    build_form_sequences,
    build_match_dataset,
    build_elo_ratings,
    lookup_sequences,
    normalize_features,
    normalize_player_features,
    build_player_sequences,
    build_player_match_dataset,
    rollingavgfeatures,
    wfeval,
    modeltrain,
    player_train_model,
    SyndyasmenoBuild,
    playermodelbuild,
    player_columns_features,
    run_menu,
    backtest_team_model,
    plot_training_history,
    plot_fold_accuracy)

base_dir=os.path.dirname(os.path.abspath(__file__))
data_root=os.path.join(base_dir, "data")

DATA_DIRS = {
    "Scotch":  os.path.join(data_root, "Scotch"),
    "English": os.path.join(data_root, "English"),
    "Greek":   os.path.join(data_root, "Greek"),
    "Italian": os.path.join(data_root, "Italian"),}

FOLD_DATES=[
    ("2022-08-01", "2022-12-01", "2023-08-01"),
    ("2023-04-01", "2023-08-01", "2024-04-01"),
    ("2024-04-01", "2024-08-01", "2025-04-01"),
    ("2025-04-01", "2025-08-01", "2026-06-01"),]

TEAM_TRAIN_CUTOFF, TEAM_VAL_CUTOFF, TEAM_TEST_END = FOLD_DATES[-1]

PLAYER_TRAIN_CUTOFF="2025-06-01"
PLAYER_VAL_CUTOFF="2025-09-01"
TEAM_ROLLING_WINDOW=7
TEAM_SEQ_WINDOW=7
PLAYER_SEQ_WINDOW=5
RETRAIN=False

TEAM_MODEL_PATH=os.path.join(base_dir, "saved_models", "team_model.pt")
PLAYER_MODEL_PATH=os.path.join(base_dir, "saved_models", "player_model.pt")

def save_model(trained: dict, path: str, extra: dict = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state":trained["model"].state_dict(),
        "feature_columns":trained.get("feature_columns", []),
        "league_to_id":trained.get("league_to_id", {}),}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"Saved: {path}")

def load_team_model(path: str, normalized: dict) -> dict:
    checkpoint=torch.load(path, map_location=device)
    league_to_id=checkpoint["league_to_id"]
    feat_cols=checkpoint["feature_columns"]
    model_kwargs=checkpoint.get("model_kwargs", {})
    sample=next(iter(normalized["home_sequences"].values()))
    lstm_in= sample.shape[-1]
    static_in=len(feat_cols)
    n_leagues=len(league_to_id)
    model=SyndyasmenoBuild(lstm_in, static_in, n_leagues, **model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Team model loaded: {path}")
    return {
        "model":model,
        "feature_columns":feat_cols,
        "league_to_id":league_to_id,
        "device":device,}

def load_player_model(path: str, normalized_player: dict) -> dict:
    checkpoint=torch.load(path, map_location=device)
    position_to_id=checkpoint["position_to_id"]
    league_to_id=checkpoint["league_to_id"]
    feat_cols=checkpoint["player_feature_columns"]
    sample=next(iter(
        normalized_player["player_sequences"].values()))
    model = playermodelbuild(
        sample.shape[-1], len(feat_cols),
        len(position_to_id), len(league_to_id)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Player model loaded:{path}")
    return {
        "model":model,
        "player_feature_columns":feat_cols,
        "position_to_id":position_to_id,
        "league_to_id":league_to_id,
        "device":device,}

def main():
    set_seed(42)
    print("Football Prediction System — Start".center(48))
    print("[ 1 / 8 ]  Loading CSV...")
    raw_df=CsvLoadMergeAndFeatureAdding(DATA_DIRS)
    raw_df=clean_contaminated_rows(raw_df,
    squad_column="Squad",
    source_column="source_file_team",
    league_column="league",
    season_column="season")
    
    print("\n[ 2 / 8 ]  Team features...")
    team_match_df=teamtable(raw_df, config=Config)
    team_match_df=build_elo_ratings(team_match_df, config=Config)
    print(team_match_df["elo_before"].describe())
    print(team_match_df.groupby("Squad")["elo_before"].mean().sort_values(ascending=False).head(10))
    print(team_match_df.groupby("Squad")["elo_before"].mean().sort_values().head(10))
    team_features_df=build_team_features(team_match_df, rolling_window=TEAM_ROLLING_WINDOW, config=Config)
    match_dataset = build_match_dataset(team_features_df,config=Config)
    print(f"Matches: {len(match_dataset)}")
    print("Match_dataset Columns:", match_dataset.columns.tolist())
    match_dataset["elo_diff"]=match_dataset["elo_before_home"] - match_dataset["elo_before_away"]
    goal_diff=match_dataset["home_goals"] - match_dataset["away_goals"]
    corr = np.corrcoef(match_dataset["elo_diff"], goal_diff)[0, 1]
    print(f" Elo_diff with goal difference association: {corr:.3f}")
    
    print("\n[ 3 / 8 ]  Team sequences (LSTM)...")
    team_seq_features = [
        c for c in [
            Config["goals_for_col"],
            Config["goals_against_col"],
            Config["xg_col"],
            Config["xga_col"],] if c in team_features_df.columns]
    all_sequences=build_form_sequences(
        team_features_df,
        sequence_features=team_seq_features,
        n_last_matches=TEAM_SEQ_WINDOW,
        config= Config,)
    home_sequences, away_sequences=lookup_sequences(
        match_df=match_dataset,
        sequences=all_sequences,
        n_last_matches=TEAM_SEQ_WINDOW,
        n_sequence_features=len(team_seq_features),)

    print("\n[ 4 / 8 ]  Team features Standardization...")
    team_feature_columns=get_team_feature_columns(match_dataset)
    team_feature_columns=team_feature_columns + ["elo_diff"]    
    
    if not team_feature_columns:
        raise ValueError("No valid team feature columns found -- check build_match_dataset().")

    normalized_team=normalize_features(
    match_df=match_dataset,
    home_sequences=home_sequences,
    away_sequences=away_sequences,
    feature_columns=team_feature_columns,
    train_cutoff_date=TEAM_TRAIN_CUTOFF,)
    normalized_team["feature_columns"] = team_feature_columns
    normalized_team["all_team_sequences"] = all_sequences
    print(f"Team feature columns: {len(team_feature_columns)}")
    run_wfeval= False 
    if run_wfeval:      
        fold_results=wfeval(
            match_dataset, team_feature_columns, home_sequences,
            away_sequences, all_sequences, FOLD_DATES,
            lr=0.0003, patience=20, num_epochs=150,)  
        plot_fold_accuracy(fold_results)
    
    print("\n[ 5 / 8 ]  Player features...")
    player_df=rollingavgfeatures(raw_df, rolling_window=PLAYER_SEQ_WINDOW)
    player_df=build_player_match_dataset(
        player_dataframe=player_df,
        team_features_dataframe = team_features_df,)
    print(f"Players: {player_df['player'].nunique()}")
    print(f"Registrations: {len(player_df)}")
    
    print("\n[ 6 / 8 ]  Player sequences + Standardization...")
    player_seq_features = [c for c in player_columns_features if c in player_df.columns]
    player_sequences = build_player_sequences(
        player_dataframe=player_df,
        sequence_features=player_seq_features,
        n_last_matches=PLAYER_SEQ_WINDOW,)
    player_static_columns=(
        [f"avg_{c}_last{PLAYER_SEQ_WINDOW}" for c in player_columns_features
         if f"avg_{c}_last{PLAYER_SEQ_WINDOW}" in player_df.columns]
        + [f"season_avg_{c}" for c in player_columns_features
           if f"season_avg_{c}" in player_df.columns]
        + [c for c in player_df.columns
   if c.startswith("opponent_") and pd.api.types.is_numeric_dtype(player_df[c])])
    normalized_player = normalize_player_features(
        player_dataframe= player_df,
        player_feature_columns=player_static_columns,
        train_cutoff_date=PLAYER_TRAIN_CUTOFF,)

    seq_scaler= StandardScaler()
    train_cutoff_dt = pd.to_datetime(PLAYER_TRAIN_CUTOFF)
    train_seqs=[arr for (_, date), arr in player_sequences.items()
        if date <train_cutoff_dt]
    if train_seqs:
        non_zero=[r[~np.all(r == 0, axis=1)] for r in train_seqs]
        non_zero=[r for r in non_zero if len(r) > 0]
        if non_zero:
            seq_scaler.fit(np.vstack(non_zero))
    
    def scale_player_sequences(sequences: dict, scaler: StandardScaler) -> dict:
        scaled = {}
        for key, arr in sequences.items():
            shape = arr.shape
            flat_scaled = scaler.transform(arr.reshape(-1, shape[-1]))
            scaled[key] = flat_scaled.reshape(shape)
        return scaled

    player_sequences_scaled = scale_player_sequences(player_sequences, seq_scaler) if non_zero else player_sequences

    normalized_player["sequence_scaler"]=seq_scaler
    normalized_player["player_sequences"]=player_sequences
    normalized_player["player_feature_columns"]=player_static_columns
   
    print("\n[ 7 / 8 ]  Models...")

    if RETRAIN or not os.path.exists(TEAM_MODEL_PATH):
        print("Team model training...")
        set_seed(46)
        trained_team = modeltrain(
            normalized=normalized_team,
            feature_columns=team_feature_columns,
            train_cutoff_date=TEAM_TRAIN_CUTOFF,
            val_cutoff_date=TEAM_VAL_CUTOFF,
            lr=0.0003,
            patience=20,
            num_epochs=150)
        save_model(trained_team, TEAM_MODEL_PATH, extra={})
        
    else:
        print("Loading team model...")
        trained_team = load_team_model(TEAM_MODEL_PATH, normalized_team)

    trained_team["device"] = device
    print(f"Learned home_advantage: {trained_team['model'].home_advantage.item():.4f}")
    
    print("\n Final team backtest")

    final_result = backtest_team_model(
    trained_team,
    normalized_team,
    match_dataset,
    val_cutoff_date=TEAM_VAL_CUTOFF,
    end_date=TEAM_TEST_END,
    rho=-0.13,
    draw_boost=1.0,)    
    
    if RETRAIN or not os.path.exists(PLAYER_MODEL_PATH):
        print("Player model training...")
        trained_player = player_train_model(
            normalized_player=normalized_player,
            player_feature_columns=player_static_columns,
            player_sequences=player_sequences_scaled,
            train_cutoff_date=PLAYER_TRAIN_CUTOFF,
            val_cutoff_date=PLAYER_VAL_CUTOFF,)
        os.makedirs(os.path.dirname(PLAYER_MODEL_PATH), exist_ok=True)
        torch.save({
            "model_state":trained_player["model"].state_dict(),
            "player_feature_columns": player_static_columns,
            "position_to_id":normalized_player["position_to_id"],
            "league_to_id":normalized_player["league_to_id"],}, PLAYER_MODEL_PATH)
        print(f"Player model saved:{PLAYER_MODEL_PATH}")
    else:
        print("Loading player model...")
        trained_player=load_player_model(PLAYER_MODEL_PATH, normalized_player)

    trained_player["device"]=device
    
    if "history" in trained_team:
        plot_training_history(trained_team["history"], "Team Model")
    if "history" in trained_player:
        plot_training_history(trained_player["history"], "Player Model")
     
    print("\n[ 8 / 8 ]  Starting the Menu...\n")
    run_menu(
        trained_team=trained_team,
        trained_player=trained_player,
        normalized_team=normalized_team,
        normalized_player=normalized_player,
        team_features_df=team_features_df,
        player_df=player_df,
        player_sequences=player_sequences,)
    
if __name__ == "__main__":
    main()