import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import math
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    
num_epochs=60 
learning_rate=0.0003 
batch_size=128 
patience=10 

Config={
        "date_col":"Date", 
        "team_col":"Squad", 
        "opponent_col":"Opponent", 
        "venue_col":"Venue", 
        "result_col":"Result",
        "goals_for_col":"GF",
        "goals_against_col":"GA",
        "xg_col":"xG",
        "xga_col":"xGA",
        "season_col":"season",
        "league_col":"league",}

def goalparsefromresult(dataframe:pd.DataFrame,config:dict=Config)->pd.DataFrame:
    c=config   
    dataframe=dataframe.copy()
    extracted=(dataframe[c["result_col"]].astype(str).str.extract(r"(\d+)\s*[–-]\s*(\d+)"))
    dataframe[c["goals_for_col"]]=pd.to_numeric(extracted[0], errors="coerce")
    dataframe[c["goals_against_col"]] = pd.to_numeric(extracted[1], errors="coerce")
    n_missing = dataframe[c["goals_for_col"]].isna().sum()
    if n_missing>0:
       print(f"Goal parsing skipped {n_missing} rows.")
    return dataframe 

def teamtable(dataframe_players:pd.DataFrame,config:dict=Config)->pd.DataFrame: 
    c=config
    dataframe=goalparsefromresult(dataframe_players,Config)
    
    match_cols=[
        c["date_col"],c["team_col"],c["opponent_col"],c["venue_col"],
        c["result_col"],c["goals_for_col"],c["goals_against_col"],
        c["season_col"],c["league_col"]]
    
    for extra in ["Comp", "Round", "Day"]:
        if extra in dataframe.columns:
            match_cols.append(extra)
    match_cols=[col for col in match_cols if col in dataframe.columns]
    team_matches = (
        dataframe.drop_duplicates(
            subset=[c["date_col"],c["team_col"],c["opponent_col"]])[match_cols].reset_index(drop=True))
    return team_matches

def build_team_features(dataframe:pd.DataFrame,
                         rolling_window:int=5,
                         config:dict=Config)-> pd.DataFrame:
    c=config
    dataframe=dataframe.copy()
    dataframe[c["date_col"]] = pd.to_datetime(dataframe[c["date_col"]], errors="coerce")
    dataframe = dataframe.sort_values([c["team_col"],c["season_col"],c["date_col"]])
    dataframe["_is_win"]=(dataframe[c["goals_for_col"]]>dataframe[c["goals_against_col"]]).astype(float)
    group_cols=[c["team_col"],c["season_col"]]
    grouped=dataframe.groupby(group_cols, group_keys=False)
    n= rolling_window

    def roll(s):
        return s.shift(1).rolling(window=n, min_periods=1).mean()

    def expand(s):
        return s.shift(1).expanding(min_periods=1).mean()

    dataframe[f"avg_goals_for_last{n}"]=grouped[c["goals_for_col"]].apply(roll)
    dataframe[f"avg_goals_against_last{n}"]=grouped[c["goals_against_col"]].apply(roll)
    dataframe[f"win_rate_last{n}"]=grouped["_is_win"].apply(roll)  
    dataframe["season_avg_goals_for"]= grouped[c["goals_for_col"]].apply(expand)
    dataframe["season_avg_goals_against"] = grouped[c["goals_against_col"]].apply(expand)
    dataframe["season_win_rate"]= grouped["_is_win"].apply(expand)

    if c["xg_col"] in dataframe.columns:
        dataframe[f"avg_xg_last{n}"]=grouped[c["xg_col"]].apply(roll)
    if c["xga_col"] in dataframe.columns:
        dataframe[f"avg_xga_last{n}"]=grouped[c["xga_col"]].apply(roll)    
    if c["venue_col"] in dataframe.columns:
        dataframe["_is_home"]=(dataframe[c["venue_col"]]=="Home").astype(float)
        venue_group= dataframe.groupby(
            group_cols+[c["venue_col"]],group_keys=False)
        dataframe[f"avg_goals_for_venue_last{n}"] = venue_group[
            c["goals_for_col"]].apply(roll)
        dataframe = dataframe.drop(columns=["_is_home"], errors="ignore")
    dataframe=dataframe.drop(columns=["_is_win"], errors="ignore")
    return dataframe

def build_form_sequences(team_features_df:pd.DataFrame,
                          sequence_features:list,
                          n_last_matches:int=5,
                          config:dict=Config)-> dict:

    c=config
    df=team_features_df.copy()
    df[c["date_col"]]=pd.to_datetime(df[c["date_col"]], errors="coerce")
    df=df.sort_values([c["team_col"], c["season_col"], c["date_col"]])
    sequences = {}
    for (team, season), group in df.groupby(
        [c["team_col"], c["season_col"]]):
        group=group.reset_index(drop=True)
        avail=[f for f in sequence_features if f in group.columns]
        values=group[avail].to_numpy(dtype=float)
        if len(avail)<len(sequence_features):
            pad=np.zeros((len(group),len(sequence_features)-len(avail)))
            values=np.hstack([values, pad])

        for i in range(len(group)):
            history=values[max(0,i-n_last_matches):i]

            if len(history)==0:
                history=np.zeros((n_last_matches,len(sequence_features)))
            elif len(history)<n_last_matches:
                pad_len=n_last_matches-len(history)
                history=np.vstack([np.zeros((pad_len, len(sequence_features))),history])

            match_date = group.loc[i, c["date_col"]]
            sequences[(team, match_date)]=history
    return sequences

def build_match_dataset(team_features_dataframe:pd.DataFrame,
                         config:dict=Config)-> pd.DataFrame:
    
    c=config
    dataframe=team_features_dataframe.copy()
    dataframe[c["date_col"]]=pd.to_datetime(dataframe[c["date_col"]], errors="coerce")

    if c["venue_col"] not in dataframe.columns:
        raise ValueError(f"Missing column '{c['venue_col']}'(Home/Away)")
    home_rows=dataframe[dataframe[c["venue_col"]]=="Home"].copy()
    away_rows=dataframe[dataframe[c["venue_col"]]=="Away"].copy()
    merged=home_rows.merge(away_rows,
        left_on=[c["date_col"], c["team_col"],c["season_col"], c["league_col"]],
        right_on=[c["date_col"], c["opponent_col"],c["season_col"], c["league_col"]],
        suffixes = ("_home", "_away"))
    merged=merged.rename(columns={
        f"{c['team_col']}_home":"home_team",
        f"{c['team_col']}_away":"away_team",
        f"{c['goals_for_col']}_home":"home_goals",
        f"{c['goals_for_col']}_away":"away_goals",})
    winner=pd.Series(np.nan,index=merged.index,dtype=object)
    winner[merged["home_goals"]>merged["away_goals"]]="H"
    winner[merged["home_goals"]==merged["away_goals"]]="D"
    winner[merged["home_goals"]<merged["away_goals"]]="A"
    merged["winner"]=winner
    keep=["Date","home_team","away_team",
            "season","league","home_goals","away_goals","winner"]
    engineered=[col for col in merged.columns
        if col not in keep and (col.endswith("_home") or col.endswith("_away") or col.startswith("home_") or col.startswith("away_"))]
    final_cols=keep+[c for c in engineered if c not in keep]
    final_cols=list(dict.fromkeys(final_cols))
    return merged[[c for c in final_cols if c in merged.columns]]
    
def get_team_feature_columns(match_dataset: pd.DataFrame) -> list:
    leak_columns = {
        "GA_home", "GA_away", "GF_home", "GF_away",
        "Result_home", "Result_away", "Venue_home", "Venue_away",
        "Opponent_home", "Opponent_away", "Comp_home", "Comp_away",
        "Round_home", "Round_away", "Day_home", "Day_away",}
    non_feature_columns={
        "Date","home_team","away_team","season","league",
        "winner","home_goals", "away_goals","league_id",}

    candidate_columns = [c for c in match_dataset.columns
        if c not in non_feature_columns
        and c not in leak_columns
        and (c.endswith("_home") or c.endswith("_away"))]
    return [c for c in candidate_columns
        if pd.api.types.is_numeric_dtype(match_dataset[c])]

def CsvLoadMergeAndFeatureAdding(data_dirs:dict)->pd.DataFrame: 
    all_dataframes=[] 
    for league_name,league_dir in data_dirs.items():
        if not os.path.isdir(league_dir):
            print(f"Directory not found: {league_dir}")
            continue
        for season_folder in sorted(os.listdir(league_dir)):
            season_path=os.path.join(league_dir,season_folder)
            if not os.path.isdir(season_path):
                continue
            for file in os.listdir(season_path):
                if not file.lower().endswith("csv"):
                    continue
                filepath=os.path.join(season_path,file)                
                try:
                    df=pd.read_csv(filepath,encoding="utf-8-sig")
                    df["season"]=season_folder
                    df["league"]=league_name
                    df["source_file_team"] = file.split("_")[0].replace("-", " ").strip()
                    all_dataframes.append(df)
                    print(f"Loaded {file} from {league_name} {season_folder} with {len(df)} rows.")
                except Exception as ex:
                    print(f"Failed to load {file}: {ex}")
                    continue
    if not all_dataframes:
        raise ValueError("No CSV files were found")
    merged=pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Total rows: {len(merged)}")
    for league in merged["league"].unique():
        count=len(merged[merged["league"]==league])
        print(f"{league} rows: {count}")
    return merged

def clean_contaminated_rows(raw_df:pd.DataFrame,
                             squad_column:str="Squad",
                             source_column:str="source_file_team",
                             league_column:str="league",
                             season_column:str="season")->pd.DataFrame:
    df=raw_df.copy()
    group_cols=[league_column, season_column, source_column]
    canonical=(
        df.groupby(group_cols)[squad_column]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("canonical_squad")
        .reset_index())
    df=df.merge(canonical, on=group_cols, how="left")
    before=len(df)
    clean_df=df[df[squad_column] == df["canonical_squad"]].drop(columns=["canonical_squad"])
    removed=before-len(clean_df)
    print(f"Removed {removed} contaminated rows out of {before} total rows ({removed/before:.2%}).")
    return clean_df.reset_index(drop=True)

def lookup_sequences(match_df:pd.DataFrame,sequences:dict,date_column:str="Date",home_team_column:str="home_team",
                     away_team_column:str="away_team",n_last_matches:int=7,n_sequence_features:int=2)->tuple:
    dataframe=match_df.copy()
    dataframe[date_column]=pd.to_datetime(dataframe[date_column],errors="coerce")
    zero_fallback=np.zeros((n_last_matches,n_sequence_features))
    home_sequences,away_sequences={},{}
    n_missing_home,n_missing_away=0,0  
    for idx,row in dataframe.iterrows():
        date=row[date_column]
        home_key=(row[home_team_column],date)
        if home_key in sequences:
            home_sequences[idx]=sequences[home_key]
        else:
            home_sequences[idx]=zero_fallback.copy()
            n_missing_home +=1
        away_key=(row[away_team_column],date)
        if away_key in sequences:
            away_sequences[idx]=sequences[away_key]
        else:
            away_sequences[idx]=zero_fallback.copy()
            n_missing_away +=1
    if n_missing_home or n_missing_away:
        print(f"Missing sequence keys for {n_missing_home} home rows and {n_missing_away} away rows.")
    return home_sequences, away_sequences

def normalize_features(match_df:pd.DataFrame, home_sequences:dict,away_sequences:dict,feature_columns:list,train_cutoff_date:str,
                       date_column:str="Date",league_column:str="league"):
    dataframe=match_df.copy()
    dataframe[date_column]=pd.to_datetime(dataframe[date_column],errors="coerce")
    train_cutoff_date=pd.to_datetime(train_cutoff_date)
    train_mask=dataframe[date_column]<train_cutoff_date    
    if train_mask.sum()==0:
        raise ValueError("No rows were found before train_cutoff_date")
        
    static_medians=dataframe.loc[train_mask,feature_columns].median()
    dataframe[feature_columns] = dataframe[feature_columns].fillna(static_medians)
    static_scaler=StandardScaler()
    static_scaler.fit(dataframe.loc[train_mask,feature_columns])
    dataframe[feature_columns]=static_scaler.transform(dataframe[feature_columns])   
    leagues_sort=sorted(dataframe[league_column].dropna().unique().tolist())
    league_to_id={name:i for i,name in enumerate(leagues_sort)}
    dataframe["league_id"]=dataframe[league_column].map(league_to_id)
    
    def train_row_collect(sequences:dict)->list:
        rows=[]
        for idx,arr in sequences.items():
            if idx in dataframe.index and train_mask.loc[idx]:
                real_rows=arr[~np.all(arr==0,axis=1)]
                if len(real_rows)>0:
                    rows.append(real_rows)
        return rows
    train_rows=train_row_collect(home_sequences)+train_row_collect(away_sequences)
    if not train_rows:
        raise ValueError("No non-zero sequence rows were found before train_cutoff_date")
    stacked=np.vstack(train_rows)   
    sequence_scaler=StandardScaler()
    sequence_scaler.fit(stacked)

    def scale_sequences(sequences:dict)->dict:
        scaled={}
        for key,arr in sequences.items():
            shape=arr.shape
            flat=arr.reshape(-1,shape[-1])
            flat_scaled=sequence_scaler.transform(flat)
            scaled[key]=flat_scaled.reshape(shape)
        return scaled

    home_sequences_scaled=scale_sequences(home_sequences)
    away_sequences_scaled=scale_sequences(away_sequences)
    
    return {
        "match_df":dataframe,
        "home_sequences": home_sequences_scaled,
        "away_sequences": away_sequences_scaled,
        "static_scaler": static_scaler,
        "sequence_scaler": sequence_scaler,
        "static_medians": static_medians,
        "league_to_id": league_to_id,}

class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,)
        self.output_size = hidden_size
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]
 
class MLPEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list = (64, 32),
                 output_size: int = 32, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.mlp = nn.Sequential(*layers)
        self.output_size = output_size
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:     
        return self.mlp(x)

class SyndyasmenoMlpLstmModel(nn.Module):
    def __init__(self,
                 lstm_input_size:int,
                 static_input_size:int,
                 n_leagues:int,
                 lstm_hidden_size:int=64,
                 lstm_num_layers:int=1,
                 lstm_dropout:float=0.0,
                 mlp_hidden_sizes:tuple=(64,32),
                 mlp_output_size:int=32,
                 mlp_dropout:float=0.2,
                 league_embedding_dim:int=8,
                 head_hidden_size:int=32):
        super().__init__()       
        self.form_encoder=LSTMEncoder(lstm_input_size,lstm_hidden_size,lstm_num_layers,lstm_dropout)
        self.static_encoder=MLPEncoder(static_input_size,mlp_hidden_sizes,mlp_output_size,mlp_dropout)
        self.league_embedding=nn.Embedding(n_leagues, league_embedding_dim)        
        combined_size=(lstm_hidden_size*2)+mlp_output_size+league_embedding_dim
        self.head=nn.Sequential(
            nn.Linear(combined_size,head_hidden_size),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(head_hidden_size,2),)
        self.home_advantage=nn.Parameter(torch.tensor(0.1))
        
    def forward(self,
                home_seq:torch.Tensor,
                away_seq:torch.Tensor,
                static_features:torch.Tensor,
                league_ids:torch.Tensor)->tuple:
        home_embedding=self.form_encoder(home_seq)
        away_embedding=self.form_encoder(away_seq)
        static_embedding=self.static_encoder(static_features)
        league_emb=self.league_embedding(league_ids)        
        combined=torch.cat([home_embedding,away_embedding,static_embedding,league_emb],dim=1)
        rout=self.head(combined)
        log_l_home=rout[:,0]+self.home_advantage
        log_l_away=rout[:,1]
        l_home=torch.exp(log_l_home)
        l_away=torch.exp(log_l_away)
        return l_home,l_away

def SyndyasmenoBuild(lstm_input_size:int,
                  static_input_size:int,
                  n_leagues:int,
                  **kwargs)->nn.Module:
    return SyndyasmenoMlpLstmModel(lstm_input_size,static_input_size,n_leagues,**kwargs)

def ypologismos_score(l_home:float,
                      l_away:float,
                      max_goals:int=7,rho:float=-0.13)->dict:
    goals_range=np.arange(0,max_goals+1)
       
    def poisson_pmf(k:np.ndarray,lam:float)->np.ndarray:
        return np.exp(-lam)*np.power(lam,k)/np.array([math.factorial(int(x)) for x in k])
    
    p_home=poisson_pmf(goals_range,l_home)
    p_away=poisson_pmf(goals_range,l_away)
    pin_score=np.outer(p_home,p_away)

    def tau(x, y, l_h, l_a, r):
        if x == 0 and y == 0:
            return 1 - l_h * l_a * r
        elif x == 0 and y == 1:
            return 1 + l_h * r
        elif x == 1 and y == 0:
            return 1 + l_a * r
        elif x == 1 and y == 1:
            return 1 - r
        else:
            return 1.0
        
    for i in range(min(2, max_goals + 1)):
        for j in range(min(2, max_goals + 1)):
            pin_score[i, j] *= tau(i, j, l_home, l_away, rho)
    pin_score = pin_score / pin_score.sum() 
    flat_idx=np.argmax(pin_score)
    pithanotero_home,pithanotero_away=np.unravel_index(flat_idx,pin_score.shape)
    pithanotero_prob=pin_score[pithanotero_home,pithanotero_away] 
    p_home_nikh = np.sum(np.tril(pin_score, k=-1))  
    p_isopalia  = np.sum(np.diag(pin_score))         
    p_away_nikh = np.sum(np.triu(pin_score, k=1))    
                                                              
    if abs(p_home_nikh - p_away_nikh) < 0.10 and p_isopalia > 0.25:
        predicted_winner = "D"
    elif p_home_nikh >= p_away_nikh:
        predicted_winner = "H"
    else:
        predicted_winner = "A"

    return {
        "pinakas_score":pin_score,
        "pithanotero_na_scorarei": (int(pithanotero_home), int(pithanotero_away)),
        "pithanotero_prob":float(pithanotero_prob),
        "p_home_nikh":float(p_home_nikh),
        "p_away_nikh":float(p_away_nikh),
        "p_isopalia":float(p_isopalia),
        "predicted_winner":predicted_winner,}

def sequences_to_tensor(sequences: dict, index_order) -> torch.Tensor:   
    stacked = np.stack([sequences[i] for i in index_order])
    return torch.tensor(stacked, dtype=torch.float32)

def modeltrain(normalized:dict,
                feature_columns:list,
                home_goals_column:str="home_goals",
                away_goals_column:str="away_goals",
                date_column:str="Date",
                train_cutoff_date:str=None,
                val_cutoff_date:str=None,
                model_kwargs:dict=None,
                device=device,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr=learning_rate,
                patience=patience)->dict:
    
    dataframe=normalized["match_df"].copy()
    dataframe[date_column]=pd.to_datetime(dataframe[date_column],errors="coerce")
    home_sequences=normalized["home_sequences"]
    away_sequences=normalized["away_sequences"]
    league_to_id=normalized["league_to_id"]   
    train_cutoff_date=pd.to_datetime(train_cutoff_date)
    val_cutoff_date = pd.to_datetime(val_cutoff_date) if val_cutoff_date is not None else None
    train_mask = dataframe[date_column] < train_cutoff_date
    if val_cutoff_date is not None:
        val_mask = (dataframe[date_column] >= train_cutoff_date) & (dataframe[date_column] < val_cutoff_date)
    else:
        val_mask = pd.Series(False, index=dataframe.index)
       
    def tensorbuild(mask:pd.Series)->tuple:
        idx=dataframe.index[mask]
        home_seq=sequences_to_tensor(home_sequences,idx)
        away_seq=sequences_to_tensor(away_sequences,idx)
        static=torch.tensor(dataframe.loc[idx,feature_columns].to_numpy(dtype=float),dtype=torch.float32)
        league_ids=torch.tensor(dataframe.loc[idx,"league_id"].to_numpy(dtype=int),dtype=torch.long)
        home_goals=torch.tensor(dataframe.loc[idx,home_goals_column].to_numpy(dtype=float),dtype=torch.float32)
        away_goals=torch.tensor(dataframe.loc[idx,away_goals_column].to_numpy(dtype=float),dtype=torch.float32)    
        return home_seq,away_seq,static,league_ids,home_goals,away_goals
        
    train_tensors = tensorbuild(train_mask)
    if train_tensors[0].shape[0] == 0:
        raise ValueError("No training rows, check cutoff date")
 
    train_dataset = torch.utils.data.TensorDataset(*train_tensors)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_tensors = tensorbuild(val_mask) if val_mask.sum() > 0 else None 
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = model_kwargs or {}
    lstm_input_size = train_tensors[0].shape[-1]
    static_input_size = len(feature_columns)
    n_leagues = len(league_to_id)
    model = SyndyasmenoBuild(lstm_input_size, static_input_size, n_leagues, **model_kwargs).to(device) 
    criterion = nn.PoissonNLLLoss(log_input=False, full=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}
 
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
 
        for home_seq_b, away_seq_b, static_b, league_b, home_goals_b, away_goals_b in train_loader:
            home_seq_b = home_seq_b.to(device)
            away_seq_b = away_seq_b.to(device)
            static_b = static_b.to(device)
            league_b = league_b.to(device)
            home_goals_b = home_goals_b.to(device)
            away_goals_b = away_goals_b.to(device) 
            optimizer.zero_grad()
            l_home, l_away = model(home_seq_b, away_seq_b, static_b, league_b)
            loss = criterion(l_home, home_goals_b) + criterion(l_away, away_goals_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1 
        train_loss = running_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)
 
        if val_tensors is not None:
            model.eval()
            with torch.no_grad():
                home_seq_v, away_seq_v, static_v, league_v, home_goals_v, away_goals_v = [
                    t.to(device) for t in val_tensors]
                l_home_v, l_away_v = model(home_seq_v, away_seq_v, static_v, league_v)
                val_loss = (criterion(l_home_v, home_goals_v)
                            + criterion(l_away_v, away_goals_v)).item()
            history["val_loss"].append(val_loss)
            print(f"Epoch {epoch} of {num_epochs}. Training loss: {train_loss:.4f}. Validation loss: {val_loss:.4f}.") 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch} after {patience} epochs without validation loss improvement.")
                    break
        else:
            print(f"Epoch {epoch} of {num_epochs}. Training loss: {train_loss:.4f}.")
 
    if best_state is not None:
        model.load_state_dict(best_state)
 
    return {
        "model": model,
        "history": history,
        "device": device,
        "feature_columns": feature_columns,
        "league_to_id": league_to_id,}

def formation_prediction(home_team:str, away_team:str, result:dict,draw_boost: float = 1.0)->str:
    home_goals, away_goals = result["pithanotero_na_scorarei"]
    p_home, p_isopalia, p_away = result["p_home_nikh"], result["p_isopalia"], result["p_away_nikh"]
    if p_home >= p_isopalia and p_home >= p_away:
        outcome_line = f"Most likely outcome: {home_team} win with probability {p_home:.1%}."
    elif p_away >= p_home and p_away >= p_isopalia:
        outcome_line = f"Most likely outcome: {away_team} win with probability {p_away:.1%}."
    else:
        outcome_line = f"Most likely outcome: draw with probability {p_isopalia:.1%}."

    score_line = (f"Most likely exact score: {home_team} {home_goals}-{away_goals} {away_team} "
                  f"with probability {result['pithanotero_prob']:.1%}.")

    return (f"{outcome_line}\n"
            f"{score_line}\n"
            f"{home_team} win probability: {p_home:.1%}.\n"
            f"Draw probability: {p_isopalia:.1%}.\n"
            f"{away_team} win probability: {p_away:.1%}.")
        
def match_prediction(model:nn.Module,
                     home_team:str,
                     away_team:str,
                     league_name:str,
                     home_static_values:dict,
                     away_static_values:dict,
                     home_raw_sequence:np.ndarray,
                     away_raw_sequence:np.ndarray,
                     feature_columns:list,
                     static_scaler,
                     sequence_scaler,
                     static_medians:pd.Series,
                     league_to_id:dict,
                     device:str=None,
                     max_goals:int=6,rho:float=-0.13)->dict:
    if league_name not in league_to_id:
        raise ValueError(f"League '{league_name}' is not present in the training league_to_id"
                         f" ({list(league_to_id.keys())}). The model cannot assign an"
                         f"embedding to a league it did not see during training.")
        
    static_row = pd.DataFrame([{**home_static_values, **away_static_values}])
    static_row = static_row.reindex(columns=feature_columns)
    missing_cols = static_row.columns[static_row.isna().any()]
    if len(missing_cols) > 0:
        static_row[missing_cols] = static_row[missing_cols].fillna(static_medians[missing_cols])
    static_scaled = static_scaler.transform(static_row)
    static_tensor = torch.tensor(static_scaled, dtype=torch.float32).to(device)
    
    def scale_sequence(raw_seq: np.ndarray) -> torch.Tensor:
        shape = raw_seq.shape
        flat_scaled = sequence_scaler.transform(pd.DataFrame(raw_seq.reshape(-1, shape[-1])))
        return torch.tensor(flat_scaled.reshape(1, *shape), dtype=torch.float32).to(device)
 
    home_seq_tensor = scale_sequence(home_raw_sequence)
    away_seq_tensor = scale_sequence(away_raw_sequence) 
    league_tensor = torch.tensor([league_to_id[league_name]], dtype=torch.long).to(device) 
    model.eval()
    with torch.no_grad():
        l_home, l_away = model(home_seq_tensor, away_seq_tensor, static_tensor, league_tensor) 
    result = ypologismos_score(l_home.item(), l_away.item(), max_goals=max_goals, rho=rho)
    result["l_home"] = l_home.item()
    result["l_away"] = l_away.item()
    result["summary"] = formation_prediction(home_team, away_team, result)
    return result

player_regression_tar=["Gls","Ast","Sh","SoT","Fls","Int","TklW"]
player_classification_tar=["Start","CrdY","CrdR","PKcon"]
player_columns_features=["Min","Gls","Ast","Sh","SoT","Fls","Fld","Int","TklW","CrdY","CrdR","PKcon","PKwon","Crs","Off"]

def rollingavgfeatures(dataframe:pd.DataFrame,
                       rolling_window:int=5,
                       player_col:str="player",
                       date_col:str="Date",
                       season_col:str="season")->pd.DataFrame:    
    dataframe=dataframe.copy()
    dataframe[date_col]=pd.to_datetime(dataframe[date_col],errors="coerce")
    
    for col in player_columns_features:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce").fillna(0.0)
        
    if "Start" in dataframe.columns:
        dataframe["Start"]=(dataframe["Start"].astype(str).str.strip().str.upper()=="Y").astype(float)
        
    dataframe=dataframe.sort_values([player_col,season_col,date_col])
    group=dataframe.groupby([player_col,season_col],group_keys=False)
    n=rolling_window
        
    for col in player_columns_features:
        if col not in dataframe.columns:
            continue
            
        dataframe[f"avg_{col}_last{n}"]=group[col].apply(lambda s: s.shift(1).rolling(window=n,min_periods=1).mean()  )
        dataframe[f"season_avg_{col}"]=group[col].apply(lambda s: s.shift(1).expanding(min_periods=1).mean()   )
            
    if "Start" in dataframe.columns:
        dataframe["start_rate_last5"]=group["Start"].apply(lambda s: s.shift(1).rolling(window=5,min_periods=1).mean()     )
    
    return dataframe

def build_elo_ratings(
    team_match_df:pd.DataFrame,
    config:dict=Config,
    initial_rating:float=1500.0,
    k_factor:float=20.0,
    season_regression:float = 0.75)->pd.DataFrame:
    c=config
    df=team_match_df.copy()
    df[c["date_col"]] = pd.to_datetime(
        df[c["date_col"]],
        errors="coerce")
    df = df.sort_values([c["league_col"], c["date_col"]]).reset_index(drop=True)
    ratings={}
    current_season={}
    elo_before=pd.Series(
        np.nan,
        index=df.index,
        dtype=float)                                                    
    row_lookup={}

    for idx, row in df.iterrows():
        key=(
            row[c["league_col"]],
            row[c["season_col"]],
            row[c["date_col"]],
            row[c["team_col"]],
            row[c["opponent_col"]])
        row_lookup[key] = idx
    processed_matches = set()

    for idx, row in df.iterrows():
        league = row[c["league_col"]]
        season = row[c["season_col"]]
        date = row[c["date_col"]]
        team = row[c["team_col"]]
        opponent = row[c["opponent_col"]]
        key_team = (league, team)
        key_opp = (league, opponent)
        match_key=(
            league,season,date,frozenset((team, opponent)))

        if match_key in processed_matches:
            continue
                                                        
        for key in (key_team, key_opp):
            if (
                key in current_season
                and current_season[key] != season):
                ratings[key] = (
                    season_regression * ratings[key]
                    + (1.0 - season_regression) * initial_rating)
            current_season[key] = season
        r_team = ratings.get(key_team, initial_rating)
        r_opp = ratings.get(key_opp, initial_rating)
        elo_before.loc[idx] = r_team
        reverse_key = (
            league,
            season,
            date,
            opponent,
            team)
        reverse_idx = row_lookup.get(reverse_key)

        if reverse_idx is not None:
            elo_before.loc[reverse_idx] = r_opp                                                            
        if row[c["goals_for_col"]] > row[c["goals_against_col"]]:
            actual = 1.0
        elif row[c["goals_for_col"]] == row[c["goals_against_col"]]:
            actual = 0.5
        else:
            actual = 0.0                           
        expected = 1.0 / (
            1.0 + 10.0 ** ((r_opp - r_team) / 400.0))
        delta = k_factor * (actual - expected)                                         
        ratings[key_team] = r_team + delta
        ratings[key_opp] = r_opp - delta
        processed_matches.add(match_key)
    df["elo_before"] = elo_before
    return df

def backtest_team_model(trained_team: dict, normalized_team: dict,
                         match_dataset: pd.DataFrame, val_cutoff_date: str,
                         end_date: str = None, date_column: str = "Date",
                         rho: float = -0.13,draw_boost: float = 1.0) -> dict:

    df = match_dataset.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    val_cutoff = pd.to_datetime(val_cutoff_date)
    mask = df[date_column] >= val_cutoff
    if end_date is not None:
        mask &= df[date_column] < pd.to_datetime(end_date)
    all_team_sequences = normalized_team["all_team_sequences"]
    feature_cols = trained_team["feature_columns"]
    n_total, n_correct_winner, n_correct_score, n_skipped = 0, 0, 0, 0
    outcome_counts = {"H": 0, "D": 0, "A": 0}
    league_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    pred_outcomes = Counter()

    for _, row in df[mask].iterrows():
        home_team, away_team, league = row["home_team"], row["away_team"], row["league"]
        date = row[date_column]

        home_key, away_key = (home_team, date), (away_team, date)
        if home_key not in all_team_sequences or away_key not in all_team_sequences:
            n_skipped += 1
            continue

        home_static = {c: row[c] for c in feature_cols if c.endswith("_home") and c in row.index}
        away_static = {c: row[c] for c in feature_cols if c.endswith("_away") and c in row.index}
        res = match_prediction(
            model=trained_team["model"], home_team=home_team, away_team=away_team, league_name=league,
            home_static_values=home_static, away_static_values=away_static,
            home_raw_sequence=all_team_sequences[home_key], away_raw_sequence=all_team_sequences[away_key],
            feature_columns=feature_cols, static_scaler=normalized_team["static_scaler"],
            sequence_scaler=normalized_team["sequence_scaler"], static_medians=normalized_team["static_medians"],
            league_to_id=trained_team["league_to_id"], device=trained_team["device"], rho=rho)
        actual_home, actual_away = row["home_goals"], row["away_goals"]
        actual_outcome = "H" if actual_home > actual_away else ("A" if actual_home < actual_away else "D")
        outcome_counts[actual_outcome] += 1
        p_h, p_d, p_a = res["p_home_nikh"], res["p_isopalia"], res["p_away_nikh"]
        p_d_boosted = p_d * draw_boost
        pred_outcome = "H" if (p_h >= p_d_boosted and p_h >= p_a) else ("A" if p_a >= p_d_boosted else "D")
        pred_outcomes[pred_outcome] += 1
        n_total += 1
        if pred_outcome == actual_outcome:
            n_correct_winner += 1
        if res["pithanotero_na_scorarei"] == (int(actual_home), int(actual_away)):
            n_correct_score += 1
        ls = league_stats[league]
        ls["total"] += 1
        if pred_outcome == actual_outcome:
            ls["correct"] += 1
    if n_total == 0:
        print("No matches found for backtesting. Check the date range and sequences.")
        return {}
    winner_accuracy = n_correct_winner / n_total
    score_accuracy = n_correct_score / n_total
    majority_class = max(outcome_counts, key=outcome_counts.get)
    baseline_accuracy = outcome_counts[majority_class] / n_total  
    print(f"Backtest matches: {n_total}. Skipped matches: {n_skipped}.")
    print(f"Actual outcome distribution: H={outcome_counts['H']}, D={outcome_counts['D']}, A={outcome_counts['A']}.")
    print(f"Predicted outcome distribution: H={pred_outcomes['H']}, D={pred_outcomes['D']}, A={pred_outcomes['A']}.")
    print(f"Majority class baseline using {majority_class}: {baseline_accuracy:.1%}.")
    print(f"Winner accuracy: {winner_accuracy:.1%}.")
    print(f"Exact score accuracy: {score_accuracy:.1%}.")
 
    for lg, s in league_stats.items():
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"{lg} accuracy: {acc:.1%} on {s['total']} matches.")

    return {
        "n_total": n_total, "winner_accuracy": winner_accuracy,
        "score_accuracy": score_accuracy, "baseline_accuracy": baseline_accuracy,
        "outcome_counts": outcome_counts,}

def wfeval(match_dataset, team_feature_columns, home_sequences,
                             away_sequences, all_sequences, fold_dates, model_kwargs=None,
                             lr=0.0001, patience=20, num_epochs=150):

    fold_results = []
    for i, (train_cutoff, val_cutoff, test_end) in enumerate(fold_dates, 1):
        set_seed(42 + i)
        print(f"Fold {i} of {len(fold_dates)}")
        print(f"Training before {train_cutoff}. Validation from {train_cutoff} to {val_cutoff}. Test from {val_cutoff} to {test_end}.")
        normalized_fold = normalize_features(
            match_df=match_dataset, home_sequences=home_sequences, away_sequences=away_sequences,
            feature_columns=team_feature_columns, train_cutoff_date=train_cutoff,)
        normalized_fold["feature_columns"] = team_feature_columns
        normalized_fold["all_team_sequences"] = all_sequences
        trained_fold = modeltrain(
            normalized=normalized_fold, feature_columns=team_feature_columns,
            train_cutoff_date=train_cutoff, val_cutoff_date=val_cutoff,
            model_kwargs=model_kwargs, lr=lr, patience=patience, num_epochs=num_epochs,)
        trained_fold["device"] = device
        result = backtest_team_model(
        trained_fold,
        normalized_fold,
        match_dataset,
        val_cutoff_date=val_cutoff,
        end_date=test_end,
        draw_boost=1.0)      
        result["fold"] = i
        fold_results.append(result)   
    accuracies = [r["winner_accuracy"] for r in fold_results if r]
    baselines = [r["baseline_accuracy"] for r in fold_results if r]  
    print(f"Walk-forward summary for {len(accuracies)} folds.")
    print(f"Mean winner accuracy: {np.mean(accuracies):.1%}. Standard deviation: {np.std(accuracies):.1%}.")
    print(f"Mean baseline accuracy: {np.mean(baselines):.1%}.")   
    return fold_results

def build_player_match_dataset(player_dataframe: pd.DataFrame,
                               team_features_dataframe: pd.DataFrame,
                               player_date_column: str = "Date",
                               opponent_column: str = "Opponent",
                               team_column: str = "Squad",
                               team_date_column: str = "Date",
                               opponent_feature_columns: list = None) -> pd.DataFrame:
    
    if opponent_feature_columns is None:
        opponent_feature_columns=[
            col for col in team_features_dataframe.columns
            if col.startswith("avg_goals_against") or col.startswith("season_avg_goals_against") or col.startswith("win_rate") or 
            col.startswith("season_win_rate")]

        if not opponent_feature_columns:
            raise ValueError(
                "No defensive feature columns were found (avg_goals_against*/win_rate*)"
            "in team_features_dataset; provide opponent_feature_columns. ")
    
    opponent_columns_needed=[team_column,team_date_column]+opponent_feature_columns
    opponent_columns_needed=[c for c in opponent_columns_needed if c in team_features_dataframe.columns  ]
    opponent_dataframe=team_features_dataframe[opponent_columns_needed].copy()
    opponent_dataframe[team_date_column]=pd.to_datetime(opponent_dataframe[team_date_column],errors="coerce"  )
    opponent_dataframe=opponent_dataframe.add_prefix("opponent_")    
    merged_dataframe=player_dataframe.copy()
    merged_dataframe[player_date_column]=pd.to_datetime(merged_dataframe[player_date_column],errors="coerce")
    merged_dataframe=merged_dataframe.merge(opponent_dataframe,left_on=[player_date_column,opponent_column],right_on=[f"opponent_{team_date_column}", f"opponent_{team_column}"],how="left")
    n_missing=merged_dataframe[f"opponent_{team_date_column}"].isna().sum()
    if n_missing>0:
        print(f"Warning: {n_missing} player rows could not be matched to opponent team features and will require imputation before training.")
 
    return merged_dataframe

class playerpredictionmodel(nn.Module):
    def __init__(self,
                 lstm_input_size:int,
                 static_input_size:int,
                 n_positions:int,
                 n_leagues:int,
                 n_regression_tar:int,
                 n_classification_tar:int,
                 lstm_hidden_size:int=64,
                 lstm_num_layers:int=1,
                 lstm_dropout:float=0.0,
                 mlp_hidden_sizes:tuple=(64,32),
                 mlp_output_size:int=32,
                 mlp_dropout:float=0.2,
                 position_embedding_dimens:int=8,
                 league_embedding_dimens:int=8,
                 head_hidden_size:int=32,
                 max_mins:float=120.0):
        super().__init__()
        self.form_encoder=LSTMEncoder(lstm_input_size,lstm_hidden_size,lstm_num_layers,lstm_dropout)
        self.static_encoder=MLPEncoder(static_input_size,mlp_hidden_sizes, mlp_output_size, mlp_dropout)
        self.position_embedding=nn.Embedding(n_positions,position_embedding_dimens)
        self.league_embedding=nn.Embedding(n_leagues,league_embedding_dimens)
        combined_size = lstm_hidden_size + mlp_output_size + position_embedding_dimens + league_embedding_dimens       
        self.trunk=nn.Sequential(nn.Linear(combined_size,head_hidden_size),nn.ReLU(),nn.Dropout(mlp_dropout))
        self.regression_head=nn.Linear(head_hidden_size,n_regression_tar)
        self.minutes_head=nn.Linear(head_hidden_size,1)
        self.classification_head=nn.Linear(head_hidden_size,n_classification_tar)
        self.max_minutes=max_mins
       
    def forward(self,
                player_seq:torch.Tensor,
                static_features:torch.Tensor,
                position_ids:torch.Tensor,
                league_ids:torch.Tensor)->tuple:
        form_embedding=self.form_encoder(player_seq)
        static_embedding=self.static_encoder(static_features)
        position_emb=self.position_embedding(position_ids)
        league_emb=self.league_embedding(league_ids)
        combined=torch.cat([form_embedding,static_embedding,position_emb,league_emb],dim=1)
        trunk_out=self.trunk(combined)        
        count_predictions = torch.exp(self.regression_head(trunk_out))
        minutes_prediction = torch.sigmoid(self.minutes_head(trunk_out)).squeeze(-1) * self.max_minutes
        classification_logits = self.classification_head(trunk_out)
 
        return count_predictions, minutes_prediction, classification_logits
    
def playermodelbuild(lstm_input_size:int,
                     static_input_size:int,
                     n_positions:int,
                     n_leagues:int,
                     n_regression_tars:int=None,
                     n_classification_tars:int=None,
                     **kwargs)->nn.Module:
    if n_regression_tars is None:
        n_regression_tars=len(player_regression_tar)
    if n_classification_tars is None:
        n_classification_tars=len(player_classification_tar)
        
    return playerpredictionmodel(lstm_input_size,
                                 static_input_size,n_positions,
                                 n_leagues,
                                 n_regression_tars,
                                 n_classification_tars,
                                 **kwargs)

def normalize_player_features(player_dataframe:pd.DataFrame,
                              player_feature_columns:list,
                              train_cutoff_date:str,
                              date_column:str="Date",
                              position_column:str="position",
                              league_column:str="league")->dict:
    dataframe=player_dataframe.copy()
    dataframe[date_column]=pd.to_datetime(dataframe[date_column],errors="coerce")
    train_cutoff=pd.to_datetime(train_cutoff_date)
    train_mask=dataframe[date_column]<train_cutoff   
    if train_mask.sum()==0:
        raise ValueError(f"No rows were found before train_cutoff date={train_cutoff_date}")    
    non_numeric = [
    col for col in player_feature_columns
    if not pd.api.types.is_numeric_dtype(dataframe[col])]
    if non_numeric:
        raise ValueError(
        f"The following columns in player_feature_columns are not numeric "
        f"and would fail in StandardScaler: {non_numeric}. "
        f"Possible cause: 'opponent_' prefix filtering also selected non-numeric "
        f"columns (e.g. opponent_Squad, opponent_Date).")       
    player_medians=dataframe.loc[train_mask,player_feature_columns].median()
    dataframe[player_feature_columns]=dataframe[player_feature_columns].fillna(player_medians)
    player_scaler=StandardScaler()
    player_scaler.fit(dataframe.loc[train_mask,player_feature_columns])
    dataframe[player_feature_columns]=player_scaler.transform(dataframe[player_feature_columns])    
    sorted_positions=sorted(dataframe[position_column].dropna().unique().tolist())
    position_to_id={pos: i for i, pos in enumerate(sorted_positions)}
    dataframe["position_id"]=dataframe[position_column].map(position_to_id).fillna(0).astype(int)    
    sorted_leagues=sorted(dataframe[league_column].dropna().unique().tolist())
    league_to_id={name: i for i, name in enumerate(sorted_leagues)}
    dataframe["league_id"]=dataframe[league_column].map(league_to_id).fillna(0).astype(int)   
    print(f"Position mapping: {position_to_id}")
    print(f"League mapping: {league_to_id}")
    print(f"Training rows: {train_mask.sum()}. Total rows: {len(dataframe)}.")
    return{
        "player_dataframe":dataframe,
        "player_scaler":player_scaler,
        "player_medians":player_medians,
        "position_to_id":position_to_id,
        "league_to_id":league_to_id}

def build_player_sequences(player_dataframe:pd.DataFrame,
                           sequence_features:list,
                           n_last_matches:int=5,
                           player_column:str="player",
                           date_column:str="Date",
                           season_column:str="season")->dict:
    dataframe=player_dataframe.copy()
    dataframe[date_column]=pd.to_datetime(dataframe[date_column],errors="coerce")
    
    for col in sequence_features:
        if col in dataframe.columns:
            dataframe[col]=pd.to_numeric(dataframe[col],errors="coerce").fillna(0.0)    
    dataframe=dataframe.sort_values([player_column,season_column,date_column])
    sequences={}
      
    for (player_name, season), group in dataframe.groupby([player_column, season_column]):
        group=group.reset_index(drop=True)
        available_columns=[c for c in sequence_features if c in group.columns]
        values=group[available_columns].to_numpy(dtype=float)
        
        if len(available_columns)<len(sequence_features):
            pad_columns=len(sequence_features)-len(available_columns)
            values = np.hstack([values, np.zeros((len(group), pad_columns))])
            
        for i in range(len(group)):
            history=values[max(0,i-n_last_matches):i]
            
            if len(history)==0:
                history = np.zeros((n_last_matches, len(sequence_features)))
            elif len(history)<n_last_matches:
                pad_len=n_last_matches-len(history)
                pad=np.zeros((pad_len,len(sequence_features)))
                history=np.vstack([pad,history])
            match_date=group.loc[i,date_column]
            sequences[(player_name,match_date)]=history
    print(f"Player sequences created: {len(sequences)}.")
    return sequences
        
def player_train_model(normalized_player:dict,
                       player_feature_columns:list,
                       player_sequences:dict,
                       date_column:str="Date",
                       player_column:str="player",
                       train_cutoff_date:str=None,
                       val_cutoff_date:str=None,
                       model_kwargs:dict=None,
                       device=device,
                       num_epochs: int=num_epochs,
                       batch_size:int=batch_size,
                       lr:float=learning_rate,
                       patience:int=patience)->dict:
    dataframe=normalized_player["player_dataframe"].copy()
    dataframe[date_column]=pd.to_datetime(dataframe[date_column],errors="coerce")
    position_to_id=normalized_player["position_to_id"]
    league_to_id=normalized_player["league_to_id"]
    train_cutoff=pd.to_datetime(train_cutoff_date)
    val_cutoff=pd.to_datetime(val_cutoff_date) if val_cutoff_date else None
    train_mask=dataframe[date_column]<train_cutoff
      
    if val_cutoff:
        val_mask=(dataframe[date_column]>=train_cutoff)&(dataframe[date_column]<val_cutoff)
    else:
        val_mask=pd.Series(False,index=dataframe.index)
                
    def player_tensor(mask:pd.Series)-> tuple:
        idx=dataframe.index[mask]
        sequence_list=[]
        for i in idx:
            player_name=dataframe.loc[i,player_column]
            match_date=dataframe.loc[i,date_column]
            key=(player_name,match_date)
            if key in player_sequences:
                sequence_list.append(player_sequences[key])
            else:
                sample=next(iter(player_sequences.values()))
                sequence_list.append(np.zeros_like(sample))        
        sequence_tensor=torch.tensor(np.stack(sequence_list),dtype=torch.float32)        
        static=torch.tensor(dataframe.loc[idx,player_feature_columns].to_numpy(dtype=float),dtype=torch.float32)
        position_ids=torch.tensor(dataframe.loc[idx,"position_id"].to_numpy(dtype=int),dtype=torch.long)
        league_ids=torch.tensor(dataframe.loc[idx,"league_id"].to_numpy(dtype=int),dtype=torch.long)
        reg_tars=torch.tensor(dataframe.loc[idx,player_regression_tar].to_numpy(dtype=float),dtype=torch.float32)
        min_tars=torch.tensor(dataframe.loc[idx,"Min"].to_numpy(dtype=float),dtype=torch.float32)
        classif_tars = torch.tensor(dataframe.loc[idx, player_classification_tar].to_numpy(dtype=float), dtype=torch.float32).clamp(0, 1)
    
        return sequence_tensor,static,position_ids,league_ids,reg_tars,min_tars,classif_tars
    
    train_tensors=player_tensor(train_mask)
    
    if train_tensors[0].shape[0]==0:
        raise ValueError("No training rows were found")
        
    train_dataset=torch.utils.data.TensorDataset(*train_tensors)
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_tensors=player_tensor(val_mask) if val_mask.sum() >0 else None    
    model_kwargs=model_kwargs or{}
    lstm_input_size=train_tensors[0].shape[-1]
    static_input_size=len(player_feature_columns)
    n_positions=len(position_to_id)
    n_leagues=len(league_to_id)   
    model=playermodelbuild(
        lstm_input_size,
        static_input_size,
        n_positions,
        n_leagues,
        **model_kwargs).to(device)    
    poisson_loss_fn = nn.PoissonNLLLoss(log_input=False, full=False)
    bce_loss_fn     = nn.BCEWithLogitsLoss()
    mse_loss_fn     = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss=float("inf")
    best_state=None
    epochs_no_improvement=0
    history={"train_loss":[],"val_loss":[]}
       
    for epoch in range(1,num_epochs+1):
        model.train()
        running_loss,n_batches=0.0,0
        
        for sequence_b,static_b,position_b,league_b,reg_tars_b,min_tars_b,classif_tars_b in train_loader:
            sequence_b=sequence_b.to(device)
            static_b=static_b.to(device)
            position_b=position_b.to(device)
            league_b=league_b.to(device)
            reg_tars_b=reg_tars_b.to(device)
            min_tars_b=min_tars_b.to(device)
            classif_tars_b=classif_tars_b.to(device)
            
            optimizer.zero_grad()
            count_preds, min_preds, cls_logits = model(
                sequence_b, static_b, position_b, league_b)
            loss_poisson=poisson_loss_fn(count_preds,reg_tars_b)
            loss_bce=bce_loss_fn(cls_logits,classif_tars_b)
            loss_mse=mse_loss_fn(min_preds / model.max_minutes, min_tars_b / model.max_minutes)
            total_loss=loss_poisson+loss_bce+loss_mse
            total_loss.backward()
            optimizer.step()
            running_loss+=total_loss.item()
            n_batches+=1
        train_loss = running_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)
        
        if val_tensors is not None:
            model.eval()
            with torch.no_grad():
                sequence_v,static_v,position_v,league_v,reg_tars_v,min_tars_v,classif_tars_v=[
                    t.to(device) for t in val_tensors]
                count_v,min_pred_v,classif_tars_v_logits=model(sequence_v,static_v,position_v,league_v)
                
                vals_loss = (poisson_loss_fn(count_v, reg_tars_v)+ bce_loss_fn(classif_tars_v_logits, classif_tars_v)
                             + mse_loss_fn(min_pred_v / model.max_minutes, min_tars_v / model.max_minutes)).item()
            history["val_loss"].append(vals_loss)
            print(f"Epoch {epoch} of {num_epochs}. Training loss: {train_loss:.4f}. Validation loss: {vals_loss:.4f}.")
    
            if vals_loss<best_val_loss:
                best_val_loss=vals_loss
                best_state={k:v.clone() for k, v in model.state_dict().items()}
                epochs_no_improvement=0
            else:
                epochs_no_improvement+=1
                if epochs_no_improvement>=patience:
                    print(f"Early stopping at epoch {epoch} after {patience} epochs without validation loss improvement.")
                    break
        else:
            print(f"Epoch {epoch} of {num_epochs}. Training loss: {train_loss:.4f}.")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return{
        "model": model,
        "history": history,
        "player_feature_columns": player_feature_columns,
        "device":device,
        "position_to_id":position_to_id,
        "league_to_id": league_to_id}

def playerprediction(model:nn.Module,
                     player_name:str,
                     position:str,
                     league_name:str,
                     player_static_values:dict,
                     player_raw_sequence:np.ndarray,
                     player_feature_columns:list,
                     player_scaler,
                     sequence_scaler,
                     player_medians:pd.Series,
                     position_to_id:dict,
                     league_to_id:dict,
                     opponent_team:str,
                     device=device,
                     max_goals:int=7)->dict:
    if league_name not in league_to_id:
        raise ValueError(f"League '{league_name}' not in training"
                         f"({list(league_to_id.keys())}")
    if position not in position_to_id:
        raise ValueError(f"Position {position} unknown")
        
    static_row=pd.DataFrame([player_static_values])
    static_row=static_row.reindex(columns=player_feature_columns)
    missing=static_row.columns[static_row.isna().any()]
    
    if len(missing)>0:
        static_row[missing]=static_row[missing].fillna(player_medians[missing])
    static_scaled=player_scaler.transform(static_row)
    static_tensor=torch.tensor(static_scaled, dtype=torch.float32).to(device)    
    shape=player_raw_sequence.shape
    flat_scaled=sequence_scaler.transform(player_raw_sequence.reshape(-1,shape[-1]) )
    seq_tensor = torch.tensor(flat_scaled.reshape(1, *shape),dtype=torch.float32).to(device)    
    pos_id=position_to_id.get(position,0)
    pos_tensor=torch.tensor([pos_id], dtype=torch.long).to(device)
    league_tensor=torch.tensor([league_to_id[league_name]], dtype=torch.long).to(device)    
    model.eval()
    with torch.no_grad():
        count_preds, min_pred, cls_logits = model(seq_tensor, static_tensor, pos_tensor, league_tensor   )
    counts = count_preds[0].cpu().numpy()
    cls_probs = torch.sigmoid(cls_logits[0]).cpu().numpy()
    minutes = float(min_pred[0].cpu())

    result={"player":player_name,
            "opponent_team":opponent_team,
            "league": league_name,
            "expected_goals":round(float(counts[0]),2),
            "expected_assists": round(float(counts[1]),2),
            "expected_shots": round(float(counts[2]),2),
            "expected_shots_on_target":round(float(counts[3]),2),
            "expected_fouls":round(float(counts[4]),2),
            "expected_interceptions":round(float(counts[5]),2),
            "expected_tackles":round(float(counts[6]),2),
            "expected_minutes_to_play":round(minutes,1),
            "probability_to_be_in_starting_11":round(float(cls_probs[0]),3),
            "yellow_card_probability":round(float(cls_probs[1]),3),
            "red_card_probability":round(float(cls_probs[2]),3),
            "probability_to_concede_penalty":round(float(cls_probs[3]),3)            }
    
    starter_str = "YES" if result["probability_to_be_in_starting_11"] >= 0.5 else "NO"
    result["summary"]=(f"Player: {player_name} ({position})\n"
                       f"Opponent: {opponent_team}\n"
                       f"Expected starter: {starter_str}\n"
                       f"Starting probability: {result['probability_to_be_in_starting_11']:.1%}\n"
                       f"Expected minutes: {result['expected_minutes_to_play']}\n"
                       f"Expected goals: {result['expected_goals']}\n"
                       f"Expected assists: {result['expected_assists']}\n"
                       f"Expected shots: {result['expected_shots']}\n"
                       f"Expected shots on target: {result['expected_shots_on_target']}\n"
                       f"Expected fouls: {result['expected_fouls']}\n"
                       f"Expected interceptions: {result['expected_interceptions']}\n"
                       f"Expected tackles won: {result['expected_tackles']}\n"
                       f"Yellow card probability: {result['yellow_card_probability']}\n"
                       f"Red card probability: {result['red_card_probability']}\n"
                       f"Penalty conceded probability: {result['probability_to_concede_penalty']}")
    return result
      
teams_league={
    "Scotch":["Celtic","Rangers","Heart of Midlothian","Dundee United","Motherwell","Ross County","Livingston","Hibernian","St Mirren",
              "Aberdeen","St Johnstone","Dundee","Kilmarnock","Falkirk"],
    "Greek":["Olympiacos","PAOK","Aris","Panathinaikos","AEK Athens","PAS Giannina","Ionikos","OFI Crete","Asteras Tripoli","Volos N.F.C.",
             "Panetolikos","Atromitos","PAS Lamia","Apollon Smyrnis","Levadiakos","Panserraikos","AE Kifisia","Kallithea","Larissa"],
    "English":["Manchester City","Liverpool","Chelsea","Tottenham","Arsenal","Manchester Utd","West Ham","Leicester City","Brighton","Wolves",
               "Newcastle","Crystal Palace","Brentford","Aston Villa","Southampton","Everton","Leeds United","Fulham","Bournemouth","Nottingham"],
    "Italian":["Milan","Inter","Napoli","Juventus","Lazio","Roma","Fiorentina","Atalanta","Hellas Verona","Torino","Sassuolo","Udinese",
               "Bologna","Empoli","Sampdoria","Spezia","Salernitana","Monza","Lecce","Spezia","Genoa","Cagliari","Como","Parma"]}

def pick_from_list(options: list, prompt: str) -> str:
    print(prompt)
    for i,option in enumerate(options, 1):
        print(f"Option {i}: {option}")
    while True:
        try:
            choice = int(input(" Selection:"))
            if 1 <= choice <= len(options):
                return options[choice-1]
            print(f"Enter a number from 1 to {len(options)}.")
        except ValueError:
            print("Enter a number.")

def menu_match_prediction(trained_team: dict,
                          trained_player: dict,
                          normalized_team: dict,
                          normalized_player: dict,
                          team_features_df,
                          player_df,
                          player_sequences: dict) -> None:
    league = pick_from_list(list(teams_league.keys()), "Select a league:")
    teams = get_current_season_teams(team_features_df, league)
    print("Home team selection")
    home_team = pick_from_list(teams, "Select the home team:")
    away_options = [t for t in teams if t != home_team]
    print("Away team selection")
    away_team = pick_from_list(away_options, "Select the away team:")
    print(f"Match: {home_team} vs {away_team} in {league}")
    print("Match prediction")
    
    def get_latest_team_features(team: str, side: str) -> dict:
        team_rows = team_features_df[team_features_df["Squad"] == team]
        if team_rows.empty:
            return {}
        latest = team_rows.sort_values("Date").iloc[-1]
        feature_cols = normalized_team["feature_columns"]
        result = {}
        for col in feature_cols:
            if side == "home" and not col.endswith("_home"):
                continue
            if side == "away" and not col.endswith("_away"):
                continue
                                                                                                            
            suffix = "_home" if side == "home" else "_away"
            raw_col = col[: -len(suffix)]                               
            if raw_col in latest.index:
                result[col] = latest[raw_col]
            else:                                                                                 
                result[col] = float("nan")
        return result

    home_static = get_latest_team_features(home_team, "home")   
    away_static = get_latest_team_features(away_team, "away") 
    if "elo_diff" in normalized_team["feature_columns"]:
        home_elo_rows = team_features_df[
        (team_features_df["Squad"] == home_team) &(team_features_df["league"] == league)]
        away_elo_rows = team_features_df[(team_features_df["Squad"] == away_team) &(team_features_df["league"] == league)]
        if not home_elo_rows.empty and not away_elo_rows.empty:
            home_elo = home_elo_rows.sort_values("Date").iloc[-1]["elo_before"]
            away_elo = away_elo_rows.sort_values("Date").iloc[-1]["elo_before"]
            elo_diff_value = float(home_elo - away_elo)
            home_static["elo_diff"] = elo_diff_value
        
    def get_latest_sequence(all_team_sequences: dict, team: str) -> np.ndarray:
        team_keys = [(t, d) for (t, d) in all_team_sequences.keys() if t == team]
        if not team_keys:
            sample = next(iter(all_team_sequences.values()))
            return np.zeros_like(sample)
        latest_key = max(team_keys, key=lambda x: x[1])
        return all_team_sequences[latest_key]
    all_team_sequences = normalized_team.get("all_team_sequences", {})
    home_seq = get_latest_sequence(all_team_sequences, home_team)
    away_seq = get_latest_sequence(all_team_sequences, away_team)    
    match_result = match_prediction(
        model=trained_team["model"],
        home_team=home_team,
        away_team=away_team,
        league_name=league,
        home_static_values=home_static,
        away_static_values=away_static,
        home_raw_sequence=home_seq,
        away_raw_sequence=away_seq,
        feature_columns=trained_team["feature_columns"],
        static_scaler=normalized_team["static_scaler"],
        sequence_scaler=normalized_team["sequence_scaler"],
        static_medians=normalized_team["static_medians"],
        league_to_id=trained_team["league_to_id"],
        device=trained_team["device"],)
    print(match_result["summary"])    
    show_players = input("\n  Do you want player predictions? (yes/no): ").strip().lower()
    if show_players not in ("ναι", "n", "yes", "y", "ν"):
        return
    
    for team, opponent in [(home_team, away_team), (away_team, home_team)]:
        
        print(f"Players for {team}")

        league_players = player_df[player_df["league"] == league]

        latest_season = sorted(league_players["season"].dropna().unique())[-1]

        team_players = (league_players[(league_players["season"] == latest_season) &(league_players["Squad"] == team)]["player"].dropna().unique().tolist())
        if not team_players:
            print(f"No players found for {team}.")
            continue
        
        print("Option 1: All players")
        print("Option 2: Specific player")
        player_choice = input("  Selection: ").strip()

        if player_choice == "2":
            selected_player = pick_from_list(sorted(team_players), "Select a player:")
            players_to_predict = [selected_player]
        else:
            players_to_predict = team_players

        for pname in players_to_predict:                                                     
            p_rows = player_df[(player_df["player"] == pname) &(player_df["Squad"] == team) &(player_df["league"] == league) &(player_df["season"] == latest_season)]
            if p_rows.empty:
                continue
            latest_p = p_rows.sort_values("Date").iloc[-1]

            p_static = {
                col: latest_p[col]
                for col in normalized_player["player_feature_columns"]
                if col in latest_p.index}
            opponent_rows = team_features_df[
                (team_features_df["Squad"] == opponent) &
                (team_features_df["league"] == league)]

            if not opponent_rows.empty:
                latest_opponent = opponent_rows.sort_values("Date").iloc[-1]

                for col in normalized_player["player_feature_columns"]:
                    if col.startswith("opponent_"):
                        raw_col = col[len("opponent_"):]
            
                        if raw_col in latest_opponent.index:
                            p_static[col] = latest_opponent[raw_col]
            p_keys = [(n, d) for (n, d) in player_sequences.keys() if n == pname]
            if p_keys:
                latest_p_key = max(p_keys, key=lambda x: x[1])
                p_seq=player_sequences[latest_p_key]
            else:
                sample=next(iter(player_sequences.values()))
                p_seq=np.zeros_like(sample)

            pred=playerprediction(
                model=trained_player["model"],
                player_name=pname,
                position=str(latest_p.get("position", "")),
                league_name=league,
                player_static_values=p_static,
                player_raw_sequence=p_seq,
                player_feature_columns=trained_player["player_feature_columns"],
                player_scaler=normalized_player["player_scaler"],
                sequence_scaler=normalized_player["sequence_scaler"],
                player_medians=normalized_player["player_medians"],
                position_to_id=trained_player["position_to_id"],
                league_to_id=trained_player["league_to_id"],
                opponent_team=opponent,
                device=trained_player["device"],)
            print(pred["summary"])

def get_current_season_teams(team_features_df: pd.DataFrame,
                              league_name: str,
                              league_column: str = "league",
                              team_column: str = "Squad",
                              season_column: str = "season") -> list:
    
    league_rows = team_features_df[team_features_df[league_column] == league_name]
    if league_rows.empty:
        return []
    latest_season = sorted(league_rows[season_column].dropna().unique())[-1]
    season_rows = league_rows[league_rows[season_column] == latest_season]
    return sorted(season_rows[team_column].dropna().unique().tolist())

def simulate_league(trained_team: dict, normalized_team: dict,
                     team_features_df, league: str, n_simulations: int = 1000,n_relegated: int = 3):
    teams = get_current_season_teams(team_features_df, league)
    all_team_sequences = normalized_team.get("all_team_sequences", {})

    def get_latest(team, side):
        rows = team_features_df[team_features_df["Squad"] == team]
        if rows.empty:
            return None, None
        latest = rows.sort_values("Date").iloc[-1]
        static = {}
        suffix = "_home" if side == "home" else "_away"
        for col in trained_team["feature_columns"]:
            if side == "home" and not col.endswith("_home"):
                continue
            if side == "away" and not col.endswith("_away"):
                continue
            raw_col = col[: -len(suffix)]                               
            if raw_col in latest.index:
                static[col] = latest[raw_col]
            else:
                static[col] = float("nan")
        keys = [(t, d) for (t, d) in all_team_sequences.keys() if t == team]
        seq = all_team_sequences[max(keys, key=lambda x: x[1])] if keys else None
        return static, seq

    print(f"Calculating expected goal intensities for all {league} team pairings.")
    lambda_cache = {}
    for home in teams:
        home_static, home_seq = get_latest(home,"home")
        if home_static is None:
            continue
        for away in teams:
            if home == away:
                continue
            away_static, away_seq = get_latest(away,"away")
            if away_static is None:
                continue
            if "elo_diff" in normalized_team["feature_columns"]:
                home_elo_rows = team_features_df[(team_features_df["Squad"] == home) &(team_features_df["league"] == league)]
                away_elo_rows = team_features_df[(team_features_df["Squad"] == away) &(team_features_df["league"] == league)]

                if not home_elo_rows.empty and not away_elo_rows.empty:
                    home_elo = home_elo_rows.sort_values("Date").iloc[-1]["elo_before"]
                    away_elo = away_elo_rows.sort_values("Date").iloc[-1]["elo_before"]
                    home_static["elo_diff"] = float(home_elo - away_elo)
            res = match_prediction(
                model=trained_team["model"], home_team=home, away_team=away, league_name=league,
                home_static_values=home_static, away_static_values=away_static,
                home_raw_sequence=home_seq, away_raw_sequence=away_seq,
                feature_columns=trained_team["feature_columns"],
                static_scaler=normalized_team["static_scaler"],
                sequence_scaler=normalized_team["sequence_scaler"],
                static_medians=normalized_team["static_medians"],
                league_to_id=trained_team["league_to_id"], device=trained_team["device"],)
            lambda_cache[(home, away)] = (res["l_home"], res["l_away"])

    print(f"Running {n_simulations} simulations.")
    points_matrix = np.zeros((n_simulations, len(teams)))
    for sim in range(n_simulations):
        points = {team: 0 for team in teams}
        for (home, away), (l_home, l_away) in lambda_cache.items():
            h_goals, a_goals = np.random.poisson(l_home), np.random.poisson(l_away)
            if h_goals > a_goals:
                points[home] += 3
            elif h_goals == a_goals:
                points[home] += 1
                points[away] += 1
            else:
                points[away] += 3
        for k, team in enumerate(teams):
            points_matrix[sim, k] = points[team]

    avg_points = points_matrix.mean(axis=0)
    std_points = points_matrix.std(axis=0)
    title_prob = (points_matrix.argmax(axis=1)[:, None] == np.arange(len(teams))).mean(axis=0)
    ranking = sorted(zip(teams, avg_points, std_points, title_prob), key=lambda x: x[1], reverse=True)
    n_releg = min(n_relegated, len(teams) - 1)
    sorted_idx = np.argsort(points_matrix, axis=1)
    bottom_idx = sorted_idx[:, :n_releg]
    relegation_prob = np.array([np.mean(np.any(bottom_idx == k, axis=1)) for k in range(len(teams))])
    ranking = sorted(
        zip(teams, avg_points, std_points, title_prob, relegation_prob),
        key=lambda x: x[1], reverse=True)
    return ranking, n_simulations

def menu_top_scorer(trained_player: dict, normalized_player: dict,
                     player_df: pd.DataFrame, player_sequences: dict) -> None:
    league = pick_from_list(list(teams_league.keys()), "Select a league:")
    results = predict_all_league_players(trained_player, normalized_player, player_df, player_sequences, league)
    if not results:
        print("No players found for this league.")
        return
    ranked = sorted(results, key=lambda x: x[2]["expected_goals"], reverse=True)[:15]    
    print(f"Players with the highest predicted goal output in {league}")
       
    for rank, (pname, team, pred) in enumerate(ranked, 1):
        print(f"Rank {rank}: {pname}, team {team}, predicted goals {pred['expected_goals']:.2f}.")
    
def menu_most_cards(trained_player: dict, normalized_player: dict,
                     player_df: pd.DataFrame, player_sequences: dict, card_type: str = "yellow") -> None:
    league = pick_from_list(list(teams_league.keys()), "Select a league:")
    results = predict_all_league_players(trained_player, normalized_player, player_df, player_sequences, league)
    if not results:
        print("No players found for this league.")
        return

    key = "yellow_card_probability" if card_type == "yellow" else "red_card_probability"
    label = "yellow" if card_type == "yellow" else "red"
    ranked = sorted(results, key=lambda x: x[2][key], reverse=True)[:15]   
    print(f"Players with the highest {'yellow' if card_type == 'yellow' else 'red'} card probability in {league}")
       
    for rank, (pname, team, pred) in enumerate(ranked, 1):
        print(f"Rank {rank}: {pname}, team {team}, probability {pred[key]:.1%}.")
    
def predict_all_league_players(trained_player: dict, normalized_player: dict,
                                player_df: pd.DataFrame, player_sequences: dict,
                                league: str) -> list:

    league_players = player_df[player_df["league"] == league]
    results = []
    for pname, group in league_players.groupby("player"):
        latest_p = group.sort_values("Date").iloc[-1]
        team = latest_p["Squad"]
        opponent = latest_p.get("Opponent", "?")

        p_static = {
            col: latest_p[col]
            for col in normalized_player["player_feature_columns"]
            if col in latest_p.index}
        p_keys = [(n, d) for (n, d) in player_sequences.keys() if n == pname]
        if not p_keys:
            continue
        latest_key = max(p_keys, key=lambda x: x[1])
        p_seq = player_sequences[latest_key]
        try:
            pred = playerprediction(
                model=trained_player["model"], player_name=pname,
                position=str(latest_p.get("position", "")), league_name=league,
                player_static_values=p_static, player_raw_sequence=p_seq,
                player_feature_columns=trained_player["player_feature_columns"],
                player_scaler=normalized_player["player_scaler"],
                sequence_scaler=normalized_player["sequence_scaler"],
                player_medians=normalized_player["player_medians"],
                position_to_id=trained_player["position_to_id"],
                league_to_id=trained_player["league_to_id"],
                opponent_team=opponent, device=trained_player["device"],)
        except (ValueError, KeyError):                                                                             
            continue
        results.append((pname, team, pred))
    return results

def menu_league_prediction(trained_team: dict, normalized_team: dict, team_features_df) -> None:
    league = pick_from_list(list(teams_league.keys()), "Select a league:")
    ranking, n_simulations = simulate_league(trained_team, normalized_team, team_features_df, league)   
    print(f"Predicted {league} league ranking from {n_simulations} simulations")
 
    for rank, (team, pts, std, title_p, releg_p) in enumerate(ranking, 1):
        print(f"Rank {rank}: {team}, expected points {pts:.1f}, point standard deviation {std:.1f}, title probability {title_p:.1%}, relegation probability {releg_p:.1%}.")
    
def menu_predict_winner(trained_team: dict, normalized_team: dict, team_features_df) -> None:
    league = pick_from_list(list(teams_league.keys()), "Select a league:")
    ranking, n_simulations = simulate_league(trained_team, normalized_team, team_features_df, league)
    champion, pts, std, prob,_ = ranking[0] 
    print(f"Predicted champion for {league} from {n_simulations} simulations")   
    print(f"Predicted champion: {champion}")
    print(f"Title probability: {prob:.1%}.")
    print(f"Expected points: {pts:.1f}. Point standard deviation: {std:.1f}.")
    if len(ranking) > 1:
        runner_up_team, _, _, runner_up_prob,_ = ranking[1]
        print(f"Closest challenger: {runner_up_team} with title probability {runner_up_prob:.1%}.")
    
def menu_predict_relegation(trained_team: dict, normalized_team: dict, team_features_df) -> None:
    league = pick_from_list(list(teams_league.keys()), "Select a league:")
    ranking, n_simulations = simulate_league(trained_team, normalized_team, team_features_df, league)
    by_relegation = sorted(ranking, key=lambda x: x[4], reverse=True) 
    print(f"Relegation prediction for {league} from {n_simulations} simulations")   
    for rank, (team, pts, std, title_p, releg_p) in enumerate(by_relegation[:5], 1):
        print(f"Rank {rank}: {team}, expected points {pts:.1f}, relegation probability {releg_p:.1%}.")
     
def menu_player_info(player_df: pd.DataFrame, trained_player: dict,
                      normalized_player: dict, player_sequences: dict) -> None:
    league = pick_from_list(list(teams_league.keys()), "Select a league:")
    teams = sorted(player_df[player_df["league"] == league]["Squad"].dropna().unique().tolist())
    if not teams:
        print("No teams found.")
        return
    team = pick_from_list(teams, "Select a team:")
    players = sorted(player_df[player_df["Squad"] == team]["player"].dropna().unique().tolist())
    if not players:
        print("No players found.")
        return
    pname = pick_from_list(players, "Select a player:")
    latest_p = player_df[player_df["player"] == pname].sort_values("Date").iloc[-1]  
    print(f"Player information: {pname}")  
    for col in ["position", "footed", "height_cm", "weight_kg", "nationality"]:
        if col in latest_p.index and pd.notna(latest_p[col]):
            print(f"{col}: {latest_p[col]}")
    print(f"Current team: {latest_p.get('Squad', '?')}")
    print(f"League: {latest_p.get('league', '?')}") 
    print("Current season statistics")
    for col in player_regression_tar + ["Min"]:
        season_col = f"season_avg_{col}"
        if season_col in latest_p.index and pd.notna(latest_p[season_col]):
            print(f"{col}: {latest_p[season_col]:.2f} per match")
    p_static = {
        col: latest_p[col]
        for col in normalized_player["player_feature_columns"]
        if col in latest_p.index}
    p_keys = [(n, d) for (n, d) in player_sequences.keys() if n == pname]
    if p_keys:
        latest_key = max(p_keys, key=lambda x: x[1])
        p_seq = player_sequences[latest_key]
        try:
            pred = playerprediction(
                model=trained_player["model"], player_name=pname,
                position=str(latest_p.get("position", "")), league_name=league,
                player_static_values=p_static, player_raw_sequence=p_seq,
                player_feature_columns=trained_player["player_feature_columns"],
                player_scaler=normalized_player["player_scaler"],
                sequence_scaler=normalized_player["sequence_scaler"],
                player_medians=normalized_player["player_medians"],
                position_to_id=trained_player["position_to_id"],
                league_to_id=trained_player["league_to_id"],
                opponent_team=str(latest_p.get("Opponent", "?")),
                device=trained_player["device"],)          
            print(pred["summary"])
        except (ValueError, KeyError) as ex:
            print(f"Prediction could not be generated: {ex}")
    
def plot_training_history(history: dict, title: str = "Training History") -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    if history.get("val_loss"):
        plt.plot(epochs[:len(history["val_loss"])], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    print(f"Saved plot: {title.replace(' ', '_')}.png")

def plot_fold_accuracy(fold_results):
    folds = [f"Fold {r['fold']}" for r in fold_results]
    model_acc = [r["winner_accuracy"] * 100 for r in fold_results]
    baseline_acc = [r["baseline_accuracy"] * 100 for r in fold_results]
    x = np.arange(len(folds))
    width = 0.35
    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(
        x - width / 2,
        model_acc,
        width,
        label="Model accuracy")
    bars2 = plt.bar(
        x + width / 2,
        baseline_acc,
        width,
        label="Baseline")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy vs Baseline")
    plt.xticks(x, folds)
    plt.ylim(0, 60)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.bar_label(bars1, fmt="%.1f%%", padding=3)
    plt.bar_label(bars2, fmt="%.1f%%", padding=3)
    plt.tight_layout()
    plt.savefig("Model_vs_Baseline.png", dpi=300, bbox_inches="tight")
    plt.close()
    
def run_menu(trained_team: dict, trained_player: dict, normalized_team: dict,
             normalized_player: dict, team_features_df, player_df, player_sequences: dict) -> None:  
    print("Football Prediction System")
    while True:     
        print("Option 1: Match prediction with score and player predictions")
        print("Option 2: League ranking prediction")
        print("Option 3: League champion prediction")
        print("Option 4: Relegation prediction")
        print("Option 5: Players with the highest predicted goal output")
        print("Option 6: Players with the highest yellow card probability")
        print("Option 7: Players with the highest red card probability")
        print("Option 8: Player information")
        print("Option 0: Exit")
        choice = input("  Selection: ").strip()
        if choice == "1":
            menu_match_prediction(trained_team, trained_player, normalized_team, normalized_player,team_features_df, player_df, player_sequences)
        elif choice == "2":
            menu_league_prediction(trained_team, normalized_team, team_features_df)
        elif choice == "3":
            menu_predict_winner(trained_team, normalized_team, team_features_df)
        elif choice == "4":
            menu_predict_relegation(trained_team, normalized_team, team_features_df)
        elif choice == "5":
            menu_top_scorer(trained_player, normalized_player, player_df, player_sequences)
        elif choice == "6":
            menu_most_cards(trained_player, normalized_player, player_df, player_sequences, "yellow")
        elif choice == "7":
            menu_most_cards(trained_player, normalized_player, player_df, player_sequences, "red")
        elif choice == "8":
            menu_player_info(player_df, trained_player, normalized_player, player_sequences)
        elif choice == "0":
            print("Exiting")
            break
        else:
            print("Invalid option. Enter a number from 0 to 8.")