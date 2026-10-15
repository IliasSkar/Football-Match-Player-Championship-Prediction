from __future__ import annotations
import os
import re
import time
from io import StringIO
import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

driver = uc.Chrome(version_main=149)

COMPETITION_FILTER = "Serie A"

DELAY_SECONDS = 10
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "Italian", "2025-2026")

TEAMS_CONFIG = [   
    {"url": "https://fbref.com/en/squads/dc56fe14/2025-2026/Milan-Stats","team": "Milan", "season": "2025-2026"},
    {"url": "https://fbref.com/en/squads/d609edc0/2025-2026/Internazionale-Stats","team": "Internazionale", "season": "2025-2026"},
]
BASE_URL="https://fbref.com"
FIRST_REQUEST=True

def fetch_page(url: str) -> BeautifulSoup | None:
    global FIRST_REQUEST
    try:
        print(f"Downloading:{url}")
        driver.get(url)
        if FIRST_REQUEST:
            input("Solve the Captcha in the Chrome tab and then press Enter here...")
            FIRST_REQUEST=False

        try:
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table[id*='stats']")))
        except:
            time.sleep(8)

        return BeautifulSoup(driver.page_source,"lxml")
    except Exception as e:
        print(f"Error in {url}: {e}")
        return None

def extract_hidden_tables(soup: BeautifulSoup) -> BeautifulSoup:
    from bs4 import Comment
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in comment:
            new_soup = BeautifulSoup(comment, "lxml")
            comment.replace_with(new_soup)
    return soup

def get_player_bio(player_url: str)->dict:
    bio={"position": "", "footed": "", "height_cm": "", "weight_kg": "", "nationality": ""}

    soup=fetch_page(BASE_URL + player_url)
    if not soup:
        return bio

    meta_div=soup.find("div", {"id": "meta"})
    if not meta_div:
        return bio

    for p_tag in meta_div.find_all("p"):
        text=p_tag.get_text(" ", strip=True)

        if "Position:" in text:
            pos_part=text.split("Position:")[-1].split("▪")[0].split("(")[0].strip()
            bio["position"]=pos_part

        if "Footed:" in text:
            footed_part=text.split("Footed:")[-1].split("▪")[0].strip()
            bio["footed"]=footed_part

    meta_text=meta_div.get_text(" ", strip=True)

    height_match=re.search(r"(\d{2,3})cm", meta_text)
    if height_match:
        bio["height_cm"]=height_match.group(1)

    weight_match=re.search(r"(\d{2,3})kg", meta_text)
    if weight_match:
        bio["weight_kg"]=weight_match.group(1)

    nat_link=meta_div.find("a", href=re.compile(r"/en/country/"))
    if nat_link:
        bio["nationality"]=nat_link.get_text(strip=True)
    else:
        for p_tag in meta_div.find_all("p"):
            if "National Team" in p_tag.get_text():
                a=p_tag.find("a")
                if a:
                    bio["nationality"]=a.get_text(strip=True)
                break
    return bio

def get_player_links(squad_url: str) -> list[dict]:
    full_url=squad_url if squad_url.startswith("http") else BASE_URL + squad_url
    soup=fetch_page(full_url)
    if not soup:
        return []

    from bs4 import Comment
    raw_html=str(soup)
    comments=soup.find_all(string=lambda t: isinstance(t, Comment))
    for comment in comments:
        if "matchlogs" in comment or "<table" in comment:
            raw_html = raw_html.replace(f"<!--{comment}-->", str(comment))
    soup = BeautifulSoup(raw_html, "lxml")
    players=[]
    seen_players=set()

    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        if "/players/" not in href or "matchlogs" not in href:
            continue

        if "/summary/" not in href:
            continue

        player_row=link.find_parent("tr")
        if not player_row:
            continue

        name_link=player_row.find("a", href=re.compile(r"/players/[^/]+/[^/]+$"))
        if not name_link:
            name_link=player_row.find("a", href=re.compile(r"/players/"))
        if not name_link:
            continue

        player_name=name_link.text.strip()
        player_url=name_link["href"]

        if player_name in seen_players:
            continue

        mp_cell=player_row.find("td", {"data-stat": "games"})
        if not mp_cell:
            continue
        seen_players.add(player_name)
        players.append({
            "name":player_name,
            "player_url":player_url,
            "matches_url":href,
            "mp":mp_cell.get_text(strip=True)})

    print(f"Found {len(players)} players")
    players.sort(key=lambda p:int(p["mp"]), reverse=True)
    players=players[:25]
    return players

def get_player_match_stats(player: dict, competition: str) -> pd.DataFrame | None:
    soup=fetch_page(BASE_URL + player["matches_url"])
    if not soup:
        return None

    soup=extract_hidden_tables(soup)
    tables=soup.find_all("table")
    if not tables:
        print(f"No tables found for {player['name']}")
        return None

    all_dfs=[]
    for table in tables:
        try:
            df=pd.read_html(StringIO(str(table)))[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[-1] if col[-1] != col[0] else col[0] for col in df.columns]
            if "Comp" not in df.columns:
                continue
            df_filtered=df[df["Comp"] == competition].copy()
            if df_filtered.empty:
                continue
            df_filtered=df_filtered[df_filtered["Date"] != "Date"]
            all_dfs.append(df_filtered)
        except Exception:
            continue
    if not all_dfs:
        print(f"No matches found {competition} for {player['name']}")
        return None

    combined=pd.concat(all_dfs, ignore_index=True)
    combined=combined.drop_duplicates(subset=["Date", "Opponent"], keep="first")
    print(f"{player['name']}: {len(combined)} matches {competition}")

    if "Min" in combined.columns:
        combined=combined[~combined["Min"].astype(str).str.contains("matchday|did not", case=False, na=False)]
        combined=combined[combined["Min"].astype(str).str.strip() != ""]
        combined=combined[combined["Min"].astype(str) != "0"]
    if "Match Report" in combined.columns:
        combined = combined.drop(columns=["Match Report"])
    return combined

def scrape_team(team_config: dict) -> None:
    team=team_config["team"]
    season=team_config["season"]
    url=team_config["url"]
    print(f"\n{'═'*60}")
    print(f"Team:{team} | Season: {season}")
    print(f"{'═'*60}")
    print(f"\n[1/3] Downloading player list...")
    players=get_player_links(url)
    if not players:
        print(f"No players found for {team} {season}")
        return

    print(f"\n[2/3] Downloading bio + statistics ({len(players)} players)...")
    all_player_dfs=[]

    for i, player in enumerate(players, 1):
        print(f"\n  [{i}/{len(players)}] {player['name']}")
        bio = {"position": "", "footed": "", "height_cm": "", "weight_kg": "", "nationality": ""}
        if player.get("player_url"):
            print(f"Downloading bio...")
            bio = get_player_bio(player["player_url"])
            time.sleep(DELAY_SECONDS)
        df = get_player_match_stats(player, COMPETITION_FILTER)

        if df is not None:
            df.insert(0,"player",player["name"])
            df.insert(1,"position",bio["position"])
            df.insert(2,"footed",bio["footed"])
            df.insert(3,"height_cm",bio["height_cm"])
            df.insert(4,"weight_kg",bio["weight_kg"])
            df.insert(5,"nationality",bio["nationality"])
            all_player_dfs.append(df)

        if i < len(players):
            print(f"Waiting {DELAY_SECONDS} seconds...")
            time.sleep(DELAY_SECONDS)

    print(f"\n[3/3] Saving CSV...")
    if not all_player_dfs:
        print(f"No data found for {team} {season}")
        return

    final_df=pd.concat(all_player_dfs, ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    comp_clean=COMPETITION_FILTER.replace(" ", "-")
    filename=f"{team}_{season}_{comp_clean}.csv"
    filepath=os.path.join(OUTPUT_DIR, filename)
    final_df.to_csv(filepath, index=False, encoding="utf-8-sig")

    print(f"\n Saved: {filepath}")
    print(f" {len(final_df)} registrations | {final_df['player'].nunique()} players")

if __name__ == "__main__":
    print(f"Championship:{COMPETITION_FILTER}")
    print(f"Teams:{len(TEAMS_CONFIG)}")
    print(f"Delay: {DELAY_SECONDS} seconds per request")
    for team_config in TEAMS_CONFIG:
        scrape_team(team_config)
    driver.quit()
    print("Finish")