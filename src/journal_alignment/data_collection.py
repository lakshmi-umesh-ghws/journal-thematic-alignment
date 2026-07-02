import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yaml


class SemanticScholarCollector:


    def __init__(self, api_key: Optional[str] = None, sleep_seconds: float = 4.0):
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
        self.sleep_seconds = sleep_seconds

        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key

    def search_papers_for_year(
        self,
        journal_name: str,
        year: int,
        fields: str,
        max_pages: int = 3,
    ) -> List[Dict]:


        all_papers = []
        token = None

        for page in range(max_pages):
            params = {
                "venue": journal_name,
                "year": str(year),
                "fields": fields,
                "publicationTypes": "JournalArticle",
                "limit": 100,
            }

            if token:
                params["token"] = token

            print(f"Collecting: {journal_name} | {year} | page {page + 1}")

            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=30,
            )

            if response.status_code == 429:
                print("Rate limit reached. Waiting 60 seconds, then continuing...")
                time.sleep(60)
                continue

            if response.status_code != 200:
                print(f"API error {response.status_code}: {response.text}")
                break

            result = response.json()
            papers = result.get("data", [])
            all_papers.extend(papers)

            token = result.get("token")

            if not token:
                break

            time.sleep(self.sleep_seconds)

        return all_papers

    def collect_journal(
        self,
        journal_name: str,
        short_name: str,
        start_year: int,
        end_year: int,
        fields: str,
    ) -> pd.DataFrame:


        rows = []

        for year in range(start_year, end_year + 1):
            papers = self.search_papers_for_year(
                journal_name=journal_name,
                year=year,
                fields=fields,
            )

            for paper in papers:
                external_ids = paper.get("externalIds") or {}
                journal_info = paper.get("journal") or {}

                rows.append(
                    {
                        "paper_id": paper.get("paperId"),
                        "journal_query": journal_name,
                        "journal_short_name": short_name,
                        "title": paper.get("title"),
                        "abstract": paper.get("abstract"),
                        "year": paper.get("year"),
                        "publication_date": paper.get("publicationDate"),
                        "venue": paper.get("venue"),
                        "journal_name_from_api": journal_info.get("name"),
                        "citation_count": paper.get("citationCount"),
                        "doi": external_ids.get("DOI"),
                        "url": paper.get("url"),
                        "fields_of_study": paper.get("fieldsOfStudy"),
                    }
                )

        return pd.DataFrame(rows)


def load_config(config_path: str = "config.yaml") -> Dict:


    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_raw_data(df: pd.DataFrame, output_path: str) -> None:


    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows to {output_file}")