from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class POSDataStore:
    """
    Simple in-memory POS dataframe store shared by the FastAPI app.
    """

    df: Optional[pd.DataFrame] = field(default=None)

    def load_from_bytes(self, content: bytes) -> None:
        buffer = io.BytesIO(content)
        df = pd.read_csv(buffer)

        # Basic normalization
        if "transaction_date" in df.columns:
            df["transaction_date"] = pd.to_datetime(df["transaction_date"])
            if "day_of_week" not in df.columns:
                df["day_of_week"] = df["transaction_date"].dt.day_name()

        # Standardize column names to lower snake_case
        df.columns = [c.strip() for c in df.columns]

        self.df = df

    def get_unique_items(self) -> List[str]:
        if self.df is None or "item_name" not in self.df.columns:
            return []
        return sorted(self.df["item_name"].dropna().unique().tolist())

    @staticmethod
    def generate_order_id() -> str:
        return str(uuid.uuid4())

