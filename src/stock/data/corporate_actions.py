"""
corporate_actions.py
--------------------
Corporate action handling: splits, bonus shares, dividends.
Adjusts historical prices to maintain continuity.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List, Dict, Any, Optional

import pandas as pd

logger = logging.getLogger("stock_corporate_actions")


class CorporateAction:
    """Base class for corporate actions."""
    
    def __init__(self, ex_date: str, ratio: float, action_type: str):
        self.ex_date = ex_date
        self.ratio = ratio
        self.action_type = action_type
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ex_date": self.ex_date,
            "ratio": self.ratio,
            "action_type": self.action_type,
        }


class StockSplit(CorporateAction):
    """Stock split (e.g., 1:2 split = ratio 0.5)."""
    
    def __init__(self, ex_date: str, split_ratio: float):
        # split_ratio: 0.5 means 1 old share = 2 new shares
        super().__init__(ex_date, split_ratio, "split")
    
    def adjust(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust prices before ex_date by multiplying by ratio."""
        df = df.copy()
        ex_ts = pd.Timestamp(self.ex_date).tz_localize(None)
        mask = pd.to_datetime(df["date"]) < ex_ts
        
        for col in ["open", "high", "low", "close"]:
            df.loc[mask, col] = df.loc[mask, col] * self.ratio
        
        df.loc[mask, "volume"] = df.loc[mask, "volume"] / self.ratio
        
        logger.info(f"Applied split {self.ratio} on {self.ex_date} to {mask.sum()} rows")
        return df


class BonusShare(CorporateAction):
    """Bonus share issuance (e.g., 1:1 bonus = ratio 1.0)."""
    
    def __init__(self, ex_date: str, bonus_ratio: float):
        # bonus_ratio: 1.0 means 1 bonus per 1 held
        super().__init__(ex_date, bonus_ratio, "bonus")
    
    def adjust(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bonus shares dilute price similarly to splits."""
        df = df.copy()
        ex_ts = pd.Timestamp(self.ex_date).tz_localize(None)
        mask = pd.to_datetime(df["date"]) < ex_ts
        
        # Price adjustment: old_price * (1 / (1 + bonus_ratio))
        adj_factor = 1.0 / (1.0 + self.ratio)
        
        for col in ["open", "high", "low", "close"]:
            df.loc[mask, col] = df.loc[mask, col] * adj_factor
        
        df.loc[mask, "volume"] = df.loc[mask, "volume"] * (1.0 + self.ratio)
        
        logger.info(f"Applied bonus {self.ratio} on {self.ex_date} to {mask.sum()} rows")
        return df


class Dividend(CorporateAction):
    """Cash dividend (stored separately, not adjusting prices for now)."""
    
    def __init__(self, ex_date: str, amount: float, currency: str = "IDR"):
        super().__init__(ex_date, amount, "dividend")
        self.currency = currency
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["currency"] = self.currency
        return d


class CorporateActionRegistry:
    """
    Registry of known corporate actions for BMRI.
    Pre-populated with historical actions for continuity.
    """
    
    # Known BMRI corporate actions (historical)
    BMRI_ACTIONS = [
        # Stock splits
        StockSplit("2008-10-27", 0.5),   # 1:2 split
        StockSplit("2010-11-08", 0.5),   # 1:2 split
        
        # Bonus shares
        BonusShare("2013-10-28", 0.5),   # 1:2 bonus
        BonusShare("2017-10-30", 0.25),  # 1:4 bonus
        
        # Dividends (sample - should be populated from data source)
        # Dividend("2024-05-15", 341.0, "IDR"),
        # Dividend("2023-05-16", 327.0, "IDR"),
    ]
    
    def __init__(self, symbol: str = "BMRI.JK"):
        self.symbol = symbol
        self.actions: List[CorporateAction] = []
        self._load_known()
    
    def _load_known(self):
        """Load known corporate actions for the symbol."""
        if self.symbol.upper().startswith("BMRI"):
            self.actions = list(self.BMRI_ACTIONS)
    
    def add(self, action: CorporateAction):
        """Add a corporate action to the registry."""
        self.actions.append(action)
    
    def apply_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all corporate actions in chronological order."""
        df = df.copy()
        
        # Sort actions by date (oldest first)
        sorted_actions = sorted(
            [a for a in self.actions if isinstance(a, (StockSplit, BonusShare))],
            key=lambda x: x.ex_date,
        )
        
        for action in sorted_actions:
            df = action.adjust(df)
        
        return df
    
    def get_dividends(self) -> List[Dividend]:
        """Get all dividend records."""
        return [a for a in self.actions if isinstance(a, Dividend)]
    
    def to_metadata(self) -> Dict[str, List[Dict]]:
        """Export to metadata.json format."""
        return {
            "splits": [a.to_dict() for a in self.actions if isinstance(a, StockSplit)],
            "bonus_shares": [a.to_dict() for a in self.actions if isinstance(a, BonusShare)],
            "dividends": [a.to_dict() for a in self.actions if isinstance(a, Dividend)],
        }
    
    def load_from_metadata(self, meta: Dict[str, List[Dict]]):
        """Load corporate actions from metadata.json."""
        for s in meta.get("splits", []):
            self.add(StockSplit(s["ex_date"], s["ratio"]))
        for b in meta.get("bonus_shares", []):
            self.add(BonusShare(b["ex_date"], b["ratio"]))
        for d in meta.get("dividends", []):
            self.add(Dividend(d["ex_date"], d["ratio"], d.get("currency", "IDR")))
