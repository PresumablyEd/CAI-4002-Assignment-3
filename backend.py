"""
backend.py


Right now this file only:
- loads the default CSV files
- defines placeholder functions for Apriori / Eclat / FP-Growth
- defines a placeholder recommendation function used by the UI
"""

from typing import List
import pandas as pd


# Data loading helpers

def load_default_transactions() -> pd.DataFrame:
    """
    Load the sample transactions from the CSV file shipped with the assignment.

    Expected format of sample_transactions.csv:
        transaction_id,items
        1,"milk,bread,butter,eggs"
        2,"apple,banana,orange"
        ...

    You can modify this later if you change the folder structure.
    """
    return pd.read_csv("sample_transactions.csv")


def load_default_products() -> pd.DataFrame:
    """
    Load the products metadata.

    Expected format of products.csv:
        product_id,product_name,category
        1,milk,dairy
        2,bread,bakery
        ...
    """
    return pd.read_csv("products.csv")


# Association rule mining

def mine_association_rules(
    transactions_df: pd.DataFrame,
    algorithm: str,
    min_support: float,
    min_confidence: float,
    min_lift: float,
) -> pd.DataFrame:
    """
    Main entry point that the Streamlit front end calls.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        DataFrame with at least two columns: transaction_id, items
        where 'items' is a comma-separated list of product names.
    algorithm : str
        "apriori", "eclat", or "fp-growth" (lowercase).
    min_support : float
        Minimum support threshold (0-1).
    min_confidence : float
        Minimum confidence threshold (0-1).
    min_lift : float
        Minimum lift threshold (>= 0).

    Returns
    -------
    pd.DataFrame
        A DataFrame of rules. Suggestion for columns:
        ['antecedent', 'consequent', 'support', 'confidence', 'lift'].

    Note
    ----
    For now this function returns an empty DataFrame.
    The backend should:

    - Convert the 'items' column into a transaction list format.
    - Implement Apriori / Eclat / FP-Growth.
    - Filter rules based on the thresholds.
    - Return them as a DataFrame that the UI can display.
    """
    # TODO: implement actual algorithms here
    columns = ["antecedent", "consequent", "support", "confidence", "lift"]
    empty_rules = pd.DataFrame(columns=columns)
    return empty_rules


def recommend_from_basket(basket_items: List[str]) -> List[str]:
    """
    Simple placeholder used on the Shopping Simulator page.

    Parameters
    ----------
    basket_items : List[str]
        Items currently in the user's basket (product names).

    Returns
    -------
    List[str]
        For now: an empty list. Later, this can:
        - look up mined rules
        - propose consequents that are not already in the basket
        - return a short list of recommended product names.
    """
    # TODO: use mined rules once they are available
    return []
