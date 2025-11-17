"""
backend.py

Backend implementation for Supermarket Association Rule Mining application.
Includes data preprocessing, Apriori algorithm, Eclat algorithm, and recommendation system.
"""

from typing import List, Tuple, Dict, Set, Any
import pandas as pd
import time
import re
from collections import defaultdict, Counter
import itertools


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


# Data Preprocessing Module

def preprocess_data(transactions_df: pd.DataFrame, products_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive data preprocessing pipeline.
    
    Handles:
    - Empty transaction detection and removal
    - Single-item transaction handling
    - Duplicate item removal within transactions
    - Product name standardization (case, whitespace)
    - Invalid product ID handling
    
    Returns:
    - Preprocessed DataFrame
    - Preprocessing report dictionary
    """
    
    # Initialize report
    report = {
        'before_cleaning': {
            'total_transactions': len(transactions_df),
            'empty_transactions': 0,
            'single_item_transactions': 0,
            'duplicate_items': 0,
            'invalid_items': 0
        },
        'after_cleaning': {
            'valid_transactions': 0,
            'total_items': 0,
            'unique_products': 0
        },
        'issues_fixed': {
            'empty_removed': 0,
            'single_item_removed': 0,
            'duplicates_removed': 0,
            'invalid_removed': 0
        }
    }
    
    # Get valid product names from products.csv
    valid_products = set(products_df['product_name'].str.lower().str.strip())
    
    # Process each transaction
    processed_transactions = []
    
    for _, row in transactions_df.iterrows():
        transaction_id = row['transaction_id']
        items_str = str(row['items']) if pd.notna(row['items']) else ""
        
        # Skip empty transactions
        if not items_str.strip():
            report['before_cleaning']['empty_transactions'] += 1
            report['issues_fixed']['empty_removed'] += 1
            continue
            
        # Split and clean items
        raw_items = [item.strip() for item in items_str.split(',')]
        
        # Remove duplicates within transaction
        unique_items = []
        seen_items = set()
        for item in raw_items:
            if item and item not in seen_items:
                unique_items.append(item)
                seen_items.add(item)
            elif item and item in seen_items:
                report['before_cleaning']['duplicate_items'] += 1
                report['issues_fixed']['duplicates_removed'] += 1
        
        # Standardize item names (lowercase, remove extra whitespace)
        standardized_items = []
        for item in unique_items:
            clean_item = item.lower().strip()
            # Remove extra internal whitespace
            clean_item = re.sub(r'\s+', ' ', clean_item)
            standardized_items.append(clean_item)
        
        # Filter invalid items
        valid_items = []
        for item in standardized_items:
            if item in valid_products:
                valid_items.append(item)
            else:
                report['before_cleaning']['invalid_items'] += 1
                report['issues_fixed']['invalid_removed'] += 1
        
        # Skip single-item transactions
        if len(valid_items) <= 1:
            report['before_cleaning']['single_item_transactions'] += 1
            report['issues_fixed']['single_item_removed'] += 1
            continue
            
        # Add valid transaction
        processed_transactions.append({
            'transaction_id': transaction_id,
            'items': ','.join(valid_items)
        })
    
    # Create cleaned DataFrame
    cleaned_df = pd.DataFrame(processed_transactions)
    
    # Update report statistics
    report['after_cleaning']['valid_transactions'] = len(cleaned_df)
    
    # Count total items and unique products
    all_items = []
    for items_str in cleaned_df['items']:
        all_items.extend(items_str.split(','))
    
    report['after_cleaning']['total_items'] = len(all_items)
    report['after_cleaning']['unique_products'] = len(set(all_items))
    
    return cleaned_df, report


# Association Rule Mining Algorithms

class AprioriAlgorithm:
    """Implementation of Apriori algorithm for association rule mining."""
    
    def __init__(self, transactions: List[Set[str]], min_support: float, min_confidence: float):
        self.transactions = transactions
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_support_count = int(min_support * len(transactions))
        self.frequent_itemsets = {}
        self.association_rules = []
        
    def get_support(self, itemset: Set[str]) -> int:
        """Calculate support count for an itemset."""
        count = 0
        for transaction in self.transactions:
            if itemset.issubset(transaction):
                count += 1
        return count
    
    def generate_candidate_itemsets(self, prev_itemsets: List[Set[str]], k: int) -> List[Set[str]]:
        """Generate candidate itemsets of size k from previous itemsets."""
        candidates = []
        n = len(prev_itemsets)
        
        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = prev_itemsets[i]
                itemset2 = prev_itemsets[j]
                
                # Join step: combine if first k-2 items are the same
                list1 = sorted(list(itemset1))
                list2 = sorted(list(itemset2))
                
                if list1[:k-2] == list2[:k-2]:
                    new_itemset = itemset1.union(itemset2)
                    if len(new_itemset) == k:
                        # Prune step: check if all subsets are frequent
                        all_subsets_frequent = True
                        for subset in itertools.combinations(new_itemset, k-1):
                            if set(subset) not in prev_itemsets:
                                all_subsets_frequent = False
                                break
                        
                        if all_subsets_frequent:
                            candidates.append(new_itemset)
        
        return candidates
    
    def find_frequent_itemsets(self) -> Dict[int, List[Tuple[Set[str], int]]]:
        """Find all frequent itemsets using Apriori algorithm."""
        # Find frequent 1-itemsets
        item_counts = Counter()
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        
        frequent_1_itemsets = []
        for item, count in item_counts.items():
            if count >= self.min_support_count:
                frequent_1_itemsets.append({item})
        
        self.frequent_itemsets[1] = [(itemset, self.get_support(itemset)) 
                                   for itemset in frequent_1_itemsets]
        
        # Find frequent k-itemsets for k >= 2
        k = 2
        while self.frequent_itemsets[k-1]:
            prev_frequent = [itemset for itemset, _ in self.frequent_itemsets[k-1]]
            candidates = self.generate_candidate_itemsets(prev_frequent, k)
            
            frequent_k_itemsets = []
            for candidate in candidates:
                support = self.get_support(candidate)
                if support >= self.min_support_count:
                    frequent_k_itemsets.append((candidate, support))
            
            if frequent_k_itemsets:
                self.frequent_itemsets[k] = frequent_k_itemsets
                k += 1
            else:
                break
        
        return self.frequent_itemsets
    
    def generate_association_rules(self) -> List[Dict[str, Any]]:
        """Generate association rules from frequent itemsets."""
        rules = []
        
        for k in range(2, len(self.frequent_itemsets) + 1):
            for itemset, itemset_support in self.frequent_itemsets[k]:
                itemset_support_pct = itemset_support / len(self.transactions)
                
                # Generate all non-empty proper subsets
                for i in range(1, k):
                    for antecedent in itertools.combinations(itemset, i):
                        antecedent_set = set(antecedent)
                        consequent_set = itemset - antecedent_set
                        
                        # Calculate confidence
                        antecedent_support = self.get_support(antecedent_set)
                        if antecedent_support > 0:
                            confidence = itemset_support / antecedent_support
                            
                            if confidence >= self.min_confidence:
                                # Calculate lift
                                consequent_support = self.get_support(consequent_set)
                                if consequent_support > 0:
                                    lift = confidence / (consequent_support / len(self.transactions))
                                    
                                    rules.append({
                                        'antecedent': antecedent_set,
                                        'consequent': consequent_set,
                                        'support': itemset_support_pct,
                                        'confidence': confidence,
                                        'lift': lift
                                    })
        
        return rules


class EclatAlgorithm:
    """Implementation of Eclat algorithm for association rule mining."""
    
    def __init__(self, transactions: List[Set[str]], min_support: float, min_confidence: float):
        self.transactions = transactions
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_support_count = int(min_support * len(transactions))
        self.vertical_db = {}
        self.frequent_itemsets = {}
        self.association_rules = []
        
    def build_vertical_database(self):
        """Build vertical database (TID-sets)."""
        self.vertical_db = {}
        for tid, transaction in enumerate(self.transactions):
            for item in transaction:
                if item not in self.vertical_db:
                    self.vertical_db[item] = set()
                self.vertical_db[item].add(tid)
    
    def eclat_depth_first(self, prefix: Set[str], items: List[Tuple[str, Set[int]]], k: int):
        """Recursive depth-first search for frequent itemsets."""
        while items:
            item1, tid_set1 = items.pop()
            new_prefix = prefix.union({item1})
            
            # Calculate support
            support_count = len(tid_set1)
            support_pct = support_count / len(self.transactions)
            
            if support_count >= self.min_support_count:
                # Store frequent itemset
                if k not in self.frequent_itemsets:
                    self.frequent_itemsets[k] = []
                self.frequent_itemsets[k].append((new_prefix, support_count))
                
                # Generate new items for next level
                new_items = []
                for item2, tid_set2 in items:
                    intersection = tid_set1.intersection(tid_set2)
                    if len(intersection) >= self.min_support_count:
                        new_items.append((item2, intersection))
                
                # Recursive call
                if new_items:
                    self.eclat_depth_first(new_prefix, new_items, k + 1)
    
    def find_frequent_itemsets(self) -> Dict[int, List[Tuple[Set[str], int]]]:
        """Find all frequent itemsets using Eclat algorithm."""
        self.build_vertical_database()
        
        # Convert vertical DB to list of (item, tid_set) sorted by support
        items_with_tids = [(item, tid_set) for item, tid_set in self.vertical_db.items()]
        items_with_tids.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Start depth-first search
        self.eclat_depth_first(set(), items_with_tids, 1)
        
        return self.frequent_itemsets
    
    def generate_association_rules(self) -> List[Dict[str, Any]]:
        """Generate association rules from frequent itemsets."""
        rules = []
        
        # Flatten all frequent itemsets for rule generation
        all_frequent = []
        for k_itemsets in self.frequent_itemsets.values():
            all_frequent.extend(k_itemsets)
        
        for itemset, itemset_support in all_frequent:
            if len(itemset) >= 2:
                itemset_support_pct = itemset_support / len(self.transactions)
                
                # Generate all non-empty proper subsets
                k = len(itemset)
                for i in range(1, k):
                    for antecedent in itertools.combinations(itemset, i):
                        antecedent_set = set(antecedent)
                        consequent_set = itemset - antecedent_set
                        
                        # Calculate confidence using vertical DB
                        antecedent_tids = set.intersection(*[self.vertical_db[item] for item in antecedent_set])
                        antecedent_support = len(antecedent_tids)
                        
                        if antecedent_support > 0:
                            confidence = itemset_support / antecedent_support
                            
                            if confidence >= self.min_confidence:
                                # Calculate lift
                                consequent_tids = set.intersection(*[self.vertical_db[item] for item in consequent_set])
                                consequent_support = len(consequent_tids)
                                if consequent_support > 0:
                                    lift = confidence / (consequent_support / len(self.transactions))
                                    
                                    rules.append({
                                        'antecedent': antecedent_set,
                                        'consequent': consequent_set,
                                        'support': itemset_support_pct,
                                        'confidence': confidence,
                                        'lift': lift
                                    })
        
        return rules


def transactions_to_list(transactions_df: pd.DataFrame) -> List[Set[str]]:
    """Convert DataFrame transactions to list of sets for algorithm processing."""
    transactions = []
    for _, row in transactions_df.iterrows():
        if pd.notna(row['items']):
            items = set(item.strip() for item in row['items'].split(','))
            transactions.append(items)
    return transactions


# Main Mining Function

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
        A DataFrame of rules with columns:
        ['antecedent', 'consequent', 'support', 'confidence', 'lift'].
    """
    
    # Load products for preprocessing
    products_df = load_default_products()
    
    # Preprocess data
    cleaned_df, _ = preprocess_data(transactions_df, products_df)
    
    # Convert to transaction list format
    transactions = transactions_to_list(cleaned_df)
    
    if not transactions:
        return pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"])
    
    # Track performance
    start_time = time.time()
    
    # Run selected algorithm
    if algorithm == "apriori":
        apriori = AprioriAlgorithm(transactions, min_support, min_confidence)
        apriori.find_frequent_itemsets()
        rules = apriori.generate_association_rules()
    elif algorithm == "eclat":
        eclat = EclatAlgorithm(transactions, min_support, min_confidence)
        eclat.find_frequent_itemsets()
        rules = eclat.generate_association_rules()
    else:
        # Fallback to Apriori for unsupported algorithms
        apriori = AprioriAlgorithm(transactions, min_support, min_confidence)
        apriori.find_frequent_itemsets()
        rules = apriori.generate_association_rules()
    
    execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Filter rules by lift and convert to DataFrame
    filtered_rules = []
    for rule in rules:
        if rule['lift'] >= min_lift:
            filtered_rules.append({
                'antecedent': ', '.join(sorted(rule['antecedent'])),
                'consequent': ', '.join(sorted(rule['consequent'])),
                'support': round(rule['support'], 4),
                'confidence': round(rule['confidence'], 4),
                'lift': round(rule['lift'], 4)
            })
    
    # Create DataFrame with explicit data types
    if filtered_rules:
        rules_df = pd.DataFrame(filtered_rules)
        # Ensure proper data types for Streamlit compatibility
        rules_df['antecedent'] = rules_df['antecedent'].astype(str)
        rules_df['consequent'] = rules_df['consequent'].astype(str)
        rules_df['support'] = rules_df['support'].astype(float)
        rules_df['confidence'] = rules_df['confidence'].astype(float)
        rules_df['lift'] = rules_df['lift'].astype(float)
        
        # Add performance info as separate columns
        rules_df['_execution_time_ms'] = round(execution_time, 2)
        rules_df['_algorithm'] = algorithm
        rules_df['_rules_count'] = len(filtered_rules)
        
        return rules_df
    else:
        return pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"])


# Recommendation System

def recommend_from_basket(basket_items: List[str]) -> List[str]:
    """
    Generate recommendations based on current basket items using mined rules.

    Parameters
    ----------
    basket_items : List[str]
        Items currently in the user's basket (product names).

    Returns
    -------
    List[str]
        List of recommended product names with confidence percentages.
    """
    
    # Load and preprocess data
    transactions_df = load_default_transactions()
    products_df = load_default_products()
    cleaned_df, _ = preprocess_data(transactions_df, products_df)
    
    # Mine rules with default parameters
    transactions = transactions_to_list(cleaned_df)
    if not transactions:
        return []
    
    # Use Apriori for recommendations (faster for this use case)
    apriori = AprioriAlgorithm(transactions, min_support=0.1, min_confidence=0.3)
    apriori.find_frequent_itemsets()
    rules = apriori.generate_association_rules()
    
    # Find rules where basket items match antecedent
    basket_set = set(item.lower().strip() for item in basket_items)
    recommendations = {}
    
    for rule in rules:
        antecedent_set = rule['antecedent']
        consequent_set = rule['consequent']
        confidence = rule['confidence']
        
        # Check if basket contains the antecedent
        if antecedent_set.issubset(basket_set):
            # Get items in consequent that aren't already in basket
            new_items = consequent_set - basket_set
            for item in new_items:
                if item not in recommendations or confidence > recommendations[item]:
                    recommendations[item] = confidence
    
    # Sort by confidence and format output
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    
    formatted_recommendations = []
    for item, confidence in sorted_recommendations[:5]:  # Top 5 recommendations
        confidence_pct = int(confidence * 100)
        strength = "Strong" if confidence_pct >= 70 else "Moderate" if confidence_pct >= 40 else "Weak"
        bar_length = int(confidence_pct / 5)  # Scale for visualization
        
        formatted_recommendations.append(
            f"- {item.title()}: {confidence_pct}% of the time {'█' * bar_length} ({strength})"
        )
    
    return formatted_recommendations


# Performance comparison function (for future enhancement)
def compare_algorithms(transactions_df: pd.DataFrame, min_support: float, min_confidence: float) -> Dict[str, Any]:
    """
    Compare performance of Apriori and Eclat algorithms.
    
    Returns performance metrics for both algorithms.
    """
    performance = {}
    
    for algorithm in ['apriori', 'eclat']:
        start_time = time.time()
        rules_df = mine_association_rules(transactions_df, algorithm, min_support, min_confidence, 1.0)
        execution_time = (time.time() - start_time) * 1000
        
        performance[algorithm] = {
            'execution_time_ms': round(execution_time, 2),
            'rules_generated': len(rules_df),
            'memory_usage': 'N/A'  # Could be enhanced with memory profiling
        }
    
    return performance
