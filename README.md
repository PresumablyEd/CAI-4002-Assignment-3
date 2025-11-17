### Interactive Supermarket Simulation with Association Rule Mining

#### Author Information

- **Name**: Nicole Sanchez and Edwin Altamirano
- **Student ID**: 6398998 and 6403500
- **Course**: CAI 4002 - Artificial Intelligence
- **Semester**: Fall 2025

#### System Overview

This application provides an interactive supermarket environment where users can browse products, create virtual shopping baskets, and perform association rule mining to discover purchasing patterns. The system implements both Apriori and Eclat algorithms for mining association rules from transaction data, with comprehensive data preprocessing and user-friendly recommendations.

#### Technical Stack

- **Language**: Python 3.8+
- **Key Libraries**: Streamlit (frontend UI), Pandas (data handling), Collections (efficient data structures)
- **UI Framework**: Streamlit
- **Algorithms**: Apriori, Eclat (implemented from scratch)

#### Installation

##### Prerequisites
- Python 3.8+
- pip package manager

##### Setup
```bash
# Clone or extract project
# Navigate to project directory
cd CAI-4002-Assignment-3

# Create and activate virtual environment (recommended)
python -m venv streamlit_env
streamlit_env\Scripts\Activate.ps1

# Install dependencies
pip install streamlit pandas

# Run application
streamlit run streamlit_app.py
```

#### Usage

##### 1. Load Data
- The **Data Explorer** page automatically loads `sample_transactions.csv` and `products.csv` from the project folder
- Users can also upload their own CSV files with transaction data
- Data preprocessing runs automatically when loading transactions

##### 2. Preprocess Data
- **Empty Transaction Detection and Removal**: Automatically removes transactions with no items
- **Single-Item Transaction Handling**: Removes transactions containing only one item (no association value)
- **Duplicate Item Removal**: Eliminates duplicate product entries within each transaction
- **Product Name Standardization**: Handles case inconsistencies and extra whitespace
- **Invalid Product ID Handling**: Removes items not found in the valid products list

##### 3. Run Mining
- On the **Association Rules** page, users can:
  - Select algorithm (Apriori or Eclat)
  - Set minimum support, confidence, and lift thresholds
  - Run the analysis to generate association rules
- Performance metrics (execution time, rules count) are displayed

##### 4. Query Results
- **Shopping Simulator**: Add items to basket and get real-time recommendations
- **Interactive Recommendations**: View associated products with confidence percentages and strength indicators
- **Business Insights**: Formatted output with visual strength bars and actionable recommendations

#### Algorithm Implementation

##### Apriori Algorithm
Implementation of the classic Apriori algorithm using horizontal data format with level-wise candidate generation and pruning. The algorithm efficiently discovers frequent itemsets by generating candidates level by level and pruning using minimum support threshold, then extracts association rules using minimum confidence threshold.

- **Data structure**: Dictionary of itemsets with support counts
- **Candidate generation**: Breadth-first, level-wise approach
- **Pruning strategy**: Minimum support threshold with subset checking
- **Rule generation**: Confidence-based filtering with lift calculation

##### Eclat Algorithm
Implementation of the Eclat algorithm using vertical data format (TID-sets) with depth-first search. The algorithm builds a vertical database of transaction IDs for each item and performs efficient set intersections for support counting.

- **Data structure**: Vertical database (TID-sets)
- **Search strategy**: Depth-first search with prefix-based extension
- **Intersection method**: Set operations for efficient support counting
- **Rule generation**: Confidence calculation using vertical database intersections

#### Performance Results

Tested on provided dataset (100 transactions, 77 valid transactions after cleaning):

| Algorithm | Runtime (ms) | Rules Generated | Memory Usage |
|-----------|--------------|-----------------|--------------|
| Apriori   | ~50-100 ms   | 15-25 rules     | Moderate     |
| Eclat     | ~30-80 ms    | 15-25 rules     | Low          |

**Parameters**: min_support = 0.2, min_confidence = 0.5, min_lift = 1.0

**Analysis**: The Eclat algorithm generally performs faster due to its efficient vertical data representation and set intersection operations, while Apriori provides similar rule generation with slightly higher memory usage for candidate storage.

#### Project Structure

```
project-root/
├── streamlit_app.py         # Streamlit frontend (Nicole)
├── backend.py               # Complete mining algorithms (Edwin)
├── sample_transactions.csv  # Provided transaction dataset
├── products.csv            # Product validation data
├── README.md               # Updated documentation
└── streamlit_env/          # Virtual environment (optional)
```

#### Data Preprocessing

Comprehensive data cleaning pipeline handles the following issues in the provided dataset:

- **Empty transactions**: 5 removed (transactions 12, 27, 41, 66, 86)
- **Single-item transactions**: 8 removed (transactions 11, 29, 51, 65, 77, 89)
- **Duplicate items**: 15+ instances cleaned (e.g., "apple,apple,banana")
- **Case inconsistencies**: All standardized to lowercase (e.g., "Milk" → "milk")
- **Invalid items**: 2 removed ("item_999", "item_888")
- **Extra whitespace**: Trimmed from all items (e.g., "Bread " → "bread")

**Final Dataset**: 77 valid transactions, 234 total items, 18 unique products

#### Testing

Verified functionality:
- [✓] CSV import and parsing
- [✓] All preprocessing operations
- [✓] Both algorithm implementations (Apriori and Eclat)
- [✓] Interactive query system
- [✓] Performance measurement
- [✓] User-friendly recommendations
- [✓] Streamlit frontend integration

Test cases:
- **Data Loading**: Verified successful loading of both CSV files with proper error handling
- **Preprocessing**: Tested all cleaning operations on the provided dirty dataset
- **Algorithm Validation**: Compared rule generation between Apriori and Eclat for consistency
- **Recommendation System**: Tested various basket combinations for relevant suggestions

#### Known Limitations

- **Memory Usage**: Apriori algorithm may use more memory for large datasets due to candidate generation
- **Performance**: Both algorithms may slow down significantly with very large transaction sets (>10,000 transactions)
- **FP-Growth**: The third algorithm mentioned in the frontend is not implemented (falls back to Apriori)
- **Data Types**: Some Arrow serialization issues may occur with very large dataframes in Streamlit

#### AI Tool Usage

Claude AI was used to assist with the backend algorithm implementation, specifically for understanding the Eclat algorithm's vertical data representation and debugging set intersection operations. The AI helped generate boilerplate code for the Apriori candidate generation and rule extraction logic, which was then thoroughly reviewed, tested, and adapted for this specific implementation. All generated code was validated against the assignment requirements and tested with the provided dataset.

#### References

- Course lecture materials on Association Rule Mining
- Streamlit documentation for UI components
- Pandas documentation for data manipulation
- Academic papers on Apriori and Eclat algorithms
- Python standard library documentation for collections and itertools
