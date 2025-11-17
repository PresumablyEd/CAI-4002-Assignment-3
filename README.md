### Interactive Supermarket Simulation with Association Rule Mining

#### Author Information

- **Name**: Nicole Sanchez and Edwin Altamirano
- **Student ID**: 6398998 and _____
- **Course**: CAI 4002 - Artificial Intelligence
- **Semester**: [Fall Year]



#### System Overview

This application provides an interactive supermarket environment where users can browse available products, create a virtual shopping basket, and explore transactions from a dataset. The system also includes a section for running association rule mining (Apriori, Eclat, FP-Growth), which will be implemented in the backend. The frontend is built using Streamlit to provide an easy-to-use interface for loading data, simulating shopping activity, and eventually viewing mined rules.


#### Technical Stack

- **Language**: Python 3
- **Key Libraries**: Streamlit (frontend UI), Pandas (CSV loading & dataframe handling)
- **UI Framework**: Streamlit



#### Installation

##### Prerequisites
- Python 3.8+
- pip package manager
##### Setup
```bash
# Clone or extract project

# Navigate to project
cd [project-directory]

# Install dependencies
pip install streamlit pandas

# Run application
streamlit run streamlit_app.py


```



#### Usage

##### 1. Load Data
-  The Data Explorer page automatically loads sample_transactions.csv and products.csv from the project folder.
- Users may also upload their own CSV files.

##### 2. Preprocess Data
- FILL OUT

##### 3. Run Mining
- On the Association Rules page, users set minimum support, confidence, and lift values.
- Running the analysis calls a backend function that is (FILL OUT)

##### 4. Query Results
- Once backend algorithms are implemented, mined rules will appear in the interface.
- The Shopping Simulator page will also support recommendations tied to mined rules.



#### Algorithm Implementation

##### Apriori
FILL OUT
[2-3 sentences on your implementation approach]
- Data structure: [e.g., dictionary of itemsets]
- Candidate generation: [breadth-first, level-wise]
- Pruning strategy: [minimum support]

##### Eclat
FILL OUT
[2-3 sentences on your implementation approach]
- Data structure: [e.g., TID-set representation]
- Search strategy: [depth-first]
- Intersection method: [set operations]

##### CLOSET
FILL OUT
[2-3 sentences on your implementation approach]
- Data structure: [e.g., FP-tree / prefix tree]
- Mining approach: [closed itemsets only]
- Closure checking: [method used]



#### Performance Results
FILL OUT
Tested on provided dataset (80-100 transactions after cleaning):

| Algorithm | Runtime (ms) | Rules Generated | Memory Usage |
|-----------|--------------|-----------------|--------------|
| Apriori   | [value]      | [value]         | [value]      |
| Eclat     | [value]      | [value]         | [value]      |
| CLOSET    | [value]      | [value]         | [value]      |

**Parameters**: min_support = 0.2, min_confidence = 0.5

**Analysis**: FILL OUT
[1-2 sentences explaining performance differences]



#### Project Structure

```
project-root/
├── streamlit_app.py         # Streamlit frontend (Nicole)
├── backend.py               # Mining algorithms (Edwin)
├── sample_transactions.csv
├── products.csv
└── README.md

```



#### Data Preprocessing
FILL OUT
Issues handled:
- Empty transactions: [count] removed
- Single-item transactions: [count] removed
- Duplicate items: [count] instances cleaned
- Case inconsistencies: [count] standardized
- Invalid items: [count] removed
- Extra whitespace: trimmed from all items



#### Testing
FILL OUT
Verified functionality:
- [✓] CSV import and parsing
- [✓] All preprocessing operations
- [✓] Three algorithm implementations
- [✓] Interactive query system
- [✓] Performance measurement

Test cases:
- [Describe 2-3 key test scenarios]



#### Known Limitations

[List any known issues or constraints, if applicable]



#### AI Tool Usage

[Required: 1 paragraph describing which AI tools you used and for what purpose]

ChatGPT was used to assist with UI development in Streamlit, debugging session state behavior, adding item deletion features, and generating template code for backend.py. All generated code was reviewed, edited, and integrated manually to fit the project requirements



#### References

- Course lecture materials
- Streamlit documentation
- Pandas documentation
