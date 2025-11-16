### Interactive Supermarket Simulation with Association Rule Mining

#### Author Information

- **Name**: [Your Full Name]
- **Student ID**: [Your ID]
- **Course**: CAI 4002 - Artificial Intelligence
- **Semester**: [Fall/Spring Year]



#### System Overview

[2-3 sentences describing what your application does]



#### Technical Stack

- **Language**: [Python 3.x / JavaScript / Java]
- **Key Libraries**: [List main dependencies]
- **UI Framework**: [If applicable]



#### Installation

##### Prerequisites
- [e.g., Python 3.8+, Node.js 14+, Java 11+]
- [Other requirements]

##### Setup
```bash
# Clone or extract project
cd [project-directory]

# Install dependencies
[command to install dependencies]

# Run application
[command to start application]
```



#### Usage

##### 1. Load Data
- **Manual Entry**: Click items to create transactions
- **Import CSV**: Use "Import" button to load `sample_transactions.csv`

##### 2. Preprocess Data
- Click "Run Preprocessing"
- Review cleaning report (empty transactions, duplicates, etc.)

##### 3. Run Mining
- Set minimum support and confidence thresholds
- Click "Analyze" to execute all three algorithms
- Wait for completion (~1-3 seconds)

##### 4. Query Results
- Select product from dropdown
- View associated items and recommendation strength
- Optional: View technical details (raw rules, performance metrics)



#### Algorithm Implementation

##### Apriori
[2-3 sentences on your implementation approach]
- Data structure: [e.g., dictionary of itemsets]
- Candidate generation: [breadth-first, level-wise]
- Pruning strategy: [minimum support]

##### Eclat
[2-3 sentences on your implementation approach]
- Data structure: [e.g., TID-set representation]
- Search strategy: [depth-first]
- Intersection method: [set operations]

##### CLOSET
[2-3 sentences on your implementation approach]
- Data structure: [e.g., FP-tree / prefix tree]
- Mining approach: [closed itemsets only]
- Closure checking: [method used]



#### Performance Results

Tested on provided dataset (80-100 transactions after cleaning):

| Algorithm | Runtime (ms) | Rules Generated | Memory Usage |
|-----------|--------------|-----------------|--------------|
| Apriori   | [value]      | [value]         | [value]      |
| Eclat     | [value]      | [value]         | [value]      |
| CLOSET    | [value]      | [value]         | [value]      |

**Parameters**: min_support = 0.2, min_confidence = 0.5

**Analysis**: [1-2 sentences explaining performance differences]



#### Project Structure

```
project-root/
├── src/
│   ├── algorithms/
│   │   ├── apriori.[py/js/java]
│   │   ├── eclat.[py/js/java]
│   │   └── closet.[py/js/java]
│   ├── preprocessing/
│   │   └── cleaner.[py/js/java]
│   ├── ui/
│   │   └── [interface files]
│   └── main.[py/js/java]
├── data/
│   ├── sample_transactions.csv
│   └── products.csv
├── README.md
├── REPORT.pdf
└── [requirements.txt / package.json / pom.xml]
```



#### Data Preprocessing

Issues handled:
- Empty transactions: [count] removed
- Single-item transactions: [count] removed
- Duplicate items: [count] instances cleaned
- Case inconsistencies: [count] standardized
- Invalid items: [count] removed
- Extra whitespace: trimmed from all items



#### Testing

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

Example:
"Used ChatGPT for explaining Eclat algorithm vertical representation and debugging file parsing errors. Used GitHub Copilot for generating UI boilerplate code. All generated code was reviewed, tested, and adapted for this specific implementation."



#### References

- Course lecture materials
- [Algorithm papers or resources consulted]
- [Library documentation links]