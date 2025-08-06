# Synthetic Data Privacy Evaluation Framework

A comprehensive framework for evaluating the privacy protection of synthetic health claims data using traditional privacy metrics and advanced privacy attacks.

## 📋 Overview

This repository contains a complete evaluation framework for assessing the privacy protection of synthetic health data. It implements both traditional privacy metrics (k-anonymity, l-diversity, t-closeness) and advanced privacy attacks (Attribute Inference Attack, Membership Inference Attack) to provide a comprehensive privacy assessment.

## 🏗️ Project Structure

```
eval-synth-data-privacy/
├── src/                           # Main source code
│   ├── data_processing/           # Data processing pipeline
│   │   ├── data_pipeline.py      # Main pipeline orchestrator
│   │   ├── preprocessing_before_join.py
│   │   ├── join_by_year_with_stats.py
│   │   ├── preprocess_joined_tables.py
│   │   ├── combine_yearly_joins.py
│   │   └── join_all_by_year.py
│   ├── database_creation/         # Database creation from CSV files
│   │   ├── create_databases.py   # Main database creation
│   │   ├── join_tables.py        # Table joining operations
│   │   └── create_minimal_db.py  # Minimal database creation
│   ├── attacks/                   # Privacy attack implementations
│   │   ├── aia.py               # Attribute Inference Attack
│   │   ├── mia.py               # Membership Inference Attack
│   │   ├── run_aia.py           # AIA execution script
│   │   └── run_mia.py           # MIA execution script
│   ├── duckdb_manager/           # Database management
│   │   └── duckdb_manager.py    # DuckDB connection and operations
│   ├── calculate_privacy_metrics.py  # Traditional privacy metrics
│   ├── distance_metrics.py          # Distance-based privacy metrics
│   ├── feature_preprocessing.py     # Feature preprocessing utilities
│   └── pickle_to_duckdb.py          # Data format conversion
├── results/                       # Analysis results
│   ├── aia_attacks/              # Attribute Inference Attack results
│   ├── mia_attacks/              # Membership Inference Attack results
│   ├── privacy_calculator/       # Traditional privacy metrics results
│   └── distance_metrics/         # Distance-based metrics results
├── diagrams/                      # Project documentation
│   └── erm.md                    # Entity Relationship Model
├── duckdb/                       # Database files and utilities
├── logs/                         # Application logs
└── testdata/                     # Test datasets
```

## 🎯 Key Features

### 🔒 Privacy Metrics
- **Traditional Metrics**: k-anonymity, l-diversity, t-closeness
- **Distance-based Metrics**: Distance to Closest Record (DCR), Nearest Neighbor Distance Ratio (NNDR)
- **Comprehensive Analysis**: Multi-table analysis with configurable quasi-identifiers and sensitive attributes

### 🛡️ Privacy Attacks
- **Attribute Inference Attack (AIA)**: Evaluates the ability to infer sensitive attributes using partial quasi-identifier knowledge
- **Membership Inference Attack (MIA)**: Determines if a record was part of the training dataset
- **Multiple Attack Strategies**: k-NN, Random Forest, distance-based approaches
- **Comprehensive Evaluation**: Success rates, risk assessment, and detailed reporting

### 📊 Data Processing
- **Health Claims Data**: Specialized processing for insurance claims data
- **Multi-year Analysis**: Support for temporal data analysis
- **Database Management**: Efficient DuckDB-based data storage and querying
- **Feature Preprocessing**: Medical code encoding, timestamp conversion, missing value handling

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install duckdb pandas numpy scikit-learn scipy matplotlib seaborn
```

### Basic Usage

#### 1. Create Database from CSV Files

```python
from src.database_creation import create_database

# Create database from CSV files
db_path = create_database("path/to/csv/files")
```

#### 2. Calculate Privacy Metrics

```python
from src.calculate_privacy_metrics import PrivacyMetricsCalculator

# Initialize calculator
calculator = PrivacyMetricsCalculator()

# Run analysis
calculator.run_analysis()
```

#### 3. Run Privacy Attacks

```python
from src.attacks.aia import AttributeInferenceAttack
from src.attacks.mia import MembershipInferenceAttack

# Attribute Inference Attack
aia = AttributeInferenceAttack()
results = aia.evaluate_aia_vulnerability(
    original_df, synthetic_df, 
    quasi_identifiers, sensitive_attributes
)

# Membership Inference Attack
mia = MembershipInferenceAttack()
results = mia.run_membership_inference_attack(
    training_data, holdout_data, synthetic_data, feature_columns
)
```

## 📈 Privacy Metrics

### Traditional Privacy Metrics

#### K-Anonymity
Measures the minimum group size for quasi-identifier combinations.

```python
# Example quasi-identifiers for health data
qi_inpatient = ['age_group', 'gender', 'region', 'admission_type']
qi_outpatient = ['age_group', 'gender', 'region', 'visit_type']
qi_drugs = ['age_group', 'gender', 'atc_group']

# Calculate k-anonymity
results = calculator.calculate_k_anonymity(df, qi_inpatient)
```

#### L-Diversity
Ensures diversity of sensitive attributes within quasi-identifier groups.

```python
# Example sensitive attributes
sensitive_attributes = ['diagnosis_code', 'procedure_code', 'drug_atc']

# Calculate l-diversity
results = calculator.calculate_l_diversity(df, qi_inpatient, sensitive_attributes)
```

#### T-Closeness
Measures distribution similarity of sensitive attributes using Wasserstein distance.

```python
# Calculate t-closeness
results = calculator.calculate_t_closeness(df, qi_inpatient, sensitive_attributes, t_threshold=0.15)
```

### Distance-Based Privacy Metrics

#### Distance to Closest Record (DCR)
Measures the distance between synthetic records and their closest original counterparts.

#### Nearest Neighbor Distance Ratio (NNDR)
Ratio of distances to nearest neighbors, indicating privacy protection level.

## 🛡️ Privacy Attacks

### Attribute Inference Attack (AIA)

Evaluates the ability to infer sensitive attributes using partial knowledge of quasi-identifiers.

**Attack Strategies:**
- **Partial Knowledge Attack**: Uses k-nearest neighbors with limited quasi-identifier knowledge
- **Machine Learning Attack**: Uses Random Forest classifier for attribute inference

**Key Features:**
- Configurable knowledge ratios (0.1 to 1.0)
- Multiple sensitive attribute support
- Comprehensive vulnerability assessment
- Risk scoring and interpretation

```python
from src.attacks.aia import AttributeInferenceAttack

aia = AttributeInferenceAttack()
results = aia.evaluate_aia_vulnerability(
    original_df=original_data,
    synthetic_df=synthetic_data,
    quasi_identifiers=['age_group', 'gender', 'region'],
    sensitive_attributes=['diagnosis_code'],
    sample_size=10000
)
```

### Membership Inference Attack (MIA)

Determines if a specific record was part of the training dataset used to generate synthetic data.

**Methodology:**
1. Calculate distances from real records to nearest synthetic records
2. Optimize classification threshold using holdout data
3. Evaluate attack success using known membership labels

**Key Features:**
- Distance-based approach using nearest neighbors
- Multiple distance metrics (Euclidean, Manhattan)
- Threshold optimization using various metrics
- Comprehensive performance evaluation

```python
from src.attacks.mia import MembershipInferenceAttack

mia = MembershipInferenceAttack()
results = mia.run_membership_inference_attack(
    training_data=training_data,
    holdout_data=holdout_data,
    synthetic_data=synthetic_data,
    feature_columns=['age', 'gender', 'diagnosis'],
    target_sample_per_set=40000
)
```

## 📊 Data Processing Pipeline

### Health Claims Data Structure

The framework is designed for health claims data with the following structure:

- **Insurants**: Basic patient information
- **Insurance Data**: Insurance coverage details
- **Inpatient Cases**: Hospital admission records
- **Inpatient Diagnosis**: Diagnosis codes for inpatient cases
- **Inpatient Procedures**: Procedure codes for inpatient cases
- **Inpatient Fees**: Fee information for inpatient cases
- **Outpatient Cases**: Outpatient visit records
- **Outpatient Diagnosis**: Diagnosis codes for outpatient cases
- **Outpatient Procedures**: Procedure codes for outpatient cases
- **Outpatient Fees**: Fee information for outpatient cases
- **Drugs**: Prescription drug information

### Data Processing Steps

1. **Preprocessing**: Handle missing values, encode medical codes, convert timestamps
2. **Joining**: Combine tables by year or across all years
3. **Feature Engineering**: Create derived features and encode categorical variables
4. **Analysis**: Apply privacy metrics and attacks

## 🔧 Configuration

### Privacy Metrics Configuration

```python
# Configure quasi-identifiers and sensitive attributes
qi_inpatient = ['age_group', 'gender', 'region', 'admission_type']
qi_outpatient = ['age_group', 'gender', 'region', 'visit_type']
qi_drugs = ['age_group', 'gender', 'atc_group']

sensitive_attributes = [
    'diagnosis_code', 'procedure_code', 'drug_atc',
    'admission_diagnosis', 'discharge_diagnosis'
]

# Initialize calculator with configuration
calculator = PrivacyMetricsCalculator(
    qi_inpatient=qi_inpatient,
    qi_outpatient=qi_outpatient,
    qi_drugs=qi_drugs,
    sensitive_attributes=sensitive_attributes,
    sample_size=5000000
)
```

### Attack Configuration

```python
# AIA Configuration
aia_config = {
    'knowledge_ratio': 0.7,  # Percentage of quasi-identifiers known
    'k_neighbors': 5,        # Number of neighbors for k-NN
    'sample_size': 10000     # Sample size for attack evaluation
}

# MIA Configuration
mia_config = {
    'target_sample_per_set': 40000,  # Sample size per dataset
    'distance_metric': 'euclidean',  # Distance metric
    'optimization_metric': 'f1'      # Threshold optimization metric
}
```

## 📁 Output and Results

### Results Structure

```
results/
├── aia_attacks/              # Attribute Inference Attack results
│   ├── 2017_KR_0.7/         # Results for 2017 data, 70% knowledge ratio
│   └── 2017_KR_1.0/         # Results for 2017 data, 100% knowledge ratio
├── mia_attacks/              # Membership Inference Attack results
│   ├── mia_results_*.json   # Attack results
│   └── mia_distances_*.png  # Distance distribution plots
├── privacy_calculator/       # Traditional privacy metrics
│   └── privacy_metrics_*.json
└── distance_metrics/         # Distance-based metrics
    └── distance_metrics_*.json
```

### Result Interpretation

#### Privacy Metrics Results
- **K-Anonymity**: Higher k-values indicate better privacy protection
- **L-Diversity**: Higher l-values indicate greater attribute diversity
- **T-Closeness**: Lower t-values indicate better distribution similarity

#### Attack Results
- **Success Rate**: Lower success rates indicate better privacy protection
- **Risk Level**: Categorized as Low, Medium, High, or Critical
- **Performance Metrics**: Precision, recall, F1-score, AUC-ROC

## 🛠️ Development

### Adding New Privacy Metrics

```python
def calculate_new_metric(self, df: pd.DataFrame, parameters: Dict) -> Dict:
    """Calculate a new privacy metric."""
    # Implementation here
    return results
```

### Adding New Attack Strategies

```python
def new_attack_strategy(self, data: pd.DataFrame, parameters: Dict) -> Dict:
    """Implement a new attack strategy."""
    # Implementation here
    return attack_results
```

## 📚 Dependencies

### Core Dependencies
- `duckdb` - Database operations
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning algorithms
- `scipy` - Statistical functions

### Visualization
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization

### Optional Dependencies
- `pycanon` - External privacy metrics library



## 📄 License

This project is licensed under the MIT License 


## 🔬 Research Context

This framework was developed for evaluating synthetic health data privacy protection in the context of medical research and healthcare analytics. It provides comprehensive tools for assessing the privacy risks associated with synthetic data generation methods.

---

**Note**: This framework is designed for research purposes and should be used in compliance with relevant data protection regulations and ethical guidelines.
