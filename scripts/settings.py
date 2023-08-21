from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction import DictVectorizer

# store global constants and object instances used across the pipeline

# EXPERIMENT_NAME = "credit-churn-experiment"
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
# EVIDENTLY_SERVICE_ADDRESS = "http://localhost:9878"
# MONGODB_ADDRESS = "mongodb://127.0.0.1:27017"
# PICKLE_PATH = "pickle"
# TEST = True

cols_to_keep = [
    "CLIENTNUM",
    "Attrition_Flag",
    "Customer_Age",
    "Dependent_count",
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
    "Months_on_book",
    "Total_Relationship_Count",
    "Credit_Limit",
    "Total_Revolving_Bal",
]

col_rename = {
    "CLIENTNUM": "client_num",
    "Attrition_Flag": "attrition_flag",
    "Customer_Age": "customer_age",
    "Dependent_count": "dependent_count",
    "Gender": "gender",
    "Education_Level": "education_level",
    "Marital_Status": "marital_status",
    "Income_Category": "income_category",
    "Card_Category": "card_category",
    "Months_on_book": "months_on_book",
    "Total_Relationship_Count": "total_relationship_count",
    "Credit_Limit": "credit_limit",
    "Total_Revolving_Bal": "total_revolving_bal",
}

# cat fields
cat_fields = ["education_level", "marital_status", "income_category", "card_category"]

# subset of fields used for Evidently monitoring
# check config.yaml in the monitoring directory
monitor = [
    'education_level',
    'income_category',
    'credit_limit',
    'total_revolving_bal',
    'customer_age',
]

dv = DictVectorizer(sparse=False)
# oe = OrdinalEncoder(encoded_missing_value=-1)
oe = OrdinalEncoder()
