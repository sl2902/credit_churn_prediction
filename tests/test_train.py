import os
import sys
import pickle
import numpy as np
# sys.path.append("scripts")
# import settings
from scripts import preprocess_data
import pandas as pd
from deepdiff import DeepDiff
pd.set_option('display.max_columns', None)

PICKLE_PATH = os.getenv("PICKLE_PATH", "pickle")

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

try:
    dv = load_pickle(f'{PICKLE_PATH}/dv.pkl')
except FileNotFoundError:
    dv = load_pickle('dv.pkl')
try:
    oe = load_pickle(f'{PICKLE_PATH}/ohe.pkl')
except FileNotFoundError:
    oe = load_pickle('ohe.pkl')

def test_data_prep():
    """Test data_prep function"""

    cols = ["client_num", "attrition_flag",	"customer_age", "gender", "dependent_count",
            "education_level", "marital_status", "income_category", "card_category",	
            "months_on_book", "total_relationship_count", "credit_limit", "total_revolving_bal"
            ]
    input_data = [
        [714726858,"Attrited Customer",31,"F",0,"Post-Graduate","Divorced","Less than $40K","Blue",36,6,4871,0],
        [709749483,"Existing Customer",40,"F",2,"Graduate","Unknown","Less than $40K","Blue",29,5,2636,1953],
        [710926383,"Existing Customer",32,"F",2,"Post-Graduate","Married","$40K - $60K","Blue",24,5,1553,1177]
    ]

    input_df = pd.DataFrame(input_data, columns=cols)
    x_df, y_df, _, _ = preprocess_data.data_prep(
        input_df, 
        oe,
        dv,
        is_train=False,
        is_drop=False,
        is_pred=False,
        is_ohe=False
    )
    # print(x_df)
    expected_cols = ["card_category", "credit_limit", "customer_age", "dependent_count",
                     "education_level", "gender=F", "gender=M", "income_category", "marital_status",
                     "months_on_book", "total_relationship_count", "total_revolving_bal"
                 ]
    x_expected_df = pd.DataFrame([
        [0.,4871.,31.,0.,4.,1.,0.,4.,0.,36.,6.,0.],
        [0.,2636.,40.,2.,2.,1.,0.,4.,3.,29.,5.,1953.],
        [0.,1553.,32.,2.,4.,1.,0.,1.,1.,24.,5.,1177.]
    ], columns=expected_cols
    )
    
    y_expected_df = pd.Series([1, 0, 0])

    x_diff = DeepDiff(
        x_df.reset_index(drop=True).to_dict(),
        x_expected_df.reset_index(drop=True).to_dict(),
        significant_digits=1,
    )

    print(f"x_diff= {x_diff}")
    assert not x_diff

    y_diff = DeepDiff(y_expected_df, y_df)
    print(f"y_diff= {y_diff}")
    assert not y_diff

test_data_prep()