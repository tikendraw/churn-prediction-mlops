import pandas as pd
from pydantic import BaseModel, field_validator, ValidationError
from typing import List, Optional, Literal
import numpy as np

class Customer(BaseModel):
    RowNumber: int
    CustomerId: int
    Surname: str
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Exited: int

    @field_validator('Age')
    def age_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Age must be positive")
        return v


dtypes = {
    'integer': [np.dtypes.Int64DType, np.dtypes.Int32DType, np.dtypes.Int16DType, np.dtypes.Int8DType, np.dtypes.UInt64DType, np.dtypes.UInt32DType, np.dtypes.UInt16DType, np.dtypes.UInt8DType],
    'number': [float, np.dtypes.Float16DType, np.dtypes.Float32DType, np.dtypes.Float64DType],
    'string': [str, np.dtypes.ObjectDType , np.dtypes.StringDType],
    'boolean': [bool, np.dtypes.BoolDType],
    'datetime': [np.dtypes.TimeDelta64DType],
    'categorical': [pd.CategoricalDtype],
}

def map_type(x):
    for key, value in dtypes.items():
        if x in value:
            return key

        

def schema_validation(csv_filepath: str, data_class: BaseModel = Customer, **kwargs) -> bool:
    try:
        df = pd.read_csv(csv_filepath, **kwargs) 

        fs = {col : map_type(type(df[col].dtype)) for col in df.columns}
        pp = data_class.model_json_schema()['properties']
        ss = {k:v['type'] for k,v in pp.items()}

        for i in pp:
            if not  fs[i]==ss[i] :
                assert False, f"Type mismatch for column {i}"
        
        return True
    
    except (ValidationError, pd.errors.ParserError, ValueError, AssertionError) as e:  
        print(f"Validation Error: {e}")
        return False
