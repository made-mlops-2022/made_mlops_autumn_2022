from pydantic import BaseModel, Field


class Patient(BaseModel):
    age: int = Field(..., ge=0, le=150, 
                     description='age in years')
    
    sex: int = Field(..., ge=0, le=1, 
                     description='sex (1 = male; 0 = female)')
    
    cp: int = Field(..., ge=0, le=3, 
                    description='chest pain type (0 = typical angina\n' + \
                                '1: atypical angina\n2: non-anginal pain\n '+ \
                                '3: asymptomatic)')
    
    trestbps: int = Field(..., ge=0, le=400, 
                          description='resting blood pressure ' + \
                                      '(in mm Hg on admission to the hospital)')
    
    chol: int = Field(..., ge=0, le=600, 
                      description='serum cholestoral in mg/dl')
    
    fbs: int = Field(..., ge=0, le=1, 
                     description='fasting blood sugar > 120 mg/dl (1 = true; 0 = false')
    
    restecg: int = Field(..., ge=0, le=2, 
                         description=' resting electrocardiographic results' + \
                                     '0: normal\n' + \
                                     '1: having ST-T wave abnormality (T wave inversions ' + \
                                     'and/or ST elevation or depression of > 0.05 mV)\n' + \
                                     '2: showing probable or definite left ventricular ' + \
                                     'hypertrophy by Estes criteria')
    
    thalach: int = Field(..., ge=0, le=300, 
                         description='maximum heart rate achieved')
    
    exang: int = Field(..., ge=0, le=1, 
                       description='exercise induced angina (1 = yes, 0 = no)')
    
    oldpeak: float = Field(..., ge=0, le=7, 
                           description='ST depression induced by exercise relative to rest')
    
    slope: int = Field(..., ge=0, le=2, 
                       description='the slope of the peak exercise ST segment\n' + \
                                    '0: upsloping\n1: flat\n2: downsloping')
    
    ca: int = Field(..., ge=0, le=3, 
                    description='number of major vessels (0-3) colored by flourosopy')
    
    thal: int = Field(..., ge=0, le=2, description='0 = normal\n1 = fixed defect\n2 = reversable defect')
