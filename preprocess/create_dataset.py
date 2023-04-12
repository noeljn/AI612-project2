import pandas as pd
import numpy as np
import os
import pickle
import warnings
import easydict
from collections import Counter

warnings.filterwarnings( 'ignore' )

targets = ['mortality_prediction_short', 
           'mortality_prediction_long', 
           'readmission_prediction',
           'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9',
           'd10', 'd11', 'd12', 'd13', 'd14', 'd15', 'd16', 'd17',
           'los_short',
           'los_long',
           'final_acuity',
           'imminent_discharge',
           'creatinine_level',
           'bilirubin_level',
           'platelet_level',
           'white_blood_cell_level'
           ]

# Create MIMIC-III dataset
def create_MIMIC_dataset(input_path, label_path):
    patient_path = os.path.join(input_path, 'PATIENTS.csv')
    icustay_path = os.path.join(input_path, 'ICUSTAYS.csv')
    
    patients = pd.read_csv(patient_path)
    icus = pd.read_csv(icustay_path)
    print('length of PATIENTS.csv  : ', len(patients))
    print('length of ICUSTAYS.csv  : ', len(icus))
    
    temp = icus[(icus['FIRST_CAREUNIT'] == icus['LAST_CAREUNIT'])]

    temp['INTIME'] = pd.to_datetime(temp['INTIME'], infer_datetime_format=True)
    temp['OUTTIME'] = pd.to_datetime(temp['OUTTIME'], infer_datetime_format=True)

    patients['DOB'] = pd.to_datetime(patients['DOB'], infer_datetime_format=True)
    patients['DOD'] = pd.to_datetime(patients['DOD'], infer_datetime_format=True)
    patients['DOD_HOSP'] = pd.to_datetime(patients['DOD_HOSP'], infer_datetime_format=True)
    patients['DOD_SSN'] = pd.to_datetime(patients['DOD_SSN'], infer_datetime_format=True)

    small_patients = patients[patients.SUBJECT_ID.isin(temp.SUBJECT_ID)]
    temp = temp.merge(small_patients, on='SUBJECT_ID', how='left')

    datediff = np.array(temp.INTIME.dt.date) - np.array(temp.DOB.dt.date)
    age = np.array([x.days // 365 for x in datediff])
    temp['age'] = age
    temp = temp[temp.age >= 18]
    print('length of temp  :', len(temp))

    cohort_mm = temp.reset_index(drop=True).copy()

    # finally editted
    cohort_mm["dataset"] = np.nan
    cohort_mm["dataset"] = cohort_mm["dataset"].fillna("mimiciii")

    labels = os.path.join(label_path, 'mimiciii_labels.csv')
    labels_df = pd.read_csv(labels)
    cohort_mm_rev = pd.merge(cohort_mm, labels_df, how = "inner", on = "ICUSTAY_ID")
    for idx, target in enumerate(targets):
        label_list = []
        for i,row in cohort_mm_rev.iterrows():
            label_list.append(int(row['labels'][1:-1].split(',')[idx]))
        cohort_mm_rev[target] = label_list

    pickle.dump(cohort_mm_rev, open(os.path.join(input_path, 'mimiciii_cohort.pk'), 'wb'), -1)

# create mimic4
def create_MIMIC4_dataset(input_path, label_path):
    patient_path = os.path.join(input_path, 'patients.csv')
    icustay_path = os.path.join(input_path, 'icustays.csv')

    patients = pd.read_csv(patient_path)
    icus = pd.read_csv(icustay_path)
    print('length of PATIENTS.csv  : ', len(patients))
    print('length of ICUSTAYS.csv  : ', len(icus))

    temp = icus[(icus['first_careunit'] == icus['last_careunit'])]

    temp['intime'] = pd.to_datetime(temp['intime'], infer_datetime_format=True)
    temp['outtime'] = pd.to_datetime(temp['outtime'], infer_datetime_format=True)

    patients['dod'] = pd.to_datetime(patients['dod'], infer_datetime_format=True)

    small_patients = patients[patients.subject_id.isin(temp.subject_id)]
    temp = temp.merge(small_patients, on='subject_id', how='left')
    temp['age'] = temp['anchor_age']
    print('length of temp  :', len(temp))

    cohort_mm = temp.reset_index(drop=True).copy()
    
    cohort_mm["dataset"] = np.nan
    cohort_mm["dataset"] = cohort_mm["dataset"].fillna("mimiciv")

    labels = os.path.join(label_path, 'mimiciv_labels.csv')
    labels_df = pd.read_csv(labels)
    cohort_mm_rev = pd.merge(cohort_mm, labels_df, how = "inner", on = "stay_id")
    for idx, target in enumerate(targets):
        label_list = []
        for i,row in cohort_mm_rev.iterrows():
            label_list.append(int(row['labels'][1:-1].split(',')[idx]))
        cohort_mm_rev[target] = label_list

    pickle.dump(cohort_mm_rev, open(os.path.join(input_path, 'mimiciv_cohort.pk'), 'wb'), -1)

# Create eICU dataset
def create_eICU_dataset(input_path, label_path):
    patient_path = os.path.join(input_path, 'patient.csv')
    patient_df = pd.read_csv(patient_path)

    print('Unique patient unit stayid : ', len(set(patient_df.patientunitstayid)))

    micu = patient_df

    null_index =micu[micu['age'].isnull()==True].index
    micu.loc[null_index, 'age'] = 1
    micu = micu.replace('> 89', 89)

    micu.loc[:, 'age'] = micu.loc[:, 'age'].astype('int')
    micuAge = micu[micu.age >= 18]

    cohort_ei = micuAge.copy().reset_index(drop=True)
    #cohort_ei = eicu_diagnosis_label(cohort_ei)
    #cohort_ei = cohort_ei[cohort_ei['diagnosis'] !=float].reset_index(drop=True)
    cohort_ei = cohort_ei.reset_index(drop=True)

    cohort_ei["dataset"] = np.nan
    cohort_ei["dataset"] = cohort_ei["dataset"].fillna("eicu")

    labels = os.path.join(label_path, 'eicu_labels.csv')
    labels_df = pd.read_csv(labels)
    cohort_ei_rev = pd.merge(cohort_ei, labels_df, how = "inner", on = "patientunitstayid")
    for idx, target in enumerate(targets):
        label_list = []
        for i,row in cohort_ei_rev.iterrows():
            label_list.append(int(row['labels'][1:-1].split(',')[idx]))
        cohort_ei_rev[target] = label_list

    pickle.dump(cohort_ei_rev, open(os.path.join(input_path, 'eicu_cohort.pk'), 'wb'), -1)


