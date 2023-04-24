import argparse
import os
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
import datetime
import re
import random
import torch
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings( 'ignore' )

def get_parser():
    """
    Note:
        Do not add command-line arguments here when you submit the codes.
        Keep in mind that we will run your pre-processing code by this command:
        `python 00000000_preprocess.py ./train --dest ./output`
        which means that we might not be able to control the additional arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        metavar="DIR",
        help="root directory containing different ehr files to pre-process (usually, 'train/')"
    )
    parser.add_argument(
        "--dest",
        type=str,
        metavar="DIR",
        default=None,
        help="output directory"
    )

    parser.add_argument(
        "--sample_filtering",
        type=bool,
        default=True,
        help="indicator to prevent filtering from being applies to the test dataset."
    )
    return parser

def main(args):
    """
    TODO:
        Implement your feature preprocessing function here.
        Rename the file name with your student number.
    
    Note:
        1. This script should dump processed features to the --dest directory.
        Note that --dest directory will be an input to your dataset class (i.e., --data_path).
        You can dump any type of files such as json, cPickle, or whatever your dataset can handle.

        2. If you use vocabulary, you should specify your vocabulary file(.pkl) in this code section.
        Also, you must submit your vocabulary file({student_id}_vocab.pkl) along with the scripts.
        Example:
            with open('./20231234_vocab.pkl', 'rb') as f:
                (...)

        3. For fair comparison, we do not allow to filter specific samples when using test dataset.
        Therefore, if you filter some samples from the train dataset,
        you must use the '--sample_filtering' argument to prevent filtering from being applied to the test dataset.
        We will set the '--sample_filtering' argument to False and run the code for inference.
        We also check the total number of test dataset.
    """

    root_dir = args.root
    dest_dir = args.dest

    mimic_csv_files = {'lab':'LABEVENTS', 
                        'med':'PRESCRIPTIONS',
                        'inf': 'INPUTEVENTS'}

    mimic4_csv_files = {'lab':'labevents', 
                        'med':'prescriptions',
                        'inf': 'inputevents'}

    eicu_csv_files = {'lab':'lab', 
                    'med':'medication',
                    'inf':'infusionDrug'}
    
    mimic_def_file = {'LABEVENTS':'D_LABITEMS', 
                    'INPUTEVENTS_CV':'D_ITEMS', 
                    'INPUTEVENTS_MV':'D_ITEMS',
                    'labevents':'d_labitems', 
                    'inputevents':'d_items'}

    mimic_columns_map = {'LABEVENTS':
                            {'HADM_ID':'ID',
                            'CHARTTIME':'code_time',
                            'ITEMID':'code_name',
                            'VALUENUM':'value',
                            'VALUEUOM':'uom',
                            'FLAG':'issue'
                            },
                        'PRESCRIPTIONS':
                            {'HADM_ID':'ID',
                            'STARTDATE':'code_time',
                            'DRUG':'code_name', 
                            'ROUTE':'route', 
                            'PROD_STRENGTH':'prod',
                            'DOSE_VAL_RX':'value',
                            'DOSE_UNIT_RX':'uom',
                            },                                      
                        'INPUTEVENTS': 
                            {'HADM_ID':'ID',
                            'CHARTTIME':'code_time', 
                            'ITEMID':'code_name',
                            'RATE':'value', 
                            'RATEUOM':'uom',
                            'STOPPED':'issue'
                            }
    }

    mimic4_columns_map = {'labevents':
                            {'hadm_id':'ID',
                            'charttime':'code_time',
                            'itemid':'code_name',
                            'valuenum':'value',
                            'valueuom':'uom',
                            'flag':'issue'
                            },
                        'prescriptions':
                            {'hadm_id':'ID',
                            'starttime':'code_time',
                            'drug':'code_name', 
                            'route':'route', 
                            'prod_strength':'prod',
                            'dose_val_rx':'value',
                            'dose_unit_rx':'uom',
                            },                                      
                        'inputevents': 
                            {'hadm_id':'ID',
                            'starttime':'code_time', 
                            'itemid':'code_name',
                            'rate':'value', 
                            'rateuom':'uom',
                            'stopped':'issue'
                            }
    }

    eicu_columns_map =  {'lab':
                            {'patientunitstayid':'ID', 
                            'labresultoffset':'code_offset',
                            'labname':'code_name',
                            'labresult':'value',
                            'labmeasurenamesystem':'uom'
                            },
                        'medication':
                            {'patientunitstayid':'ID',
                            'drugstartoffset':'code_offset',
                            'drugname':'code_name', 
                            'routeadmin':'route',
                            'ordercancelled':'issue'
                            },      
                        'infusionDrug':
                            {'patientunitstayid':'ID',
                            'infusionoffset':'code_offset',
                            'drugname':'code_name',
                            'infusionrate':'value'
                            }
    }

    issue_map = {'LABEVENTS': 
                        ['abnormal'],                            
                    'INPUTEVENTS':
                        ['Restart',
                        'NotStopd', 
                        'Rewritten', 
                        'Changed', 
                        'Paused', 
                        'Flushed', 
                        'Stopped'
                        ] ,
                    'MEDICATION': 
                        ['Yes'],
                    'labevents': 
                        ['abnormal'],                            
                    'inputevents':
                        ['Restart',
                        'NotStopd', 
                        'Rewritten', 
                        'Changed', 
                        'Paused', 
                        'Flushed', 
                        'Stopped'
                        ] ,
                    'medication': 
                        ['Yes'],          
                    }
    csv_files_dict = {'mimiciii':mimic_csv_files, 
                        'mimiciv':mimic4_csv_files,
                        'eicu':eicu_csv_files
        }

    columns_map_dict = {'mimiciii':mimic_columns_map, 
                        'mimiciv':mimic4_columns_map,
                        'eicu':eicu_columns_map
    }

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
    

    def create_MIMIC_dataset(input_path, label_path):
        patient_path = os.path.join(input_path, 'mimiciii/PATIENTS.csv')
        icustay_path = os.path.join(input_path, 'mimiciii/ICUSTAYS.csv')

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

    def create_MIMIC4_dataset(input_path, label_path):
        patient_path = os.path.join(input_path, 'mimiciv/patients.csv')
        icustay_path = os.path.join(input_path, 'mimiciv/icustays.csv')
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

    def create_eICU_dataset(input_path, label_path):
        patient_path = os.path.join(input_path, 'eicu/patient.csv')
        patient_df = pd.read_csv(patient_path)
        print('Unique patient unit stayid : ', len(set(patient_df.patientunitstayid)))
        micu = patient_df
        null_index =micu[micu['age'].isnull()==True].index
        micu.loc[null_index, 'age'] = 1
        micu = micu.replace('> 89', 89)
        micu.loc[:, 'age'] = micu.loc[:, 'age'].astype('int')
        micuAge = micu[micu.age >= 18]
        cohort_ei = micuAge.copy().reset_index(drop=True)
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

    def mimic_inf_merge(file, input_path, src):
        df_inf_mv_mm = pd.read_csv(os.path.join(input_path, src,'INPUTEVENTS_MV'+'.csv'), skiprows=lambda i: i>0 and random.random() > 0.01)
        df_inf_cv_mm = pd.read_csv(os.path.join(input_path, src, 'INPUTEVENTS_CV'+'.csv'), skiprows=lambda i: i>0 and random.random() > 0.01)
        df_inf_mv_mm['CHARTTIME'] = df_inf_mv_mm['STARTTIME']
        df_inf_mm = pd.concat([df_inf_mv_mm, df_inf_cv_mm], axis=0).reset_index(drop=True)
        print('mimic INPUTEVENTS merge!') 
        
        return df_inf_mm
    
    def eicu_med_revise(file, input_path, src):
        df = pd.read_csv(os.path.join(input_path, src, file+'.csv'), skiprows=lambda i: i>0 and random.random() > 0.01)
        df['split'] = df['dosage'].apply(lambda x: str(re.sub(',', '',str(x))).split())
        def second(x):
            try:
                if len(pd.to_numeric(x))>=2:
                    x = x[1:]
                return x
            except ValueError:
                return x

        df['split'] = df['split'].apply(second).apply(lambda s:' '.join(s))
        punc_dict = str.maketrans('', '', '.-')
        df['uom'] = df['split'].apply(lambda x: re.sub(r'[0-9]', '', x))
        df['uom'] = df['uom'].apply(lambda x: x.translate(punc_dict)).apply(lambda x: x.strip())
        df['uom'] = df['uom'].apply(lambda x: ' ' if x=='' else x)
        
        def hyphens(s):
            if '-' in str(s):
                s = str(s)[str(s).find("-")+1:]
            return s
        df['value'] = df['split'].apply(hyphens)
        df['value'] = df['value'].apply(lambda x: [float(s) for s in re.findall(r'-?\d+\.?\d*', x)])
        df['value'] = df['value'].apply(lambda x: x[-1] if len(x)>0 else x)
        df['value'] = df['value'].apply(lambda d: str(d).replace('[]',' '))

        return df

    def eicu_inf_revise(file, input_path, src):
        df = pd.read_csv(os.path.join(input_path, src, file+'.csv'), skiprows=lambda i: i>0 and random.random() > 0.01 )
        df['split'] = df['drugname'].apply(lambda x: str(x).rsplit('(', maxsplit=1))
        def addp(x):
            if len(x)==2:
                x[1] = '(' + str(x[1])
            return x

        df['split'] = df['split'].apply(addp)
        df['split']=df['split'].apply(lambda x: x +[' '] if len(x)<2 else x)

        df['drugname'] = df['split'].apply(lambda x: x[0])
        df['uom'] = df['split'].apply(lambda x: x[1])
        df['uom'] = df['uom'].apply(lambda s: s[s.find("("):s.find(")")+1])

        toremove = ['()','', '(Unknown)', '(Scale B)', '(Scale A)',  '(Human)', '(ARTERIAL LINE)']

        df['uom'] = df['uom'].apply(lambda uom: ' ' if uom in toremove else uom)
        df = df.drop('split',axis=1)
        
        testing = lambda x: (str(x)[-1].isdigit()) if str(x)!='' else False
        code_with_num = list(pd.Series(df.drugname.unique())[pd.Series(df.drugname.unique()).apply(testing)==True])
        add_unk = lambda s: str(s)+' [UNK]' if s in code_with_num else s
        df['drugname'] = df['drugname'].apply(add_unk)
        
        return df

    def column_rename(df, columns_map):
        df = df.rename(columns_map, axis='columns')

        return df
    
    def issue_delete(df, csv_file, issue_map):
        if 'issue' in df.columns:
            issue_label = issue_map[csv_file]
            df.drop(df[df['issue'].isin(issue_label)].index, inplace=True)

        return df
    
    def name_dict(df, csv_file, input_path, src, mimic_def_file):
        if csv_file in mimic_def_file:
            dict_name= mimic_def_file[csv_file]
            dict_path = os.path.join(input_path, src, dict_name+'.csv')
            code_dict = pd.read_csv(dict_path)
            if src == "mimiciii":
                key = code_dict['ITEMID']
                value = code_dict['LABEL']
            elif src == "mimiciv":
                key = code_dict['itemid']
                value = code_dict['label']
            code_dict = dict(zip(key,value))
            df['code_name'] = df['code_name'].map(code_dict)

        return df

    def null_convert(df):
        df = df.fillna(' ')

        return df
    
    def ID_filter(df_icu, df):
        return df[df['ID'].isin(df_icu['ID'])]

    def time_filter(df_icu, df, source, data_type):
        time_delta = datetime.timedelta(hours=12)
        if source =='mimiciii': 
            df = pd.merge(df, df_icu[['ID', 'INTIME', 'OUTTIME']], how='left', on='ID')
            df = df[df['code_time']!=' ']
            if 'INTIME_x' in df.columns:
                df['INTIME'] = df['INTIME_x']
            for col_name in ['code_time', 'INTIME', 'OUTTIME']:
                df[col_name] = pd.to_datetime(df[col_name])
            if data_type =='MICU':  
                df['INTIME+12hr'] = df['INTIME'] + time_delta
                df = df[(df['code_time']> df['INTIME']) & (df['code_time'] < df['OUTTIME']) & (df['code_time'] < df['INTIME+12hr'])]
            elif data_type =='TotalICU':
                df = df[(df['code_time']> df['INTIME']) & (df['code_time'] < df['OUTTIME'])]
    
            df['code_offset'] = df['code_time'] - df['INTIME']
            df['code_offset'] = df['code_offset'].apply(lambda x : x.seconds//60, 4)

        elif source =='mimiciv': 
            df = pd.merge(df, df_icu[['ID', 'intime', 'outtime']], how='left', on='ID')
            df = df[df['code_time']!=' ']
            for col_name in ['code_time', 'intime', 'outtime']:
                df[col_name] = pd.to_datetime(df[col_name])
            if data_type =='MICU':  
                df['intime+12hr'] = df['intime'] + time_delta
                df = df[(df['code_time']> df['intime']) & (df['code_time'] < df['outtime']) & (df['code_time'] < df['intime+12hr'])]
            elif data_type =='TotalICU':
                df = df[(df['code_time']> df['intime']) & (df['code_time'] < df['outtime'])]
    
            df['code_offset'] = df['code_time'] - df['intime']
            df['code_offset'] = df['code_offset'].apply(lambda x : x.seconds//60, 4)


        elif source =='eicu':
            if data_type =='MICU':
                df = df[(df['code_offset']> 0 ) | (df['code_offset'] < 12*60)]    
            elif data_type =='TotalICU':
                df = df[df['code_offset']> 0]   
        return df            

  
    def min_length(df, min_length):
        df = df[df['code_name'].map(type) ==list]
        df['code_length'] = df['code_name'].map(len)
        df = df[df['code_length']>=min_length]
        df = df.drop('code_length', axis=1)

        return df


    def offset2order(offset_seq):
        offset_set = set(offset_seq)

        dic = {}
        for idx, offset in enumerate(list(offset_set)):
            dic[offset] = idx
        
        def convert(x):
            return dic[x]
        
        order_seq = list(map(convert, offset_seq))

        return order_seq

    
    def text2idx(seq, vocab):
        def convert(x):
            return vocab[x]
        
        return seq.apply(lambda x : convert(x))


    def merge_df(df_lab, df_med, df_inf):
        df_merge = pd.concat([df_lab, df_med, df_inf], axis=0)
        df_merge = df_merge[['ID', 'table_name', 'code_name', 'value', 'uom', 'code_offset']]
            
        return df_merge 


    def list_prep(df, df_icu):
        column_list = ['table_name', 'code_name', 'code_offset', 'value', 'uom']
        df_agg = df.groupby(['ID']).agg({column: lambda x: x.tolist() for column in column_list})
        df = pd.merge(df_icu, df_agg, how='left', on=['ID'])

        return df


    def making_vocab(df):
        vocab_dict = {}
        vocab_dict['[PAD]'] = 0
        
        df['merge_code_set'] = df['code_name'].apply(lambda x : list(set(x)))
        vocab_set = []
        for codeset in df['merge_code_set']:
            vocab_set.extend(codeset) 
        vocab_set = list(set(vocab_set))
        for idx, vocab in enumerate(vocab_set):
            vocab_dict[vocab] = idx+1
            
        return vocab_dict


    def ID_rename(df_icu, src):
        if src =='mimiciii' : 
            icu_ID = 'HADM_ID'
        elif src =='mimiciv' : 
            icu_ID = 'hadm_id'
        elif src=='eicu':
            icu_ID = 'patientunitstayid'
            
        df_icu['ID'] = df_icu[icu_ID]
        df_icu = df_icu.drop(columns=icu_ID)

        return df_icu

    def pad(sequence, max_length):
        if len(sequence) > max_length:

            return sequence[:max_length]
        else:
            pad_length = max_length-len(sequence)
            zeros = list(np.zeros(pad_length))
            sequence.extend(zeros) 

            return sequence

    def sampling(sequence, walk_len, max_length):
        seq_len = len(sequence)
        seq_index_start = [i*walk_len  for i in range(((seq_len-max_length)//walk_len)+1)]
        
        return [sequence[i:(i+max_length)] for i in seq_index_start]

    def sortbyoffset(df):
        print('sortbyoffset')
        sorted = df.sort_values(['ID', 'code_offset'], ascending=[0,1])
        return sorted

    def preprocess(input_path,
                    item_list,
                    csv_files_dict, 
                    columns_map_dict, 
                    issue_map, 
                    mimic_def_file,
                    max_length,
                    data_type):

        for src in ['mimiciii', 'eicu', 'mimiciv']:
            df_icu = pd.read_pickle(os.path.join(input_path, f'{src}_cohort.pk'))
            df_icu = ID_rename(df_icu, src)
            for item in item_list:
                print('data preparation initialization .. {} {}'.format(src, item))
                file = csv_files_dict[src][item]
                columns_map = columns_map_dict[src][file] # the files from mimic that we want
                if src =='mimiciii' and item =='inf':
                    df = mimic_inf_merge(file, input_path, src)
                elif src=='eicu' and item=='med':
                    df = eicu_med_revise(file, input_path, src)
                elif src=='eicu' and item=='inf':
                    df = eicu_inf_revise(file, input_path, src)
                elif src =='mimiciv' and item == 'lab':
                    df = pd.read_csv(os.path.join(input_path, src, file+'.csv'))
                    df = df.drop(columns='value')
                    df = df.reset_index(drop=True)
                else:
                    df = pd.read_csv(os.path.join(input_path, src, file+'.csv'))
                print('df_load ! .. {} {}'.format(src, item))

                df = column_rename(df, columns_map)
                df = issue_delete(df, file, issue_map)
                df = name_dict(df, file, input_path, src, mimic_def_file)
                df = null_convert(df)
                df = ID_filter(df_icu, df)
                df = time_filter(df_icu, df, src, data_type)

                if item == 'lab':
                    lab = df.copy()
                    lab['table_name'] = np.nan
                    lab["table_name"] = lab["table_name"].fillna("lab")
                elif item =='med':
                    med = df.copy()
                    med['table_name'] = np.nan
                    med["table_name"] = med["table_name"].fillna("med")
                    #med = med_align(src, med)
                elif item =='inf':
                    inf = df.copy()
                    inf['table_name'] = np.nan
                    inf["table_name"] = inf["table_name"].fillna("inf")
            del(df)
            print('data preparation finish for three items \n second preparation start soon..')

            df = merge_df(lab, med ,inf)
            print('lab med inf three categories merged in one!')

            df = sortbyoffset(df)
            print('sortbyoffset finish!')
            df = list_prep(df, df_icu)
            print('list_prep finish!')
            df = min_length(df, 5).reset_index(drop=True)
            df['code_order'] = df['code_offset'].map(lambda x : offset2order(x))  
            df['seq_len'] = df['code_name'].map(len)

            print('column prepare!')
            column_list = ['table_name', 'code_name', 'code_offset', 'value', 'uom', 'code_order']
            if data_type == 'MICU':
                for column in column_list:
                    df[column] = df[column].map(lambda x : pad(x, max_length))
            
            elif data_type == 'TotalICU':
                df_short = df[df['seq_len'] <= max_length].reset_index(drop=True)
                df_long = df[df['seq_len'] > max_length].reset_index(drop=True)
            
                for i, column in enumerate(column_list):
                    df_short[column] = df[column].map(lambda x : pad(x, max_length))
                    df_long[column] = df[column].map(lambda x: sampling(x, max_length//3, max_length))

                df_long = df_long.explode(column_list).reset_index(drop=True)
                df = pd.concat([df_short, df_long], axis=0).reset_index(drop=True)
                del df_short, df_long

            if src =='mimiciii':
                df['ID'] = list(df['ICUSTAY_ID'])

            if src =='mimiciv':
                df['ID'] = list(df['stay_id'])
                df = df.drop_duplicates(['ID', 'seq_len'])
                df = df.sort_values('ID')
                df = df.reset_index(drop=True)

            print('Preprocessing completed.')
            print('Writing', '{}_df.pkl'.format(src), 'to', input_path)
            df.to_pickle(os.path.join(input_path,'{}_df.pkl'.format(src)))
        print('Writing', 'pooled_df.pkl', 'to', input_path)
        df_mm = pd.read_pickle(os.path.join(input_path,'mimiciii_df.pkl'.format(src)))
        df_m4 = pd.read_pickle(os.path.join(input_path,'mimiciv_df.pkl'.format(src)))
        df_ei = pd.read_pickle(os.path.join(input_path,'eicu_df.pkl'.format(src)))
        df_pooled = pd.concat((df_mm,df_ei), axis=0).reset_index(drop=True)
        df_pooled_all = pd.concat((df_pooled,df_m4), axis=0).reset_index(drop=True)
        df_pooled_final = df_pooled_all[['dataset', 'table_name', 'ID','age', 'code_name', 'code_offset', 'value', 'uom', 'code_order', 'seq_len', 'mortality_prediction_short',
        'mortality_prediction_long', 'readmission_prediction', 'd1', 'd2', 'd3',
        'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'd11', 'd12', 'd13', 'd14',
        'd15', 'd16', 'd17', 'los_short', 'los_long', 'final_acuity',
        'imminent_discharge', 'creatinine_level', 'bilirubin_level',
        'platelet_level', 'white_blood_cell_level']]
        df_pooled_final.to_pickle(os.path.join(input_path,'pooled_df.pkl'))
        print('finished Writing', 'pooled_df.pkl', 'to', input_path)
        del df_mm, df_ei, df_m4, df_pooled, df_pooled_all, df_pooled_final

    input_path = root_dir
    label_path = os.path.join(input_path, 'labels')

    item_list = ['lab','med', 'inf']

    create_MIMIC_dataset(input_path, label_path)
    create_MIMIC4_dataset(input_path, label_path)
    create_eICU_dataset(input_path, label_path)
    
    max_length = 150
    data_type = 'MICU'
    
    preprocess(input_path, 
                    item_list,
                    csv_files_dict, 
                    columns_map_dict, 
                    issue_map,
                    mimic_def_file,
                    max_length,
                    data_type)

    print('Start Embedding!!') 

    with open(os.path.join(input_path, 'pooled_df.pkl'), 'rb') as f:
        pooled = pickle.load(f)

    p = re.compile("[a-zA-Z'0-9]+")
    seq_len = 100

    final_list = []
    for txt_id in tqdm(range(len(pooled))):
        final_list_one_id = []
        events_one_id = []
        for i in range(len(pooled['code_name'][txt_id])):
            if pooled['code_name'][txt_id][i] != 0.0:
                event_list = []
                event_list.append(str(pooled['table_name'][txt_id][i]))
                event_list.append(str(pooled['code_name'][txt_id][i]))
                event_list.append(str(pooled['code_offset'][txt_id][i]))
                event_list.append(str(pooled['value'][txt_id][i]))
                event_list.append(str(pooled['uom'][txt_id][i]))
                event_list.append(str(pooled['code_order'][txt_id][i]))
            else:
                continue
            events_one_id.append(event_list)
        events_one_id.sort(key=lambda x:x[-1])
        for n in range(len(events_one_id)):
            events_one_id[n].pop()
            string = ','.join(events_one_id[n])
            text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", string)
            text = p.findall(text.lower())
            final_list_one_id.append(text)
        
        if len(final_list_one_id) < seq_len:
            pad_seq_len = seq_len - len(final_list_one_id)
            for iter in range(pad_seq_len):
                    final_list_one_id.append(['pad'])
        else:
            final_list_one_id = final_list_one_id[:100]
        final_list.append(final_list_one_id)
    with open(os.path.join(input_path, 'final_dataset_in_word.pkl'), 'wb') as f:
        pickle.dump(final_list, f)
    with open(os.path.join(input_path, 'final_dataset_in_word.pkl'), 'rb') as f:
        final_list_pickle = pickle.load(f)

    model_name = "google/bert_uncased_L-2_H-128_A-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def embed(input_text):
        tokens = tokenizer(input_text, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        embeddings = model(**tokens).last_hidden_state
        pooled_embeddings = embeddings.mean(dim=1)
        return pooled_embeddings
    seq_len = 100
    final_list = final_list_pickle

    final_embed = []
    for k in tqdm(range(len(final_list))):
        df_embed = []
        for n in range(seq_len):
            text_features = embed(' '.join(final_list[k][n]))
            df_embed.append(text_features[0])
            del text_features
        final_one_id_tensor = torch.cat(df_embed, dim=0)
        final_one_id_cpu = final_one_id_tensor.detach().cpu()
        final_embed.append(final_one_id_cpu)
        del final_one_id_tensor
    final_embed = torch.stack(final_embed, dim=0)
    
    print('Start saving embedded features!!')    
    torch.save(final_embed, os.path.join(input_path, 'features.pkl'))
    print('Finished saving embedded features!!')
    
    print('Start saving labels!!')
    labels = []
    labels.append(list(pooled.mortality_prediction_short))
    labels.append(list(pooled.mortality_prediction_long))
    labels.append(list(pooled.readmission_prediction))
    labels.append(list(pooled.d1))
    labels.append(list(pooled.d2))
    labels.append(list(pooled.d3))
    labels.append(list(pooled.d4))
    labels.append(list(pooled.d5))
    labels.append(list(pooled.d6))
    labels.append(list(pooled.d7))
    labels.append(list(pooled.d8))
    labels.append(list(pooled.d9))
    labels.append(list(pooled.d10))
    labels.append(list(pooled.d11))
    labels.append(list(pooled.d12))
    labels.append(list(pooled.d13))
    labels.append(list(pooled.d14))
    labels.append(list(pooled.d15))
    labels.append(list(pooled.d16))
    labels.append(list(pooled.d17))
    labels.append(list(pooled.los_short))
    labels.append(list(pooled.los_long))
    labels.append(list(pooled.final_acuity))
    labels.append(list(pooled.imminent_discharge))
    labels.append(list(pooled.creatinine_level))
    labels.append(list(pooled.bilirubin_level))
    labels.append(list(pooled.platelet_level))
    labels.append(list(pooled.white_blood_cell_level))
    labels_rev = np.array(labels)
    final_label = labels_rev.swapaxes(0,1)
    ffinal_label = final_label.tolist()
    torch.save(ffinal_label, os.path.join(input_path, 'labels.pkl'))
    print('Finished saving labels!!')

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)