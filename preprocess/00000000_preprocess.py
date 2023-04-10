from create_dataset import create_MIMIC_dataset, create_MIMIC4_dataset, create_eICU_dataset
from dataframe_gen import preprocess
from numpy_convert import convert2numpy
from preprocess_utils import label_npy_file
import os
import argparse

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
        help="output directory"
    )
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--min_length', type=int, default=5)
    parser.add_argument('--window_time',  type=int, default=12)
    parser.add_argument('--data_type', type=str, choices=['MICU', 'TotalICU'], default='MICU')
    
    return parser

def main(args):
    """
    TODO:
        Implement your feature preprocessing function here.
        Rename the file name with your student number.
    
    Note:
        This script should dump processed features to the --dest directory.
        Note that --dest directory will be an input to your dataset class (i.e., --data_path).
        You can dump any type of files such as json, cPickle, or whatever your dataset can handle.
    """
    root = arg.root
    
    args = get_parser().parse_args()
    # file names
    mimic_csv_files = {'lab':'LABEVENTS', 
                        'med':'PRESCRIPTIONS',
                        'inf': 'INPUTEVENTS'}
    
    mimic4_csv_files = {'lab':'labevents', 
                        'med':'prescriptions',
                        'inf': 'inputevents'}

    eicu_csv_files = {'lab':'lab', 
                    'med':'medication',
                    'inf':'infusionDrug'}

    # definition file name        
    
    mimic_def_file = {'LABEVENTS':'D_LABITEMS', 
                    'INPUTEVENTS_CV':'D_ITEMS', 
                    'INPUTEVENTS_MV':'D_ITEMS',
                    'labevents':'d_labitems', 
                    'inputevents':'d_items'}

    # columns_map
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

    item_list = ['lab','med', 'inf']
    wd = os.getcwd()
    print('working directory .. : ', wd)

    create_MIMIC_dataset(os.path.join(root, 'mimiciii'))
    create_MIMIC4_dataset(os.path.join(root, 'mimiciv'))
    create_eICU_dataset(os.path.join(root, 'eicu'))
     

    #root_dir = args.root
    #dest_dir = args.dest
    
    preprocess(root, 
                    item_list,
                   csv_files_dict, 
                   columns_map_dict, 
                   issue_map,
                   mimic_def_file,
                   args.max_length,
                   args.data_type) 
    
    print('preprocess finish!!')

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
