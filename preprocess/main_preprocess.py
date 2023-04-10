import argparse
import sys
sys.path.append("DescEmb-master/preprocess")
sys.path.append("DescEmb-master/trainers")
sys.path.append("DescEmb-master/datasets")
from create_dataset import create_MIMIC_dataset, create_eICU_dataset
from dataframe_gen import preprocess
from numpy_convert import convert2numpy
from preprocess_utils import label_npy_file
import os

import torch
import torch.multiprocessing as mp
from trainers import Trainer, Word2VecTrainer

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

    mimic_csv_files = {'lab':'LABEVENTS', 
                        'med':'PRESCRIPTIONS',
                        'inf': 'INPUTEVENTS'}

    eicu_csv_files = {'lab':'lab', 
                    'med':'medication',
                    'inf':'infusionDrug'}

    # definition file name        
    mimic_def_file = {'LABEVENTS':'D_LABITEMS', 
                    'INPUTEVENTS_CV':'D_ITEMS', 
                    'INPUTEVENTS_MV':'D_ITEMS'}

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
                'medication': 
                    ['Yes'],          
                }
    

    csv_files_dict = {'mimic':mimic_csv_files, 
                        'eicu':eicu_csv_files
    }
    columns_map_dict = {'mimic':mimic_columns_map, 
                           'eicu':eicu_columns_map
    }
    item_list = ['lab','med', 'inf']
    wd = os.getcwd()
    print('working directory .. : ', wd)

    src_list = ['eicu']

    #preprocess(args.root, src_list, item_list, csv_files_dict, columns_map_dict, issue_map, mimic_def_file)
    
    args = argparse.Namespace()
    args.batch_size = 128
    args.value_embed_type = 'nonconcat'
    args.data = "train/"
    args.save_dir = "checkpoint"
    args.save_prefix = "checkpoint"
    args.epoch = 10

    trainer = Word2VecTrainer(args)
    trainer.train()


    
if __name__ == "__main__":
    #parser = get_parser()
    #args = parser.parse_args()
    args = argparse.Namespace()
    args.root = "train/"
    args.dest = "output/"

    main(args)