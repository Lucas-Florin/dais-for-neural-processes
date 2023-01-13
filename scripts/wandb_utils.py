import wandb
import json
import pandas as pd
import copy
from wandb2numpy import export_data
from wandb2numpy.config_loader import load_config
from wandb2numpy.filtering import get_filtered_runs

default_base_config = {
    'entity': 'lucas-florin',
    'project': 'NP_baseline',    
}

def get_file_from_run(run, file_name):
    for file in run.files():
        if file_name in file.name:
            return file
    return None
        
def get_table_from_run(run, file_name):
    file = get_file_from_run(run, file_name)
    if file is None:
        return None
    wrapper = file.download(replace=True, root='../plots/')
    file_contents = json.load(wrapper)
    df = pd.DataFrame(**file_contents)
    return df

def get_objective_tasks_list(run):
    mc_task_list = get_table_from_run(run, 'mc_objective_tasks_list')
    ais_task_list = get_table_from_run(run, 'ais_objective_tasks_list')
    dais_task_list = get_table_from_run(run, 'dais_objective_tasks_list')
    dais_new_task_list = get_table_from_run(run, 'dais_new_objective_tasks_list')
    return (
        mc_task_list,
        ais_task_list,
        dais_task_list,
        dais_new_task_list
    )
    
def get_run(name, group, job_type, base_config=None, verbose=True):
    base_config = default_base_config if base_config is None else base_config
    assert base_config.keys() == default_base_config.keys()
    config = copy.deepcopy(base_config)
    config['groups'] = [group]
    config['job_types'] = [[job_type]]
    
    api = wandb.Api(timeout=15)
    run_list = get_filtered_runs(config, api)
    if verbose:
        print("Found following runs that match the filters:")
        for run in run_list:
            print(run.name)
    for run in run_list:
        if name in run.name:
            if verbose:
                print("Selecting the follwoing run: ")
                print(run.name)
            return run
    if verbose:
        print('No run matches the given name. ')
    return None