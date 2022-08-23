from sweep_work.create_sweep import create_sweep
from sweep_work.sweep_logger import SweepLogger
from sweep_work.experiment_wrappers import wrap_experiment

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging

import train_hpo
import wandb

class MyExperiment(experiment.AbstractIterativeExperiment):
    # ...

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Skip for Quickguide
        pass

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        print("printing from iterate function")
        objective = train_hpo.train(cw_config['params'])
        print("objective in interate function: " + str(objective))
        #wandb.log({"objective": objective})
        return {"objective": objective}
    
    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        # Skip for Quickguide
        pass
    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass

if __name__ == "__main__":
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(wrap_experiment(MyExperiment))
    #cw = cluster_work.ClusterWork(MyExperiment)
    # this next line is important in order to create a new sweep!
    create_sweep(cw)
    # Sweepwork expects that 1 SweepLogger is present. Additional other loggers should not be a problem
    cw.add_logger(SweepLogger())

    # RUN!
    cw.run()