# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from hp.hp import HP
from experiment.experiment import Experiment
from trainer.sample_trainer import Hartmann6Trainer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hp = HP(name="hartmann6_test", file_path="params/sample_param_hartmann6.yaml")
    experiment = Experiment(hp, Hartmann6Trainer)

    experiment.run()

    print(experiment.get_best_result())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
