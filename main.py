# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# standard modules
from pprint import pprint
from logging import getLogger, StreamHandler, Formatter, INFO, DEBUG
# third party modules

# original modules
from hp.hp import HP
from experiment.experiment import Experiment
from trainer.sample_trainer import Hartmann6Trainer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logger = getLogger(__name__)
    logger.setLevel(INFO)

    # ストリームハンドラ作成
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter("[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s] %(message)s"))

    # ロガーに追加
    logger.addHandler(stream_handler)

    logger.info(f"Read Hyper Parameter file.")
    hp = HP(name="hartmann6_test", file_path="params/sample_param_hartmann6.yaml", logger=logger)

    pprint(hp.to_dict())

    # check hp
    logger.info(f"Check contents of Hyper Param.")
    hp.check_params(Hartmann6Trainer.required_params)

    # prepare and exec the experiment
    logger.info(f"Exec Experiment.")
    experiment = Experiment(hp, Hartmann6Trainer, logger=logger)
    experiment.run()

    best_result_x, best_result_y = experiment.get_best_result()
    best_result_x = best_result_x.detach().numpy().tolist() if best_result_x is not None else "No variable exists"
    best_result_y = best_result_y.detach().numpy().tolist()[0]
    logger.info(
        f'The best result of the experiment is (x:{hp.get_var_dict_from_tensor(best_result_x)}, y:{best_result_y}.'
    )

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
