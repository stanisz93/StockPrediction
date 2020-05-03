
BATCH_SIZE = 100
HISTORY = 20
EPOCHS = 30
PO_PARAMETERS = 2
COMPONENTS = 3


PARAMS = {'hidden_size': 50,
          'batch_size': BATCH_SIZE,
          'optimizer': "adam",
         'history_steps': HISTORY,
         'epochs': EPOCHS}


class PROBLEM:
    UNIVARIATE = "univariate"


problem_config = {PROBLEM.UNIVARIATE: PARAMS}