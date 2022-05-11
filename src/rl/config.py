class RL_Config(object):
    def __init__(self, args) -> None:
        super().__init__()

        if args.target == 'ATPG_PC' or args.target == 'ATPG_TC':
            ###################################################
            # ATPG Setting
            ###################################################
            print('[CONFIG-INFO] Load ATPG RL Setting')
            # RL model and training
            self.OBSERVE = 100. # timesteps to observe before training
            self.REPLAY_MEMORY = 1e6 # number of previous transitions to remember
            self.BATCH_SIZE = 8 # size of minibatch
            self.GAMMA = 0.5 # decay rate of past observations
            self.UPDATE_TIME = 50
            self.EACH_TRAIN_TIMES = 100
            self.SAVE_ROUND_GAP = -1
            self.RANDOM_ACTION = 100

            # Test Point
            self.MAX_CP_CNT = 30
            self.gate2num = args.gate_to_index
            self.TESTABILITY_THRO = 0.1

            # ATPG
            self.tc_error = 0.95

        else:
            ###################################################
            # LBIST Setting
            ###################################################
            print('[CONFIG-INFO] Load LBIST RL Setting')
            # RL model and training
            self.OBSERVE = 50. # timesteps to observe before training
            self.REPLAY_MEMORY = 1000 # number of previous transitions to remember
            self.BATCH_SIZE = 8 # size of minibatch
            self.GAMMA = 0.5 # decay rate of past observations
            self.UPDATE_TIME = 50
            self.EACH_TRAIN_TIMES = 30
            self.SAVE_ROUND_GAP = -1
            self.RANDOM_ACTION = 50

            # Test Point
            self.MAX_CP_CNT = 30
            self.gate2num = args.gate_to_index
            self.TESTABILITY_THRO = 0.1

