import nets, imp

class CTrain():
    def __init__(self, learning_rate = 0.001):
        self.__strDevice = "cuda" if torch.cuda.is_available() else "cpu"
        self.__fLearningRate learning_rate

    def Setting(self, drl_model):
        self.__oModel = nets.CAieaNets().to(self.__strDevice)
        self.__oDrlModel = drl_model

    def Train(self):
        
