import train
from enum import IntEnum

class eSettingCmd(IntEnum):
    eSettingCmd_NONE = 1
    eSettingCmd_IMAGE_DATA = 2
    eSettingCmd_IMAGE_CHANNEL = 3
    eSettingCmd_CONFIG = 4

class CAiea():
    def __init__(self, drl_model = 'dqn'):
        if(drl_model == 'dqn'):
            self.__oDrlModule = imp.load_source(drl_model, "./model/dqn.py")
            self.__oDrlModel = self.__oDrlModule.RLnet()

    def Setting(self, eCommand:int, Value=None):
        eCmd = eSettingCmd(eCommand)
        if(eCmd == eSettingCmd.eSettingCmd_IMAGE_DATA):
            self.__Image = Value

    def Write(self):
        oTrain = train.CTrain()
        oTrain.Setting(self.__oDrlModel)
        oTrain.Train()
