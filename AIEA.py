from enum import IntEnum
import imp
import numpy as np

class eSettingCmd(IntEnum):
    eSettingCmd_NONE = 1
    eSettingCmd_IMAGE_DATA = 2
    eSettingCmd_IMAGE_CHANNEL = 3
    eSettingCmd_CONFIG = 4

class CAiea():
    def __init__(self, drl_model='dqn', mode='eval'):
        self.__uActionsNum = 2                   # Action = {contrast up, contrast down}
        if(drl_model == 'dqn'):
            self.__oDrlModule = imp.load_source(drl_model, "./model/dqn.py")
            
        if(mode == 'train'):
            self.__oTrain = self.__oDrlModule.CTrain(drl_module=self.__oDrlModule, num_episodes=50)

    def Setting(self, eCommand:int, Value=None):
        eCmd = eSettingCmd(eCommand)
        if(eCmd == eSettingCmd.eSettingCmd_IMAGE_DATA):
            self.__oImage = Value
            if(len(self.__oImage.shape) < 3):
                self.__oImage = np.expand_dims(np.asarray(self.__oImage), axis = 0)

    def Write(self):
        self.__oTrain.Setting(self.__oImage)
        self.__oTrain.Run()

    def Read(self):
        print("Read")