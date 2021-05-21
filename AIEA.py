from enum import IntEnum
import imp, os
import numpy as np

class eSettingCmd(IntEnum):
    eSettingCmd_NONE = 1
    eSettingCmd_IMAGE_DATA = 2
    eSettingCmd_IMAGE_CHANNEL = 3
    eSettingCmd_SAVE_VIDEO = 4

class CAiea():
    def __init__(self, episodes, drl_model='dqn', mode='eval', kpt_model='sift'):
        self.__strCkptPath = "./checkpoints/"
        if(not os.path.exists(self.__strCkptPath)):
            os.mkdir(self.__strCkptPath)
        
        if(drl_model == 'dqn'):
            self.__oDrlModule = imp.load_source(drl_model, "./model/dqn.py")
        self.__oModel = self.__oDrlModule.CModel(num_episodes=episodes, keypoint_detection=kpt_model)
        
    def Setting(self, eCommand:int, Value=None):
        eCmd = eSettingCmd(eCommand)
        if(eCmd == eSettingCmd.eSettingCmd_IMAGE_DATA):
            self.__oImage = Value
            if(len(self.__oImage.shape) < 3):
                self.__oImage = np.expand_dims(np.asarray(self.__oImage), axis = 0)
        if(eCmd == eSettingCmd.eSettingCmd_SAVE_VIDEO):
            if(Value == "True" or Value == "true" or Value == 1):
                self.__bSaveVideo = True
            else:
                self.__bSaveVideo = False

    def Write(self):
        self.__oModel.Setting(image=self.__oImage, video=self.__bSaveVideo, checkpoint_path=self.__strCkptPath)
        self.__oModel.Train()
        self.__oModel.Reset()

    def Read(self):
        self.__oModel.Setting(image=self.__oImage, video=self.__bSaveVideo, checkpoint_path=self.__strCkptPath)
        self.__oModel.Test()
        self.__oModel.Reset()
