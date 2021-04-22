import AIEA
from skimage import io

if __name__ == "__main__":
    oAIEA = AIEA.CAiea(mode='train')
    oImage = io.imread("./Sample.png")
    oAIEA.Setting(AIEA.eSettingCmd.eSettingCmd_IMAGE_DATA, oImage)
    oAIEA.Write()