import AIEA
from skimage import io
import argparse, os, sys
from common.Log import DebugPrint
from glob import glob

parser = argparse.ArgumentParser(description='Active Image Enhancing Agent (AIEC) for Keypoint Detection')
parser.add_argument('--mode', '-o', type=str, default='eval', dest='mode',
                    help='Mode select: eval, train')
parser.add_argument('--db', '-d', type=str, dest='db', default=None,
                    help='DB path for TRAINING mode')
parser.add_argument('--query', '-q', type=str, dest='query', 
                    help='Image query file path for EVAL mode')

args = parser.parse_args()


def readFolder(strImgFolder):
    if(not os.path.isdir(strImgFolder)):
        log.DebugPrint().warning("Path does not exist!")
        return False
    strPngList = [os.path.basename(x) for x in glob(strImgFolder + "*.png")]
    strJpgList = [os.path.basename(x) for x in glob(strImgFolder + "*.jpg")]
    strFileList = strPngList + strJpgList
    strFileList.sort()
    return strFileList

def Training(model, file_list):
    for fileIdx in file_list:
        strImgPath = args.db + '/' + fileIdx
        oImage = io.imread(strImgPath)
        model.Setting(AIEA.eSettingCmd.eSettingCmd_IMAGE_DATA, oImage)
        model.Write()


if __name__ == "__main__":
    oAIEA = AIEA.CAiea(mode=args.mode)
    if(args.mode == 'train'):
        if(args.db == None):
            DebugPrint().error("Error: No DB Path for Training")
            sys.exit()
        strFileList = readFolder(args.db)
        Training(oAIEA, strFileList)
        

    