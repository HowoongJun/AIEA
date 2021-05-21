import AIEA
from skimage import io
import argparse, os, sys
from common.Log import DebugPrint
from glob import glob
from common.utils import CudaStatus

parser = argparse.ArgumentParser(description='Active Image Enhancing Agent (AIEC) for Keypoint Detection')
parser.add_argument('--mode', '-o', type=str, default='test', dest='mode',
                    help='Mode select: test, train')
parser.add_argument('--method', '-m', type=str, default='b', dest='method',
                    help='Method select: a, b')
parser.add_argument('--db', '-d', type=str, dest='db', default=None,
                    help='DB path for TRAINING mode')
parser.add_argument('--query', '-q', type=str, dest='query', default=None,
                    help='Image query file path for "test" mode')
parser.add_argument('--video', '-v', type=bool, dest='video',
                    help='Save video (True/False)')
parser.add_argument('--episode', '-e', type=int, dest='episode',
                    help='Total episode number for training')
parser.add_argument('--keypoint', '-k', type=str, dest='keypoint',
                    help='Select keypoint detection algorithm (sift, orb)')

args = parser.parse_args()

def readFolder(strImgFolder):
    if(not os.path.isdir(strImgFolder)):
        DebugPrint().warning("Path does not exist!")
        return False
    strPngList = [os.path.basename(x) for x in glob(strImgFolder + "*.png")]
    strJpgList = [os.path.basename(x) for x in glob(strImgFolder + "*.jpg")]
    strFileList = strPngList + strJpgList
    strFileList.sort()
    return strFileList

def trainDRL(model, file_list):
    for fileIdx in file_list:
        DebugPrint().info("Image: " + str(fileIdx))
        strImgPath = args.db + '/' + fileIdx
        oImage = io.imread(strImgPath, as_gray=True) * 255
        model.Setting(AIEA.eSettingCmd.eSettingCmd_IMAGE_DATA, oImage)
        model.Setting(AIEA.eSettingCmd.eSettingCmd_SAVE_VIDEO, args.video)
        model.Write()

def testDRL(model, file_list):
    for fileIdx in file_list:
        strImgPath = args.query + '/' + fileIdx
        DebugPrint().info("Read file .. " + fileIdx)
        oImage = io.imread(strImgPath, as_gray=True) * 255
        model.Setting(AIEA.eSettingCmd.eSettingCmd_IMAGE_DATA, oImage)
        model.Setting(AIEA.eSettingCmd.eSettingCmd_SAVE_VIDEO, args.video)
        model.Read()

if __name__ == "__main__":
    oAIEA = AIEA.CAiea(mode=args.mode, method=args.method, episodes=args.episode, kpt_model=args.keypoint)
    if(args.mode == 'train'):
        if(args.db == None):
            DebugPrint().error("No DB path for training")
            sys.exit()
        strFileList = readFolder(args.db + "/")
        trainDRL(oAIEA, strFileList)

    elif(args.mode == 'test'):
        if(args.query == None):
            DebugPrint().error("No query path for testing")
            sys.exit()
        strFileList = readFolder(args.query)
        DebugPrint().info("Test mode start - video: " + str(args.video))
        testDRL(oAIEA, strFileList)