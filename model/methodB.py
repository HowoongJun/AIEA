import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
import cv2, random, torch, math, os
from common.Log import DebugPrint
from skimage.transform import resize
from common.utils import CudaStatus
import random

class RLnet(nn.Module):
    def __init__(self, h, w, outputs):
        super(RLnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class CModel():
    def __init__(self, keypoint_detection = "sift", learning_rate = 0.001, num_episodes = 100):
        self.__strDevice = "cuda" if torch.cuda.is_available() else "cpu"
        self.__fLearningRate = learning_rate
        self.__uBatchSize = 128
        self.__fGamma = 0.999
        self.__fLambda = 0.1
        self.__sRewardFail = -20
        self.__sRewardSuccess = 20
        self.__uNumEpisodes = num_episodes
        self.__uActionSpace = 6
        self.__strKptDetection = keypoint_detection
        if(self.__strKptDetection == "sift"):
            DebugPrint().info("SIFT Create!")
            self.__oKptModel = cv2.SIFT_create()
            self.__oBfMatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif(self.__strKptDetection == "orb"):
            DebugPrint().info("ORB Create!")
            self.__oKptModel = cv2.ORB_create()
            self.__oBfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif(self.__strKptDetection == "surf"):
            DebugPrint().info("SURF Create!")
            self.__oKptModel = cv2.SURF_create()
            self.__oBfMatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def Setting(self, image, img_size = (1,480,640), video=False, checkpoint_path='./checkpoints/'):
        oImg = resize(image, img_size)
        self.__oState = torch.from_numpy(np.uint8(oImg)).to(self.__strDevice)
        _, self.__uHeight, self.__uWidth = oImg.shape
        self.__fCenter = (self.__uWidth / 2, self.__uHeight / 2)
        self.__oPolicyNet = RLnet(self.__uHeight, self.__uWidth, self.__uActionSpace).to(self.__strDevice)
        self.__oTargetNet = RLnet(self.__uHeight, self.__uWidth, self.__uActionSpace).to(self.__strDevice)
        self.__oTargetNet.load_state_dict(self.__oPolicyNet.state_dict())
        self.__oTargetNet.eval()

        self.__oOptimizer = torch.optim.RMSprop(self.__oPolicyNet.parameters())
       
        self.__strCkptPath = checkpoint_path + "dqn_" + self.__strKptDetection + "_checkpoint_B.pth"
        self.__bVideo = video

        if(video):
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            self.__oVideo = cv2.VideoWriter("./Video_B_" + str(self.__strKptDetection) + ".mp4", fourcc, 25.0, (self.__uWidth, self.__uHeight))

    def Test(self):
        bDone = False
        self.__uSteps=0
        self.__LoadCheckpoint()
        vKptOriginal, vDescOriginal = self.__oKptModel.detectAndCompute(np.squeeze(np.asarray(self.__oState.cpu()), axis=0), None)
        self.__vKptThreshold = [0, len(vKptOriginal) * 1.2]
        self.__uRotAngle = random.randint(0, 360)
        self.__fRotScale = random.uniform(0.5, 1.5)
        oImgWarp = self.__RotateImage(np.asarray(self.__oState.cpu()), self.__uRotAngle, self.__fRotScale)
        vKpWarpInit, vDescWarpInit = self.__oKptModel.detectAndCompute(oImgWarp, None)
        vMatchesInit = self.__oBfMatcher.match(vDescOriginal, vDescWarpInit)
        self.__uKptMatch = len(vMatchesInit)
        fMaxReward = -10
        sMaxKptDist = -100
        while not bDone:
            vAction = self.__SelectAction(self.__oState)
            oNextState, fReward, bDone, sKptDst = self.__TakeAction(vAction, np.asarray(self.__oState.cpu()), "Test")
            self.__oState = oNextState
            if(fReward > fMaxReward):
                fMaxReward = fReward
                sMaxKptDist = sKptDst
                oImg = np.squeeze(self.__oState, axis=0)
                cv2.imwrite("./result.png", oImg.numpy())
            if self.__uSteps % 100 == 0:
                DebugPrint().info("Reward: " + str(fReward))
                cudaStatus = CudaStatus()
                DebugPrint().info("CUDA Memory: " + str(cudaStatus["allocated"]) + "/" + str(cudaStatus["total"]))
            if(self.__uSteps > 100):
                break
        DebugPrint().info("Reward: " + str(fReward) + ", Init -> Dst: " + str(self.__uKptMatch) + " -> " + str(sMaxKptDist))
            

    def Train(self):
        TARGET_UPDATE = 1
        oSrcState = self.__oState
        self.__vKptOriginal, self.__vDescOriginal = self.__oKptModel.detectAndCompute(np.squeeze(np.asarray(oSrcState.cpu()), axis=0), None)
        
        # Convert original image to dark image
        oDarkImg = np.asarray((oSrcState.cpu() / 255.0) ** (1.0 / 0.4) * 255).astype(np.uint8)
        self.__vKptDark, _ = self.__oKptModel.detectAndCompute(np.squeeze(oDarkImg, axis=0), None)
        if(os.path.isfile(self.__strCkptPath)):
            self.__LoadCheckpoint()

        for iEpisode in range(self.__uNumEpisodes):
            bDone = False
            self.__uSteps = 0
            self.__oState = torch.from_numpy(oDarkImg)
            self.__oMemory = ReplayMemory(10000)
            fSumReward = 0
            while not bDone:
                vAction = self.__SelectAction(self.__oState)
                oNextState, fReward, bDone, sKptDst = self.__TakeAction(vAction, np.asarray(self.__oState.cpu()), iEpisode)
                self.__oState = torch.unsqueeze(self.__oState, 0).to(self.__strDevice, dtype=torch.float)
                oNextStateMem = torch.unsqueeze(oNextState, 0).to(self.__strDevice, dtype=torch.float)
                if(self.__uSteps > 100):
                    bDone = True
                    fReward = sKptDst - len(self.__vKptDark)
                oReward = torch.tensor([fReward], device=self.__strDevice)
                self.__oMemory.push(self.__oState, vAction, oNextStateMem, oReward)
                self.__oState = oNextState
                self.__OptimizeModel()
                fSumReward += fReward
                
            if iEpisode % TARGET_UPDATE == 0:
                DebugPrint().info("Episode: " + str(iEpisode) + ", Step: " + str(self.__uSteps) + ", Reward: " + str(fSumReward) + ", Init -> Kpt Dst: " + str(len(self.__vKptDark)) + " -> " + str(sKptDst))
                self.__oTargetNet.load_state_dict(self.__oPolicyNet.state_dict())
                cudaStatus = CudaStatus()
                DebugPrint().info("CUDA Memory:" + str(cudaStatus["allocated"]) + "/" + str(cudaStatus["total"]))
                self.__SaveCheckpoint()

    def __RotateImage(self, image, uAngle, fScale):
        mRot = cv2.getRotationMatrix2D(self.__fCenter, uAngle, fScale)
        image = np.squeeze(image, axis=0)
        oImageWarp = cv2.warpAffine(image, mRot, (self.__uWidth, self.__uHeight))
        return oImageWarp

    def __LoadCheckpoint(self):
        checkpoint = torch.load(self.__strCkptPath)
        self.__oPolicyNet.load_state_dict(checkpoint['Policy'])
        self.__oTargetNet.load_state_dict(checkpoint['Target'])
        self.__oOptimizer.load_state_dict(checkpoint['Optimizer'])

    def __SaveCheckpoint(self):
        checkpoint = {
            'Policy': self.__oPolicyNet.state_dict(),
            'Target': self.__oTargetNet.state_dict(),
            'Optimizer': self.__oOptimizer.state_dict()
        }
        torch.save(checkpoint, self.__strCkptPath)

    def Reset(self):
        if(self.__bVideo):
            self.__oVideo.release()

    def __SelectAction(self, state):
        EPS_END = 0.05
        EPS_START = 0.9
        EPS_DECAY = 200
        fSample = random.random()
        fEpsThreshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.__uSteps / EPS_DECAY)
        self.__uSteps += 1

        state = torch.unsqueeze(state, 0).to(self.__strDevice, dtype=torch.float)
        if(fSample > fEpsThreshold):
            with torch.no_grad():
                return self.__oPolicyNet(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.__uActionSpace)]], device=self.__strDevice, dtype=torch.long)

    def __TakeAction(self, action, state, episode):
        if action.item() == 0:
            fImgGamma = 1
            fAlpha = 0
        elif action.item() == 1:
            fImgGamma = 1.03
            fAlpha = 0
        elif action.item() == 2:
            fImgGamma = 1 / 1.03
            fAlpha = 0
        elif action.item() == 3:
            fAlpha = 0.01
            fImgGamma = 1
        elif action.item() == 4:
            fAlpha = -0.01
            fImgGamma = 1
        elif action.item() == 5:
            fAlpha = 0
            fImgGamma = 1
        
        oImage = np.clip(((1 + fAlpha) * state - 128 * fAlpha), 0, 255).astype(np.uint8)
        oImage = (((state / 255.0) ** (1.0 / (fImgGamma))) * 255).astype(np.uint8)
        
        vKpDst, vDesDst = self.__oKptModel.detectAndCompute(np.squeeze(oImage, axis=0), None)
        # vMatches = self.__oBfMatcher.match(vDesDst, self.__vDescOriginal)

        # uMatchNumber = len(vMatches)
        bFail = False
        bSuccess = False
        sDeltaN=0
        if(len(vKpDst) > len(self.__vKptOriginal)):
            bSuccess = True
        elif(len(vKpDst) < len(self.__vKptDark) * 0.9):
            bFail = True
        else:
            sDeltaN = -0.1
        # sDeltaN = uMatchNumber - self.__uKptMatch

        sReward = self.__fLambda * sDeltaN + bSuccess * self.__sRewardSuccess + bFail * self.__sRewardFail
        if(self.__bVideo):
            oImgVideo = np.squeeze(oImage, axis=0)
            oImgKeypt = cv2.drawKeypoints(oImgVideo, vKpDst, None)
            cv2.putText(oImgKeypt, "Episode " + str(episode), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(oImgKeypt, "(Action: " + str(action.item()) + ")", (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.__oVideo.write(oImgKeypt)

        return torch.from_numpy(oImage), sReward, bSuccess + bFail, len(vKpDst)

    def __OptimizeModel(self):
        if len(self.__oMemory) < self.__uBatchSize:
            return
        transitions = self.__oMemory.sample(self.__uBatchSize)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.__strDevice, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.__oPolicyNet(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.__uBatchSize, device=self.__strDevice)
        next_state_values[non_final_mask] = self.__oTargetNet(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.__fGamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.__oOptimizer.zero_grad()
        loss.backward()
        for param in self.__oPolicyNet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__oOptimizer.step()
