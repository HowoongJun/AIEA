import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
import cv2, random, torch, math
from skimage import io
from common.Log import DebugPrint

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
        self.__sRewardFail = -10
        self.__sRewardSuccess = 10
        self.__uNumEpisodes = num_episodes
        self.__uActionSpace = 5
        self.__strKptDetection = keypoint_detection
        if(self.__strKptDetection == "sift"):
            self.__oKptModel = cv2.SIFT_create()
        elif(self.__strKptDetection == "orb"):
            self.__oKptModel = cv2.ORB_create()

    def Setting(self, image, video=False, checkpoint_path='./checkpoints'):
        self.__oState = torch.from_numpy(np.uint8(image)).to(self.__strDevice)
        _, uHeight, uWidth = image.shape
        
        self.__oPolicyNet = RLnet(uHeight, uWidth, self.__uActionSpace).to(self.__strDevice)
        self.__oTargetNet = RLnet(uHeight, uWidth, self.__uActionSpace).to(self.__strDevice)
        self.__oTargetNet.load_state_dict(self.__oPolicyNet.state_dict())
        self.__oTargetNet.eval()

        self.__oOptimizer = torch.optim.RMSprop(self.__oPolicyNet.parameters())
        self.__oMemory = ReplayMemory(10000)

        self.__strCkptPath = checkpoint_path
        self.__bVideo = video

        if(video):
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            self.__oVideo = cv2.VideoWriter("./TrainVideo.mp4", fourcc, 25.0, (uWidth, uHeight))

    def Test(self):
        bDone = False
        self.__uSteps=0

        vKptOriginal, _ = self.__oKptModel.detectAndCompute(np.squeeze(np.asarray(self.__oState.cpu()), axis=0), None)
        self.__vKptThreshold = [len(vKptOriginal) * 0.8, len(vKptOriginal) * 1.1]

        while not bDone:
            vAction = self.__SelectAction(self.__oState)
            oNextState, fReward, bDone, _ = self.__TakeAction(vAction, np.asarray(self.__oState.cpu()), "Test")
            
            if self.__uSteps % 100 == 0:
                DebugPrint().info("Reward: " + str(fReward))
            if(self.__uSteps > 1000):
                break

    def Train(self):
        TARGET_UPDATE = 10
        oSrcState = self.__oState
        vKptOriginal, _ = self.__oKptModel.detectAndCompute(np.squeeze(np.asarray(oSrcState.cpu()), axis=0), None)
        self.__vKptThreshold = [len(vKptOriginal) * 0.8, len(vKptOriginal) * 1.2]
        DebugPrint().info("Original Kpt Number: " + str(len(vKptOriginal)) + ", " + str(self.__vKptThreshold))
        for iEpisode in range(self.__uNumEpisodes):
            bDone = False
            self.__uSteps = 0
            self.__oState = oSrcState
            if(iEpisode > 0):
                self.__oTargetNet.load_state_dict(torch.load(self.__strCkptPath + "/dqn_checkpoint.pth"))
                
            while not bDone:
                vAction = self.__SelectAction(self.__oState)
                oNextState, fReward, bDone, sKptDst = self.__TakeAction(vAction, np.asarray(self.__oState.cpu()), iEpisode)
                oReward = torch.tensor([fReward], device=self.__strDevice)
                self.__oState = torch.unsqueeze(self.__oState, 0).to(self.__strDevice, dtype=torch.float)
                oNextStateMem = torch.unsqueeze(oNextState, 0).to(self.__strDevice, dtype=torch.float)
                self.__oMemory.push(self.__oState, vAction, oNextStateMem, oReward)
                self.__oState = oNextState
                self.__OptimizeModel()

            if iEpisode % TARGET_UPDATE == 0:
                DebugPrint().info("Episode: " + str(iEpisode) + ", Reward: " + str(fReward) + ", Kpt Dst: " + str(sKptDst))
                self.__oTargetNet.load_state_dict(self.__oPolicyNet.state_dict())
                torch.save(self.__oTargetNet.state_dict(), self.__strCkptPath + "/dqn_checkpoint.pth")
            torch.cuda.empty_cache()

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
        if action == 0:
            fAlpha = 0
            fImgGamma = 1
        elif action == 1:
            fAlpha = 0.01
            fImgGamma = 1.02
        elif action == 2:
            fAlpha = -0.01
            fImgGamma = 1.02
        elif action == 3:
            fAlpha = 0.01
            fImgGamma = 0.98
        elif action == 4:
            fAlpha = -0.01
            fImgGamma = 0.98

        oImage = np.clip(((1 + fAlpha) * state - 128 * fAlpha), 0, 255).astype(np.uint8)
        # oImage = (((oImage / 255) ** (1 / (fImgGamma + 0.000000001))) * 255).astype(np.uint8)
        
        vKpSrc, _ = self.__oKptModel.detectAndCompute(np.squeeze(state, axis=0), None)
        vKpDst, _ = self.__oKptModel.detectAndCompute(np.squeeze(oImage, axis=0), None)

        bFail = False
        bSuccess = False

        if(len(vKpDst) > self.__vKptThreshold[1]):
            bSuccess = True
        elif(len(vKpDst) < self.__vKptThreshold[0]):
            bFail = True
        sDeltaN = len(vKpDst) - len(vKpSrc)
        sReward = self.__fLambda * sDeltaN + bSuccess * self.__sRewardSuccess + bFail * self.__sRewardFail
        if(self.__bVideo):
            oImgKeypt = cv2.drawKeypoints(np.squeeze(state, axis=0), vKpSrc, None)
            cv2.putText(oImgKeypt, "Episode " + str(episode), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(oImgKeypt, "Reward " + str(sReward), (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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
