import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
import cv2, random, torch, math
from skimage import io

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


class CTrain():
    def __init__(self, drl_module, learning_rate = 0.001, num_episodes = 50):
        self.__strDevice = "cuda" if torch.cuda.is_available() else "cpu"
        self.__fLearningRate = learning_rate
        self.__uBatchSize = 128
        self.__fGamma = 0.999
        self.__uNumEpisodes = num_episodes
        self.__oDrlModule = drl_module
        self.__uActionSpace = 3
        self.__vKptThreshold = [100, 1000] #Temp - Need to be edited heuristically
        self.__oSift = cv2.SIFT_create()

    def Setting(self, image):
        self.__oState = torch.from_numpy(image).to(self.__strDevice)
        _, uHeight, uWidth = image.shape
        self.__oPolicyNet = self.__oDrlModule.RLnet(uHeight, uWidth, self.__uActionSpace).to(self.__strDevice)
        self.__oTargetNet = self.__oDrlModule.RLnet(uHeight, uWidth, self.__uActionSpace).to(self.__strDevice)
        self.__oTargetNet.load_state_dict(self.__oPolicyNet.state_dict())
        self.__oTargetNet.eval()

        self.__oOptimizer = torch.optim.RMSprop(self.__oPolicyNet.parameters())
        self.__oMemory = ReplayMemory(10000)

    def Run(self):
        TARGET_UPDATE = 10
        for iEpisode in range(self.__uNumEpisodes):
            bDone = False
            self.__uSteps = 0
            while not bDone:
                vAction = self.__SelectAction(self.__oState)
                self.__oState = np.asarray(self.__oState.cpu())
                oNextState, fReward, bDone = self.__TakeAction(vAction)
                
                self.__oMemory.push(self.__oState, vAction, oNextState, fReward)
                self.__oState = oNextState
                self.__OptimizeModel()

            if iEpisode % TARGET_UPDATE == 0:
                self.__oTargetNet.load_state_dict(self.__oPolicyNet.state_dict())
        io.imsave("./result.png", np.squeeze(self.__oState.cpu(), axis=0))
        

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

    def __TakeAction(self, action):
        if action == 0:
            fAlpha = 0
        elif action == 1:
            fAlpha = 0.1
        elif action == 2:
            fAlpha = -0.1
        oImage = np.clip(((1 + fAlpha) * self.__oState - 128 * fAlpha), 0, 255).astype(np.uint8)
        
        vKpSrc, _ = self.__oSift.detectAndCompute(self.__oState, None)
        vKpDst, _ = self.__oSift.detectAndCompute(oImage, None)
        
        sReward = 0
        if(len(vKpDst) > len(vKpSrc)):
            sReward = 1
        elif(len(vKpDst) < len(vKpSrc)):
            sReward = -1
        oImage = torch.from_numpy(oImage)
        bDone = False
        if(len(vKpDst) > self.__vKptThreshold[1] or len(vKpDst) < self.__vKptThreshold[0]):
            bDone = True

        return oImage, sReward, bDone

    def __OptimizeModel(self):
        if len(self.__oMemory) < self.__uBatchSize:
            return
        transitions = self.__oMemory.sample(self.__uBatchSize)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.__oDrlNet(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.__oTargetNet(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.__fGamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        torch.optimizer.zero_grad()
        loss.backward()
        for param in self.__oDrlNet.parameters():
            param.grad.data.clamp_(-1, 1)
        torch.optimizer.step()