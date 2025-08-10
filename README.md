
# Image-Processing-RL-Agent
A RL agent that Learn appropriate Preprocessing factors to be applied to images such that it can improve the performance of Yolov5 predictions of Various objects 

---

## Continuous Action Space Design
In this project, the action space is a continuous vector representing image transformation parameters.  
Each dimension of the action vector corresponds to a preprocessing parameter:
  - action[0] - Brightness | `[-1, 1]` | Adjusts pixel intensity
  - action[1] - Contrast | `[-1, 1]` | Modifies image contrast
  - action[2] - Sharpness | `[-1, 1]` | Sharpens or blurs the image
    
---
## Actor-Critic Architecture (From Scratch)
I've implemented an Actor-Critic network in PyTorch without using pre-built RL frameworks like Stable Baselines3.  

### Actor Network  [no_states - 256 - 256 - no_actions]
- **Input:** Image state representation  
- **Hidden Layers:** 2 fully connected layers with ReLU activation  
- **Output:** Continuous action vector (via `tanh` or scaled activation)  

### Critic Network  [no_states - (256 + no_action) - 256 - 1]
- **Input:** Concatenation of state and action  
- **Hidden Layers:** 2 fully connected layers with ReLU activation  
- **Output:** Q-value estimate for the given (state, action) pair  

### Learning Algorithm
-  DDPG (Deep Deterministic Policy Gradient)
  
## Training Loop
For each episode:
1. **Environment Reset** : Receive initial state.  
   
2. **Action Selection** : Actor predicts an action from the current state. After adding Noise Exploration is encouraged earlier on and more exploitation later in training.  

3. **Environment Step** : Apply action → receive `(next_state, reward, done, info)` from environment.  

4. **Replay Buffer Storage** : Store `(state, action, reward, next_state, done)` in memory.  

5. **Network Updates** *(if buffer > batch_size)*  
   - **Critic Update**  
     - **Target Q-value**:  
      **target_Q = rewards + GAMMA * (1 - dones) * target_Q**
     - Minimize MSE between `current_Q` and `y`.  
     - Apply gradient clipping (`max_grad_norm`).
       
   - **Actor Update** *(delayed by `policy_delay` steps)*  
     - Maximize \( Q(s, \pi(s)) \) via gradient ascent (implemented as **negative mean** for loss).
       
   - **Soft Update** of target networks after Actor update.  
---

