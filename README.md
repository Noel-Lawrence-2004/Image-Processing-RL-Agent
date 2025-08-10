
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
     - **Target Q-value**:  *target_Q = rewards + GAMMA * (1 - dones) * target_Q*
     - Minimize MSE between `current_Q` and `target_Q`.  
     - Apply gradient clipping (`max_grad_norm`).
       
   - **Actor Update** *(delayed by `policy_delay` steps)*  
     - Maximize \( Q(s, \pi(s)) \) via gradient ascent (implemented as **negative mean** for loss).
       
   - **Soft Update** of target networks after Actor update.
     
---
## Training Reward Plot

Below is the cumulative reward per episode recorded during training.  
![Cumulative Reward per Episode](results/cumulative_reward.png)

---
## Observations & Analysis

### Did the agent learn meaningful transformations?
In our runs, the agent **did** learn useful and consistent transformations for many images.  
It often increased **brightness**, **contrast**, or modestly **sharpened** images that originally had low YOLO detection confidence.  

This improvement is visible as:
- A **rising rolling mean** of episodic reward in the saved training plot.
- **Qualitative samples** showing processed images with more confident detections and, in some cases, more detected objects.

---

### Concrete indicators we tracked
- **Episode reward (training)** — shows the agent’s episodic objective (YOLO confidence changes) improving over time.
- **Evaluation (no-noise) episodes** — run periodically to get a stable performance curve measured without exploration noise.
- **Per-image confidence change histogram** — useful to verify whether most images improved or only a few outliers.
- **5×2 qualitative grid** — visually compares original vs processed detections.

---

### Primary challenges encountered with continuous control

#### 1. Reward noise & instability
- YOLO confidences can jump drastically for small image changes, producing noisy rewards.  
- Mitigation: **tanh** scaling + short-step reward smoothing (moving average over the last 3–5 steps).

#### 2. Exploration vs exploitation trade-off
- Continuous action spaces need careful noise scheduling.  
- Used **exponentially decaying Gaussian noise** (start → end) so the agent explores widely early, then fine-tunes later.

#### 3. Action scaling / ranges
- Sensible ranges for brightness/contrast/sharpness are crucial:  
  - Too large → destructive transformations, unstable training.  
  - Too small → flat reward landscape.  
- We started with **moderate ranges** and **annealed scale** during training.

#### 4. Replay buffer / update cadence
- Too many gradient updates per environment step can overfit to stale samples and destabilize the critic.  
- Solution: Reduce `updates_per_step` and use `policy_delay` (update actor less often than critic) for stability.

#### 5. Sample efficiency
- With a limited dataset, balancing replay usage and fresh data is important.  
- Increasing `MAX_STEPS` per episode or using a **moderate replay buffer size** helped maintain batch diversity.

