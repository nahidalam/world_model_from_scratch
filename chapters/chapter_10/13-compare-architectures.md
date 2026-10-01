# 10.13 Compare the Architectures

We have seen eleven ways to build a world model. To compare them, start with what we want the prediction to do. We might need a video we can inspect, a state that helps a policy learn, or a prediction that lets a planner compare actions. Each use places different demands on the model.

The models store the world in different tensors. That choice affects the training target, rollout cost, available planning method, and whether we can decode the prediction into a video.

Use the [architecture explorer](https://github.com/nahidalam/world_model_from_scratch_private/tree/main/interactive/chapter_10/architecture_explorer.html) to place any two models side by side. The tables compare their state, dynamics, training target, uncertainty, and use of predicted futures. Follow the same choices across both models as you read.

## Compare the Prediction Interfaces

First, trace what goes into a prediction and what comes out. This tells us which questions we can ask each model and how we can use its answers.

| Model             | Dynamics inputs                                                                                                                    | Training targets along rollout                                                               | Model uncertainty        | Action alternatives                                                                        | How predictions are used                                                | Pixel decoder                                                 |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------- |
| World Models      | VAE latent, previous action, LSTM memory                                                                                           | Per-coordinate next-latent likelihood                                                        | Gaussian-mixture sample  | Controller output; CMA-ES varies controller parameters                                     | Supply memory to a linear controller; support dream training in VizDoom | VAE decoder                                                   |
| MuZero            | Learned task state + candidate action                                                                                              | Reward, policy, and value                                                                    | Deterministic transition | Search-tree branches                                                                       | Back up rewards and values in MCTS                                      | None                                                          |
| IRIS              | Discrete image-token and action history                                                                                            | Next token, reward, termination                                                              | Categorical token sample | Actor output                                                                               | Train an actor-critic in imagination                                    | Discrete autoencoder                                          |
| DIAMOND           | Recent RGB frames, action, noisy next frame                                                                                        | Denoised next RGB frame; separate reward and termination targets                             | Gaussian diffusion noise | Actor output                                                                               | Train an actor-critic in imagined observations                          | Direct pixel generation                                       |
| Dreamer 4         | Continuous context, actions, noisy future representations                                                                          | Clean future representation                                                                  | Gaussian flow noise      | Policy output                                                                              | Train policy and value heads in imagination                             | Causal tokenizer decoder                                      |
| MIRA              | Tiled per-view codec latents, four action streams, noisy latent frames                                                             | Flow velocity for independently noised latent frames                                         | Gaussian flow noise      | Any four-player action combination                                                         | Advance a real-time multiplayer simulation                              | Causal video-codec decoder                                    |
| H3-World          | Static scene and first-image encoder tokens, first-frame VAE condition, per-latent action text, noisy full-horizon video and audio | MiniMax-H3 flow velocity for recorded gameplay video and encoded silence in the audio stream | Gaussian flow noise      | Complete scheduled keyboard sequence                                                       | Render a fixed video segment that follows the supplied schedule         | MiniMax-H3 visual VAE decoder                                 |
| DINO-WM           | Frozen DINOv2 patch features + candidate actions                                                                                   | Future DINOv2 patch features                                                                 | Deterministic prediction | CEM or optimized sequences                                                                 | Minimize terminal feature distance                                      | None for planning                                             |
| V-JEPA 2-AC       | Frozen V-JEPA 2 features, robot state, candidate actions                                                                           | Future V-JEPA 2 features                                                                     | Deterministic prediction | CEM sequences                                                                              | Minimize terminal feature energy in MPC                                 | None for planning                                             |
| LeWM              | Jointly learned projected CLS states + candidate actions                                                                           | Next-state MSE; SIGReg regularizes the state                                                 | Deterministic prediction | CEM sequences                                                                              | Minimize terminal latent distance in MPC                                | None for training or planning; separate visualization decoder |
| Cosmos-Predict2.5 | Observed video latents, text, condition mask, noisy full-clip latents                                                              | Flow velocity in the generated region                                                        | Initial Gaussian noise   | Prompt describes the requested event; no explicit per-step control in the studied baseline | Render a future video for an external consumer                          | WAN2.1 causal VAE decoder                                     |

When comparing rollouts, distinguish two sources of variation: sampling another prediction under the same conditions and changing the actions. World Models, IRIS, DIAMOND, Dreamer 4, MIRA, H3-World, and Cosmos-Predict2.5 can sample different predictions for the same condition. MuZero, DINO-WM, V-JEPA 2-AC, and LeWM compare futures produced by different actions. MIRA and H3-World expose both sources: Gaussian noise can change a rollout, and supplied controls can change the generated future.

## Choose What the Model Must Preserve

The training target tells the model which information it needs to keep. A reconstruction objective asks the representation to preserve enough detail to recreate the observation. World Models and IRIS make this requirement explicit through an autoencoder. Dreamer 4 trains a continuous tokenizer with pixel and perceptual losses. MIRA trains a DINOv3-based representation codec with pixel and perceptual losses. H3-World reuses MiniMax-H3's pretrained visual VAE and multimodal H3 encoder. Cosmos-Predict2.5 uses the WAN2.1 causal VAE to compress video before its Transformer predicts latent velocity. DIAMOND keeps pixels as its prediction space.

MuZero needs a state that supports reward, policy, and value prediction along search trajectories. Visual details can disappear if those targets do not need them. DINO-WM and V-JEPA 2-AC instead reuse a visual encoder, then learn how actions change its representations. LeWM learns the encoder and dynamics together. It uses SIGReg from LeJEPA to push the state distribution toward an isotropic Gaussian without a reconstruction target.

When choosing among these designs, consider what you need to do with the prediction:

* Use an observation-generating model when people must inspect predicted futures or another system consumes pixels.
* Use a task-targeted state when rewards, values, or search statistics define all required outputs.
* Use pretrained patch features when the task supplies goal observations and planning can compare predicted features directly with encoded goals.
* Train a compact state with the dynamics when aligned action data covers the task and fast goal-image planning matters.

These choices also shape how we evaluate the result. A sharp generated image can show the wrong consequence of an action. A latent prediction can guide the correct action even if we cannot turn it into a recognizable image. When applying what we learned in Chapter 3, choose measurements that fit what the architecture predicts and how we use that prediction.

## Choose the Temporal Mechanism

Next, consider how much history the model needs to make a useful prediction. The architectures keep that history in different ways:

```
World Models       recurrent LSTM memory
MuZero             recurrent state along a search branch
IRIS               causal Transformer token context
DIAMOND             four-frame transition window + recurrent reward and actor states
Dreamer 4          block-causal Transformer context and KV cache
MIRA               rolling joint-view window + factorized space-time attention
H3-World           bidirectional attention over one complete 37-interval video block
DINO-WM            several encoded history frames in a Transformer
V-JEPA 2-AC        current representation plus an action sequence
LeWM               short history of one projected CLS state per frame in a causal Transformer
Cosmos             a noisy video-latent block with observed positions fixed
```

With recurrence, we compress the past into a fixed-size state. With attention, we retain individual context tokens and let each query choose what to read. A fixed observation window gives the model a bounded history to work with, without a recurrent state. To choose a mechanism, consider how far the environment requires the model to look back and how much time we have for each new prediction.

## Connect a Prediction to an Action

A prediction becomes useful for control when it changes what the agent does. The architectures make that connection at different points: some use predictions to train a policy, some search at each decision, and others generate a world under controls supplied by an external system.

| Architecture      | How predictions are used                                          | When the model supplies predictions                                   | Action executed at deployment                                            |
| ----------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| World Models      | Evaluate controller parameters in an environment or learned dream | During CMA-ES evaluation; the learned environment is used for VizDoom | Controller output                                                        |
| IRIS              | Update actor and critic parameters                                | During actor-critic training in imagined token episodes               | Actor output                                                             |
| DIAMOND           | Update actor and critic parameters                                | During actor-critic training in imagined pixel episodes               | Actor output                                                             |
| Dreamer 4         | Update policy and value parameters                                | During reinforcement learning in imagined representations             | Policy output                                                            |
| MIRA              | Advance the generated joint-view state                            | At every 10 Hz latent step under four external action streams         | Supplied player controls; model-filled behavior for omitted streams      |
| H3-World          | Render the consequence of a complete action schedule              | During one full-horizon denoising pass                                | None inside H3-World; an external system supplies the complete schedule  |
| MuZero            | Update search-tree values and visit counts                        | During MCTS at every real decision                                    | Most-visited root action                                                 |
| DINO-WM           | Rank candidate action sequences                                   | During CEM or trajectory optimization at every real decision          | First action or short prefix                                             |
| V-JEPA 2-AC       | Rank candidate action sequences                                   | During CEM at every real decision                                     | First action                                                             |
| LeWM              | Rank candidate action sequences                                   | During CEM at every image-goal decision                               | Entire five-step sequence, equal to 25 environment actions               |
| Cosmos-Predict2.5 | Render a continuation of an observed scene                        | During repeated updates to a full latent video                        | None inside the studied generator; an external system consumes its video |

World Models, IRIS, DIAMOND, and Dreamer 4 move most of the optimization into controller or policy training. MuZero searches a branching action tree at deployment. DINO-WM, V-JEPA 2-AC, and LeWM optimize action sequences at deployment by comparing predicted terminal states with encoded goal states. MIRA advances its generated world under external actions. H3-World receives all scheduled actions before generation and renders their fixed-horizon consequences. Cosmos-Predict2.5 generates a complete clip conditioned on text and an observed prefix. These three generators supply predictions to an external consumer without selecting actions from a reward objective.

We can try the goal-feature planning pattern with a small calculation. The planner below predicts the result of each candidate action sequence and compares it with a goal:

```python
import torch
from world_models.architecture_examples import (
    mean_absolute_goal_cost,
    select_action_sequence,
)

def dynamics(state, action):
    return state + action

initial = torch.tensor([0.0, 0.1])
goal = torch.tensor([2.0, 0.0])
candidates = torch.tensor([
    [[1.0, 0.0], [1.0, 0.0]],
    [[0.0, 1.0], [0.0, 1.0]],
    [[1.0, 0.0], [0.0, 1.0]],
])

best, costs, trajectories = select_action_sequence(
    initial, candidates, goal, dynamics
)
vjepa_best, vjepa_costs, _ = select_action_sequence(
    initial, candidates, goal, dynamics, cost=mean_absolute_goal_cost
)
print("DINO-WM / LeWM squared-error family:", best, costs)
print("V-JEPA 2-AC-style L1:", vjepa_best, vjepa_costs)
```

The helper uses mean squared feature error, which matches DINO-WM. LeWM sums the same elementwise squared errors, so it ranks fixed-size candidates in the same order. Pass `mean_absolute_goal_cost` as the `cost` argument to match V-JEPA 2-AC.

## Place Related Models on the Map

Once we can trace these choices, it becomes easier to understand related models. The following systems combine or extend ideas we have already seen:

* PlaNet introduced the recurrent state-space model that combines a deterministic recurrent state with a stochastic latent state. The Dreamer family built policy learning in imagination on that foundation. Section 10.5 [traces the four Dreamer versions](05-dreamer-lineage.md). [PlaNet paper](https://arxiv.org/abs/1811.04551)
* TD-MPC2 learns latent dynamics, reward, value, and policy predictions, then plans with model-predictive control. It belongs beside MuZero on the task-directed latent branch. [TD-MPC2 paper](https://arxiv.org/abs/2310.16828)
* GAIA-1 represents video, text, and actions as tokens and predicts future video tokens autoregressively. It scales the token-modeling idea toward driving scenes. [GAIA-1 paper](https://arxiv.org/abs/2309.17080)
* GameNGen uses action-conditioned diffusion to simulate DOOM. It belongs beside DIAMOND on the observation-generating diffusion branch. [GameNGen paper](https://arxiv.org/abs/2408.14837)

Section 10.12 [traces Cosmos-Predict2.5](12-cosmos.md) as this chapter's full-clip video generator. [Chapter 4](../chapter_04/) implements its Transformer and sampler and checks them against the released weights.

The scale, data, and training details change, but we can keep using the same four questions: what represents the world, what changes that representation, what target trains the change, and how do the predictions affect actions?

> Try it yourself
>
> Pick one task you care about. Write down its observation, action, desired prediction, and decision deadline. Select two architectures from the table and state which tensor would carry each item. If a tensor has no clear place, the architecture needs another input or output before it fits the task.
