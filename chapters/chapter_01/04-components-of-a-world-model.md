# 1.4 Components of a World Model

A world model receives recent observations and, when applicable, actions. It
converts the observations into an internal state. It predicts how that state
changes. It then converts the predicted state into an output.

![](../../figures/chapter_01/fig_1_10_blueprint.png)

*Figure 1.5: Information moves from an observation to a representation, through
dynamics, and into a prediction. An action can affect the dynamics update.*

## Observations and Actions

An observation is information available from the environment at one time step.
It can be an image, a sensor vector, a text sequence, or a structured state.

A sequence of observations forms an observation history. One image can show an
object's position. A sequence can also show how its position changes.

An action is a command applied to the environment. Examples include a steering
command, a robot movement, or a mouse click. Some prediction tasks do not
include actions.

## Representation

An encoder converts the observation history into an internal state. This state
is also called a representation. It contains information used to predict what
happens next. For a moving object, that information may include its position
and motion.

## Dynamics

The dynamics function predicts how the internal state changes. It uses the
current state and, when applicable, an action. Its output can be one next state
or a distribution over possible next states.

## Prediction

The decoder converts the predicted internal state into an output. The output
may be the next image, the next sensor reading, or another observation.

The encoder, dynamics function, and decoder describe three roles in the
prediction process. An architecture may implement them as separate components
or combine them.

## One Transition Through the Pipeline

Consider the ball on the one-dimensional track from Section 1.1:

1. The observation history contains recent positions of the ball.
2. The encoder produces a representation of its position and motion.
3. The action specifies whether to push left, push right, or not push.
4. The dynamics function uses the representation and action to predict the next
   state.
5. The decoder converts that state into the predicted next position.

These steps produce one predicted transition. Passing the predicted state
through the dynamics function again produces another transition. Repeating the
update produces a rollout.

The pipeline walkthrough applies these steps to a ball moving along a line.
Select Step to move through the operations. Change the observation format,
the action, or whether the dynamics function produces one or several next
states.

[Open the world-model pipeline in a new tab](https://nahidalam.github.io/world_model_from_scratch/interactive/chapter_01/pipeline_walkthrough.html).

Chapter 2 applies this pipeline using a pretrained world model. It begins by
preparing an observation and generating a rollout.
