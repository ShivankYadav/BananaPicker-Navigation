# BananaPicker-Navigation
We train an agent to manuver in a 3-D environment avoiding black bananas and picking yellow ones as fast as possible. We do this by implementing Deep Q network based on this [research paper](https://www.nature.com/articles/nature14236). The agent works thorugh Agent class in [dqn_agent.py](https://github.com/ShivankYadav/BananaPicker-Navigation/blob/master/dqn_agent.py) and the model architecture is described in [model.py](https://github.com/ShivankYadav/BananaPicker-Navigation/blob/master/model.py).

## Environment Details
!["GIF"](https://github.com/ShivankYadav/BananaPicker-Navigation/blob/master/images/banana.gif)

A **reward** of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The **state space** has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four **discrete actions** are available, corresponding to:

  * 0 - move forward.
  * 1 - move backward.
  * 2 - turn left.
  * 3 - turn right.

The task is **episodic**, and in order to solve the environment, your agent must get an **average score of +13 over 100** consecutive episodes.

## Getting Started
  * ```pip install unityagents``` **Unity Machine Learning Agents (ML-Agents)** is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. For game developers, these trained agents can be used for multiple purposes, including controlling NPC behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release.You can read more about ML-Agents by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents).
  * The user must install **pytorch** according to the specifications on his/her workspace. I used 
      torch                     1.4.0 and 
      torchvision               0.4.2 corrosponding to CUDA 10.1
  * Other python modules like **numpy** and **matplotlib** should be installed.
 ## Downloading and setting up Environment
 For this project, you will not need to install Unity - this is because we are using pre-built environment, and you can download it from one of the links below. You need only select the environment that matches your operating system:

  * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
  * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
  
Unzip (or decompress) the file in this repository.

(For Windows users) Check out this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

## Instructions

