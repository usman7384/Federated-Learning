# Federated Learning & Data Privacy, 2025-2026

## Third Lab - 05 November 2025

Welcome to the third lab session of the Federated Learning & Data Privacy course! In our first two  labs, we ran experiments using our framework. Today, we will test a real production-based framework.


### EXERCISE 7: Get Started with Flower Framework
**Objective** Familiarize with [Flower](https://flower.ai/) and run your first Federated Simulation.

**Setup**:
Follow the [Get sterted with Flower](https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html) tutorial and launch your first Frederated Learning simulation. 

**Tip**: to create a new Python environment as indicated in the tutorial, you can first create and activate a new `conda` environment. To create a new environment use: 
```
conda create  -n <env_name> 
```
Then, to activate, use:
```
conda activate <env_name>
```
Now you are ready to proceed with Flower installation.

**Questions**:  
1. In the basic Flower setup described in the tutorial, what are the two main applications that the user must define? What are their respective roles?
2. How do clients and server communicate and share parameters in Flower?Describe the object they use and the purpose of each specific field.

3. How can a user define how a client should perform training? Is there any constraint on the name of the training function?

4. What is the difference between implementing an  `@app.evaluate` function on a server and on a client?

5. What is the purpose of the `Context` object? How can we modify the simulation parameters? 

### EXERCISE 8: Tackle device Heterogeneity

**Objective** understand how to handle system heterogeneity in federated learning.

### Exercise 8.1: Handle device heterogeneity with FjORD

Review the paper [FjORD: Fair and Accurate Federated Learning under heterogeneous targets with Ordered Dropout](https://openreview.net/forum?id=4fLr7H5D_eT). The paper presents Ordered Dropout as a method to adapt federated learning to heterogenous setting.

**Preliminary Questions**: 
1. What is system heterogeneity and why it is a problem in Federated Learning?
2. What is Ordered Dropout? 
3. How does the aggregation rule account for device heterogeneity?

**Implementation**: Follow the tutorial available at [Flower documentation](https://flower.ai/docs/baselines/fjord.html) and reproduce the results. Note that the tutorial reproduces the result for three different seeds. If you want to reduce the computational time, modify the `run.sh` script to use only one seed.

**IMPORTANT**: The tutorial uses an older version of Flower, different from the one used in the previous exercise. To reproduce the experiments, first create a new environment following the tutorial [Use Baselines](https://flower.ai/docs/baselines/how-to-use-baselines.html), and then, try to reproduce the experiment.


**Analysis**: Plot the results.   
1. Which one of the two implementation (with and without knowledge distillation) works better? Why?
2. How do different values of p impact the model’s accuracy? Motivate your answer.

### BONUS EXERCISE Federated  Distillation

**Objective** implement Federated Distillation strategy described in [Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data](https://arxiv.org/pdf/1811.11479). Reproduce and test only the Federated Distillation strategy, without Federated Augmentation.

**Tips**: 
- Flower automatically handles data communication between `ClientApp` and `ServerApp`. Focus on designing a custom `Message` and add a field in the `RecordDict` objects containing the logits of the client. Check [Communicate custom Messages](https://flower.ai/docs/framework/tutorial-series-customize-the-client-pytorch.html) to see how to design custmoized messages.
- Define a personalized strategy to implement the ensembling procedure in the `ServerApp`.  You can find some references here: [Flower Strategy Abstraction](https://flower.ai/docs/framework/explanation-flower-strategy-abstraction.html), [Customize a Flower Strategy](https://flower.ai/docs/framework/tutorial-series-build-a-strategy-from-scratch-pytorch.html).

---
At the end of the lesson, you can send your answers and code to: [francesco.diana@inria.fr](mailto:francesco.diana@inria.fr)

**IMPORTANT: you have time until 24/11/2025 to send me your solution. Late answers will be penalized.**
