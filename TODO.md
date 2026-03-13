# TODO List

## Prompts

I want to build an agent orchestrator that spans multiple agents to optimize a machine learning model. The orchestrator will be responsible for understanding the problems with the current model and the optimization requirements, and then dispatching tasks to different agents (which can be a general-purpose agent, a research agent, or a hyperparameter tuning agent) to optimize the model and propose the skills needed for the optimization process.

There are a few ways to improve the model:
- Propose changes to the model architecture, more advanced learning algorithms, data augmentation, and preprocessing, but this requires a research agent that can read research papers and propose changes.
- Hyperparameter tuning, which can be done by a hyperparameter tuning module or skill that can run different hyperparameter configurations and record the results. It should use a hyperparameter tuning module that can run different hyperparameter configurations and record the results.

For every dispatched task to the agents, the orchestrator should also define the log format, report format, and development logs to keep track of the optimization process. The development logs are notes used to record development progress over time. Bash scripts used for running the experiments should also be recorded and organized. Any outputs from the experiments should be recorded and analyzed. The orchestrator will also be responsible for recording all findings and reporting the results.

The agent orchestrator should be able to work autonomously, making its own decisions and justifying them in the write-up. It should also be able to monitor the optimization process of every experiment it dispatches and catch divergence early.

The optimization process and the decisions on which methods or hyperparameters to use should be reasonable and based on the findings of previous experiments. For experiments that require model training, the orchestrator should check how many GPUs are available and send tasks to different agents to run experiments in parallel if multiple GPUs are available.

The agent orchestrator should propose an initial optimization plan and goal, and confirm them with the user before executing the plan.

---

Help me to create a test case to evaluate the plugin's ability to optimize a machine learning model using the agent orchestrator. Use conda 'base' environment for the test case.

---

Understand the requirements for the ml-optimizer plugin and identify bugs and areas for improvement. Also, make sure it is a universal plugin that can be used to optimize any machine learning model, not just a specific one, especially ensuring that all the skills and agents are designed to be flexible and adaptable to different optimization tasks. You may use /skill-creator to check for skill implementation and /simplify skill to aid for code simplification and readability.

Only use the conda "base" environment for the test cases.

I am confused about the project directory structure. What is the difference between the reports folder and the results folder? Should we merge them into one folder and save all the reports and results in Markdown format?

---

Understand the requirements for the ml-optimizer plugin and identify bugs and areas for improvement. You may use /skill-creator to check for skill implementation and /simplify skill to aid for code simplification and readability.

---

Review the test case for the ml-optimizer plugin and provide feedback on its effectiveness in evaluating the plugin's ability to optimize a machine learning model using the agent orchestrator. Keep the end-to-end test case, and try to simplify the other unit test cases but must also have a comprehensive coverage of the plugin's functionality.
Run all test cases in the conda "base" environment to ensure that they are working correctly and providing desired outputs.

---

Review the template for the reports. It should be comprehensive and cover all the necessary information about the optimization process, and it needs to be generic too.

Skip it if you think the template is already good enough and does not require any changes.

---

Double-check that when a user initializes this plugin by calling the agent orchestrator, it must enter plan mode to ask the user questions about the optimization process. This is important to ensure that the user has a clear understanding of the optimization process and can provide the necessary information for the agent orchestrator to make informed decisions.

---

Ultrathink and Suggest areas for improvement in the optimization process of the ml-optimizer plugin.

---

Suggest which skills or agents can utilize "ultrathink" in the prompt. You know the "ultrathink" feature, right? It sets the thinking mode to the maximum.

---

Ultrathink and suggest areas for improvement in the optimization process of the ml-optimizer plugin. Make sure that the plugin is general and can be used to optimize any machine learning model, not just a specific one. Ensure that all the skills and agents are designed to be flexible and adaptable to different optimization tasks.
Make sure that the unit tests are comprehensive and cover all the functionalities of the plugin, while also being simplified for readability. Run all test cases in the conda "base" environment to ensure that they are working correctly and providing the desired outputs.
You may use the /skill-creator skill to check the skill implementation and the /simplify skill to aid code simplification and readability.

Do you think the plugin is already good enough and does not require any changes? If not, please suggest improvements.

---

Only accept python3.10 or above for the ml-optimizer plugin. 

---

For the implement agent, the implementation could be done in two ways:
- From scratch based on the description of the methods in the paper.
- Refer to scripts from a reference repository that implements the method in the paper. You should be able to find the reference repository from the paper and clone it from GitHub.
Ultrathink, and make sure the current implementation of the agent can handle both cases, and it should be flexible enough to adapt to different methods and papers. The agent should also be able to analyze the paper and identify the key components and steps needed for the implementation process.

---

Do you think the optimization workflow is already good enough and does not require any changes? If not, please suggest improvements.

---

Add an agent to check the prerequisites for running the experiments, such as preparing the dataset and setting up the environment.

The user may only provide the paths to the training and validation datasets, but they might not meet the requirements for the training process, such as the dataset format or the necessary preprocessing steps.
Therefore, the agent needs to understand the codebase so it can prepare the dataset accordingly in the required format in a **new folder**.

For the environment, the agent needs to ask about the environment (e.g., which conda environment, or whether it uses a uv package). It also needs to check the modules used in the codebase, install any missing modules, and set up the environment for the training process.

---
Go to worktrees 'prerequisites'.

ULTRATHINK, do you think the prerequisites agent is already good enough to handle dataset preparation according to the requirements in the codebase, and can fit well into the full workflow without requiring any changes? If not, please suggest improvements.
Use conda "base" environment for testing the prerequisites agent.

---

This plugin should have self-improvement capabilities and should save any errors that occur when executing skills or agents. It should also suggest improvements to the user for the skills or agents after the session.

---

Go to worktrees 'self-improvement'.

This plugin should have self-improvement capabilities and should save any errors that occur when executing skills or agents. It should also suggest improvements to the user for the skills or agents after the session.

ULTRATHINK, do you think the review agent is already good enough to effectively analyze errors and suggest improvements for the skills or agents, fit well into the full workflow, and not require any changes? If not, please suggest improvements.
Use conda "base" environment for testing the review agent.

---

ULTRATHINK, and understand the ml-optimizer plugin, and do you think the optimization workflow is already good enough and does not require any changes? If not, please suggest improvements, and use git worktrees for the implementation.

- Make sure that the plugin is general and can be used to optimize any machine learning model, not just a specific one. Ensure that all the skills and agents are designed to be flexible and adaptable to different optimization tasks.
- Make sure that the unit tests are comprehensive and cover all the functionalities of the plugin, while also being simplified for readability. Run all test cases in the conda "base" environment to ensure that they are working correctly and providing the desired outputs. 

You may use /skill-creator to check for skill implementation and /simplify skill to aid for code simplification and readability.

---

Checkout this repo: https://github.com/karpathy/autoresearch

ULTRATHINK: Understand the ml-optimizer plugin and consider whether the AutoResearch module used in the repository can be integrated into the ml-optimizer plugin.

For example, the research and implementation agents could utilize AutoResearch to perform their tasks. Sometimes it may be necessary to integrate proposals from different papers. In other cases, the agents may simply propose good practices based on the LLM’s knowledge and web search, because the hp-tune agents currently only propose hyperparameter configurations based on the results of previous experiments and do not propose optimization methods or techniques.

The expected result outcomes are:
- Baseline; baseline + default proposed method with default hyperparameters (if there is a new implementation from a paper); and baseline + proposed method with tuned hyperparameters.
- Baseline; baseline + suggested proposed methods from the LLM’s knowledge and web search; and baseline + suggested proposed methods with tuned hyperparameters.

---

ULTRATHINK

I still think the hp-tune agent should only be responsible for hyperparameter tuning, and the research and implementation agents should be responsible for proposing new optimization methods and techniques and implementing them. The research agent can utilize web search and deep thinking to automatically propose optimization methods and techniques, and new changes should be created in Git worktrees. Anyway, all agents should be able to do web search.

If there are no research papers or newly proposed or advanced methods (e.g., changing the model architecture or learning method), which are outside the scope of the current repository, then only hyperparameter tuning is needed. This means the hp-tune agent performs hyperparameter tuning by exploring the search space available in the repository without any code changes, and the research and implementation agents can be skipped.

---

Checkout this repo: https://github.com/karpathy/autoresearch

ULTRATHINK: Understand the ml-optimizer plugin and consider whether the method used in the repository can be integrated into the ml-optimizer plugin.

Do you think the new enhancements for the optimization workflow are already good enough and do not require any changes? If not, please suggest improvements.

- Make sure that all the components of the optimization workflow are flexible and adaptable to different optimization tasks, and that the unit tests are comprehensive and cover all functionalities of the plugin.
- Make sure that all the components of the optimization workflow are fully automated, meaning the tasks can be done without human intervention and with high utilization of resources, with no breaks or idle time for the CPU and GPU. Tasks should be executed sequentially in an automated way.

---

ULTRATHINK

Option for an infinite-loop experiment (for the full flow, including the research and implementation agents) until the user asks to stop, after which the experiment is concluded and the results are reported.

The optimization flow should loop for an infinite number of rounds of research, implementation, and hyperparameter tuning until the user asks to stop. Then the agent orchestrator will conclude the experiment and report the results. This allows for continuous optimization and improvement of the machine learning model based on the findings from each round of research, implementation, and hyperparameter tuning.

The AutoResearch repository also has very nice plotting to show the results of a mini full run with improvements. Please check it and include it.

---

ULTRATHINK
Can you check which installed official Claude skills can be used for the planning, research, and implementation agents? Also, are there any Superpowers skills that can be used?

---

ULTRATHINK

Create an ascii diagram to show the workflow of the ml-optimizer plugin, including the interactions between the agent orchestrator, all the different agents (with available skills), and the user. The diagram should also illustrate the flow of information and tasks between the agents and how they contribute to the optimization process. The diagram should be detailed enough to provide the overall picture of the optimization workflow.

Then, review the end-to-end unit tests and make sure they cover all aspects of the optimization workflow. Run the tests in the conda "base" environment to ensure they are working correctly and providing the desired outputs.

---

What should the output directory format be for many sequential experiments? How should bash scripts be stored, since every experiment agent needs to save the bash scripts? Research and implementation may involve many changes (e.g., changing the model architecture, using different learning methods, different loss functions, etc.), which could require fully reramping the entire original machine learning method.

Why do I feel like the agents are not following the output format I want?

---

Mention the research agent also can consider training-free methods, test-time adaptation, and Monte Carlo Tree Search.

---

Checkout the worktree 'autoresearch-integration'.

ULTRATHINK, Double-check the previous runs in order to properly close the improvement loop.
Within the current scope, do you think the new enhancements for the optimization workflow are already good enough and do not require any changes? If not, please suggest improvements. Make sure all the components are fully autonomous and automated, and the workflow is flexible and adaptable to different optimization tasks.

---

Show me the output directory structure for the experiments. What do you think about the current output directory structure? Do you think it is good enough to capture multiple sequential experiments with different methods and hyperparameters and to keep track of everything in an organized way? If not, please suggest improvements.


---

Checkout this repo: https://github.com/karpathy/autoresearch

ULTRATHINK

It has very nice plotting to show improving results from the experiments.
Can you help me understand why? Are the different dots single experiments, or are they stacks of experiments?
For example, if I find that method 1 is useful, can I stack method 1 with method 2 and method 3 to see if it can further improve the results? Or do I need to run method 1, then method 2, and then method 3 separately to see the improvement from each method?

---

First, the agent should ask the user whether they want to stack methods or not.

Let’s say I have a baseline. After the agent implements a new method from a paper (merging the baseline with the new method), it should run the experiment and then perform hyperparameter tuning (if needed, based on the LLM’s judgment).

Continue until there are five new methods with improvements (meaning implementing five separate new methods on top of the baseline), and only then perform stacking.

For stacking, stack the method with the highest improvement first, followed by the rest accordingly.
Make sure it stack the different implementation methods (can be from different branches or repositories or papers), not just only stack the hyperparameters.

---

Checkout this repo: https://github.com/karpathy/autoresearch

ULTRATHINK

Double-check that the optimization workflow can plot graphs similar to those in the repository and is capable of stacking different methods and techniques, not just hyperparameters. 
At the same time, identify any bugs in the optimization workflow and suggest improvements if needed.

---

ULTRATHINK

- Help me check why the orchestrate skill did not call the necessary agents to perform the optimization process, and why the sub-agent was not able to call the correct skills to perform the tasks.
- Also, I still did not get the desired output format I want. Maybe this is because of the issue above, or perhaps I can use hooks to enforce the output format.


- context, eg. Set to fork to run in a forked subagent context. agen, eg. Which subagent type to use when  is set.
- Preload skills into subagents

---

ULTRATHINK

- I want to create one more output folder for artifacts and checkpoints that can be used to store any intermediate files, model checkpoints, or other artifacts or images generated during the optimization process. Please ensure the hooks are properly set up to ensure the correct output format for all requirements.

- Are there any hooks you can suggest to improve the security and reliability of the plugin if the agent needs to perform tasks in a fully automated manner without human intervention?

Hooks documentation:

---

- Double-check the new changes and make sure there are no bugs.
- I want to run an end-to-end test case for the full optimization workflow (all agents and skills involved) to ensure everything is working correctly and producing the desired outputs. Use the conda "base" environment for testing.

---

I think we should only have the orchestrate skill be called, not the other skills. Then, the orchestrate skill will call the necessary agents, and the agents will call the necessary skills to perform the tasks.  

- Make sure all the necessary agents (baseline-agent, monitor-agent, analysis-agent, report-agent, and review-agent) are properly set up.  
- Make sure the agents can call the correct skills to perform the tasks; do not allow the user to call the skills directly.

---

Double-check the new changes and make sure there are no bugs, and I think we should only have the orchestrate skill be called, not the other skills. Then, the orchestrate skill will call the necessary agents, and the agents will call the necessary skills to perform the tasks.  