# BREATHE - Batch-constrained Reinforcement Environment for Airway Therapy Health Enhancement

## Introduction
Critically ill patients in the Intensive Care Unit (ICU) often require mechanical ventilation to support respiratory 
function. Setting the appropriate ventilator parameters is a complex task, as these must be tailored to individual 
patient factors and adjusted dynamically in response to changes in the patient's condition.

The goal of this project was to train a reinforcement learning (RL) agent to recommend optimal ventilator settings that
improve patient health outcomes. Unlike clinicians, an RL agent has the potential to continuously monitor the patient’s
condition and propose real-time adjustments, offering a personalized and dynamic approach to ventilator management.

## Methods
We utilized the “A Temporal Dataset for Respiratory Support in Critically Ill Patients”, a component of the MIMIC v2.2 
dataset. This provided 90-day hourly ventilation data for 50,920 adult ICU patients.

We processed this data to generate states consisting of static state (age, gender, height, weight) plus a 4-hour sliding
window of dynamic state, a discretized action space, and custom reward function, to train our neural networks.

## Deep Learning Networks
Training RL models on medical data poses unique challenges due to the strictly offline nature of training. Since the 
data is entirely historical, there is no opportunity for unconstrained exploration by the RL agent, increasing the
risk of spurious or unsafe recommendations in poorly represented regions of the observation space.

To mitigate this risk, we employed batch-constrained reinforcement learning, which restricts the agent’s action space
for any given observation state to actions deemed feasible by actual clinical practice. This ensures the agent is 
trained and operates within a realistic and clinically relevant domain.

Our model architecture involves three components:
1. Generator Network:
• Trained to predict actions, based on state
• The generator constrains the RL agent’s choices to the top k most probable actions, ensuring feasible settings during 
training and operation.
2. Batch-Constrained DDQN RL Agent:
• DDQN was used to train the RL agent on historical ICU data, with the action space constrained by the Generator network
3. Predictor Network:
• Trained on historical state-action pairs to predict subsequent states.
• Used during validation to compare predicted health outcomes against observed outcomes in the historical dataset

## Results
After training, the generator network accurately identified the exact historical action 39.4% of the time. Importantly, 
the correct action was present among the top k (with k=4) actions predicted by the generator in 73.3% of cases on the 
validation data.

To validate the predictor network, we discretized the continuous observation state outputs for evaluation. The trained
predictor network successfully predicted the subsequent state, given a (state, action) pair, with 83.1% accuracy.

The trained batch-constrained DDQN RL agent, was deployed to recommend actions for every state in the validation 
dataset. The primary health outcomes evaluated were the maintenance of optimal mean blood pressure ([70,80] mmHg), and 
optimal oxygen saturation levels ([94%,98%]).

The RL agent demonstrated nominal improvement in maintaining optimal mean blood pressure. 

In contrast, the improvement in maintaining optimal oxygen saturation was substantial, with the agent achieving 84.8% 
optimality compared to 50.7% in the historical data. This disparity aligns with the understanding that oxygen saturation
is more directly influenced by respiratory support, making it more amenable to optimization through continuous 
ventilator adjustment than other health measures.

## Conclusion
This study demonstrates the potential of batch-constrained reinforcement learning to enhance ventilator management for 
ICU patients. By leveraging historical data and enforcing clinically informed constraints on the action space, the RL 
agent successfully recommended ventilator settings that improved the maintenance of optimal oxygen saturation levels 
compared to historical clinical outcomes.

The nominal improvement in mean blood pressure suggests that broader physiological outcomes may require further 
refinements to the model, such as enhancing state representations or exploring alternative reward structures.

The outcome measures in this study were derived from predictions by a secondary neural network. While this is a common
approach in offline reinforcement learning, further work is needed to validate the generalizability and clinical 
applicability of these findings in a medical setting.

Nonetheless, these findings represent a meaningful step toward the development of intelligent, data-driven tools that 
can assist clinicians in delivering more personalized and responsive care to critically ill patients