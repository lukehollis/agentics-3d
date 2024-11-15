using UnityEngine;
using System;
using System.Collections.Generic;
using Unity.MLAgents;

namespace Agentics.Planning
{
    [System.Serializable]
    public class AgentPlan
    {
        public string overview;
        public List<PlanAction> actions;
        public float completionRate;
        public Dictionary<string, float> actionWeights;
    }

    [System.Serializable]
    public class PlanAction
    {
        public string actionType;
        public string targetLocation;
        public string emoji;
        public float duration;
        public float priority;
        public float completionStatus;
        public List<ActionSubtask> subtasks;
    }

    [System.Serializable]
    public class ActionSubtask
    {
        public string taskType;
        public Vector2 relativePosition;
        public string toolType;
        public Dictionary<string, object> parameters;
        public float completionStatus;
    }

    public class AgentPlanSystem : MonoBehaviour
    {
        private AgentBrain agentBrain;
        private AgentRewardSystem rewardSystem;
        private MotivationSystem motivationSystem;
        
        [Header("Plan Settings")]
        public float planUpdateInterval = 300f; // 5 minutes
        public float taskCompletionThreshold = 0.9f;
        
        private AgentPlan currentPlan;
        private PlanAction currentAction;
        private ActionSubtask currentTask;
        private float lastPlanUpdate;

        private void Awake()
        {
            agentBrain = GetComponent<AgentBrain>();
            rewardSystem = GetComponent<AgentRewardSystem>();
            motivationSystem = GetComponent<MotivationSystem>();
        }

        public void ProcessNewPlan(string planJson)
        {
            var dayPlan = JsonUtility.FromJson<DayPlan>(planJson);
            currentPlan = ConvertToAgentPlan(dayPlan);
            
            // Update motivation based on plan overview
            motivationSystem.ProcessPlanContext(currentPlan.overview);
            
            // Initialize action weights based on motivation
            UpdateActionWeights();
        }

        private void UpdateActionWeights()
        {
            foreach (var action in currentPlan.actions)
            {
                // Calculate priority based on motivation and current state
                float emotionalWeight = motivationSystem.GetEmotionalWeightForAction(action.actionType);
                float needWeight = motivationSystem.GetNeedWeightForAction(action.actionType);
                float timeWeight = GetTimeBasedWeight(action.duration);
                
                action.priority = (emotionalWeight + needWeight + timeWeight) / 3f;
            }
        }

        public void AddToObservations(Unity.MLAgents.Sensors.VectorSensor sensor)
        {
            if (currentPlan != null)
            {
                sensor.AddObservation(currentPlan.completionRate);
                sensor.AddObservation(currentAction?.completionStatus ?? 0f);
                sensor.AddObservation(currentTask?.completionStatus ?? 0f);
            }
        }

        private void OnActionComplete(PlanAction action, float successRate)
        {
            // Update completion status
            action.completionStatus = successRate;
            UpdatePlanCompletionRate();

            // Apply rewards
            float reward = rewardSystem.taskCompletionReward * successRate;
            
            // Add motivation-based reward modifier
            float motivationBonus = motivationSystem.GetActionCompletionBonus(action.actionType);
            reward *= (1f + motivationBonus);
            
            agentBrain.AddReward(reward);
        }
    }
}