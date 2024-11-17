using UnityEngine;
using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Agentics;

namespace Agentics
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

    public class PlanSystem : MonoBehaviour
    {
        private Brain agentBrain;
        private RewardSystem rewardSystem;
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
            agentBrain = GetComponent<Brain>();
            rewardSystem = GetComponent<RewardSystem>();
            motivationSystem = GetComponent<MotivationSystem>();
        }

        public void ProcessNewPlan(string planJson)
        {
            currentPlan = ConvertToAgentPlan(planJson);
            
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

        private AgentPlan ConvertToAgentPlan(string planJson)
        {
            var agentPlan = JsonUtility.FromJson<AgentPlan>(planJson);
            
            // Initialize any null collections
            if (agentPlan.actions == null)
                agentPlan.actions = new List<PlanAction>();
            if (agentPlan.actionWeights == null)
                agentPlan.actionWeights = new Dictionary<string, float>();

            // Ensure all actions have initialized subtasks lists
            foreach (var action in agentPlan.actions)
            {
                if (action.subtasks == null)
                    action.subtasks = new List<ActionSubtask>();
                    
                // Initialize completion status if not set
                action.completionStatus = 0f;
            }

            // Initialize overall completion rate
            agentPlan.completionRate = 0f;

            return agentPlan;
        }

        private float GetTimeBasedWeight(float duration)
        {
            // Get current hour from the game's timeline
            int currentHour = Timeline.Instance.currentDate.Hour;
            
            // Base weight starts at 1.0
            float weight = 1.0f;
            
            // Reduce priority for long duration tasks during non-optimal hours
            if (currentHour >= 19 || currentHour < 6) // Night time
            {
                // Penalize long duration tasks at night
                weight = Mathf.Lerp(1.0f, 0.2f, duration / planUpdateInterval);
            }
            else if (currentHour >= 6 && currentHour < 9) // Early morning
            {
                // Slightly favor shorter tasks in early morning
                weight = Mathf.Lerp(1.0f, 0.6f, duration / planUpdateInterval);
            }
            else // Day time (9-19)
            {
                // Favor medium duration tasks during the day
                float normalizedDuration = duration / planUpdateInterval;
                weight = 1.0f - Mathf.Abs(normalizedDuration - 0.5f);
            }
            
            return Mathf.Clamp01(weight);
        }

        private void UpdatePlanCompletionRate()
        {
            if (currentPlan == null || currentPlan.actions == null || currentPlan.actions.Count == 0)
                return;

            float totalCompletion = 0f;
            float totalWeight = 0f;

            foreach (var action in currentPlan.actions)
            {
                float weight = 1f;
                if (currentPlan.actionWeights.ContainsKey(action.actionType))
                {
                    weight = currentPlan.actionWeights[action.actionType];
                }

                totalCompletion += action.completionStatus * weight;
                totalWeight += weight;
            }

            currentPlan.completionRate = totalWeight > 0 ? totalCompletion / totalWeight : 0f;
        }
    }
}