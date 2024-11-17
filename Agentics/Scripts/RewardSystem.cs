using UnityEngine;
using Unity.MLAgents;
using Agentics;

namespace Agentics
{
    public class RewardSystem : MonoBehaviour
    {
        [Header("Movement Rewards")]
        public float efficientMovementReward = 0.1f;
        public float inefficientMovementPenalty = -0.05f;
        public float collisionPenalty = -0.1f;
        
        [Header("Task Rewards")]
        public float taskCompletionReward = 1.0f;
        public float taskProgressReward = 0.2f;
        public float taskFailurePenalty = -0.5f;
        
        [Header("Interaction Rewards")]
        public float successfulInteractionReward = 0.3f;
        public float failedInteractionPenalty = -0.1f;
        
        [Header("Time Management")]
        public float idleTimePenalty = -0.01f;
        public float maxIdleTime = 5f;
        
        private Brain agentBrain;
        private UnityEngine.AI.NavMeshAgent agent;
        private Vector3 lastPosition;
        private float lastRewardTime;
        private float idleTimer;

        private void Awake()
        {
            agentBrain = GetComponent<Brain>();
            agent = GetComponent<UnityEngine.AI.NavMeshAgent>();
            lastPosition = transform.position;
            lastRewardTime = Time.time;
        }

        public void Initialize()
        {
            ResetTracking();
        }

        private void ResetTracking()
        {
            lastPosition = transform.position;
            lastRewardTime = Time.time;
            idleTimer = 0f;
        }

        public void UpdateRewards()
        {
            EvaluateMovementEfficiency();
            EvaluateTaskProgress();
            EvaluateTimeManagement();
            
            lastPosition = transform.position;
            lastRewardTime = Time.time;
        }

        private void EvaluateMovementEfficiency()
        {
            if (agent.hasPath)
            {
                float distanceMoved = Vector3.Distance(transform.position, lastPosition);
                float directDistance = Vector3.Distance(transform.position, agent.destination);
                
                // Calculate efficiency ratio
                float efficiencyRatio = directDistance > 0 ? distanceMoved / directDistance : 0;
                
                if (efficiencyRatio <= 1.1f) // Allow for slight inefficiency
                {
                    agentBrain.AddReward(efficientMovementReward * Time.deltaTime);
                }
                else
                {
                    agentBrain.AddReward(inefficientMovementPenalty * Time.deltaTime);
                }
            }
        }

        private void EvaluateTaskProgress()
        {
            // Implement based on your task system
            // Example:
            if (agent.remainingDistance < agent.stoppingDistance && agent.hasPath)
            {
                agentBrain.AddReward(taskProgressReward);
                agent.ResetPath();
            }
        }

        private void EvaluateTimeManagement()
        {
            if (agent.velocity.magnitude < 0.1f)
            {
                idleTimer += Time.deltaTime;
                if (idleTimer > maxIdleTime)
                {
                    agentBrain.AddReward(idleTimePenalty * Time.deltaTime);
                }
            }
            else
            {
                idleTimer = 0f;
            }
        }

        public void OnTaskComplete(float successRatio)
        {
            agentBrain.AddReward(taskCompletionReward * successRatio);
        }

        public void OnTaskFailed()
        {
            agentBrain.AddReward(taskFailurePenalty);
        }

        public void OnSuccessfulInteraction()
        {
            agentBrain.AddReward(successfulInteractionReward);
        }

        public void OnFailedInteraction()
        {
            agentBrain.AddReward(failedInteractionPenalty);
        }

        public void OnCollision()
        {
            agentBrain.AddReward(collisionPenalty);
        }
    }
}