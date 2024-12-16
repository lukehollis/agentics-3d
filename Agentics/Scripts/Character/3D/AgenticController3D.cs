using UnityEngine;
using UnityEngine.AI;
using System.Collections;
using System.Collections.Generic;
using Agentics;

namespace Agentics
{
    [RequireComponent(typeof(AgenticCharacter3D))]
    [RequireComponent(typeof(NavMeshAgent))]
    public class AgenticController3D : AgenticController
    {
        public NavMeshAgent agent;
        public NavMeshObstacle navMeshObstacle;
        
        protected override void SetupComponents()
        {
            character = GetComponent<AgenticCharacter3D>();
            agent = GetComponent<NavMeshAgent>();
            navMeshObstacle = GetComponent<NavMeshObstacle>();
            
            // Configure NavMesh Agent
            agent.speed = walkSpeed;
            agent.stoppingDistance = stoppingDistance;
            agent.updateRotation = true;
        }
        
        protected override void CheckMovement()
        {
            // Implement 3D movement logic here
        }

        public override void SetDestination(Vector3 position)
        {
            agent.SetDestination(position);
        }

        public override void Interact()
        {
            // Implement interaction logic here
        }

        public override void UpdatePlan(string planJson)
        {
            if (string.IsNullOrEmpty(planJson)) return;

            currentDayPlan = JsonUtility.FromJson<DayPlan>(planJson);
            
            if (currentDayPlan != null && currentDayPlan.actions != null && currentDayPlan.actions.Count > 0)
            {
                currentDayPlanAction = currentDayPlan.actions[0];
                
                // Update task indicator if available
                if (taskIndicator != null)
                {
                    taskIndicator.SetActive(true);
                }
            }
        }

        protected override IEnumerator ExecuteCurrentAction()
        {
            if (currentDayPlanAction == null) yield break;

            isInteracting = true;
            
            // Get location from action
            var location = GameObject.Find(currentDayPlanAction.location);
            if (location != null)
            {
                // Move to location
                SetDestination(location.transform.position);
                
                // Wait until we reach the destination
                while (agent.pathStatus == NavMeshPathStatus.PathPartial || 
                       agent.remainingDistance > agent.stoppingDistance)
                {
                    yield return null;
                }
                
                // Perform action at location
                yield return new WaitForSeconds(taskDuration);
            }
            
            // Action complete
            isInteracting = false;
            
            // Remove completed action and get next one
            if (currentDayPlan != null && currentDayPlan.actions != null && currentDayPlan.actions.Count > 0)
            {
                currentDayPlan.actions.RemoveAt(0);
                currentDayPlanAction = currentDayPlan.actions.Count > 0 ? currentDayPlan.actions[0] : null;
            }
        }
    }
}