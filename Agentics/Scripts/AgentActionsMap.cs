using UnityEngine;
using System;
using System.Collections.Generic;

[Serializable]
public enum ActionType
{
    Movement,
    Interaction,
    TaskSelection,
    EmotionalResponse
}

[Serializable]
public class ActionMapping
{
    public ActionType actionType;
    public int actionIndex;
    public int actionSpaceSize;
}

public class AgentActionsMap : MonoBehaviour
{
    [Header("Action Space Configuration")]
    public List<ActionMapping> actionMappings = new List<ActionMapping>();
    
    [Header("Movement Settings")]
    public bool useDiscretizedMovement = true;
    public int movementDirections = 4; // 4 for cardinal, 8 for including diagonals
    
    private AgentBrain agentBrain;
    private NavMeshAgent agent;
    
    private void Awake()
    {
        agentBrain = GetComponent<AgentBrain>();
        agent = GetComponent<NavMeshAgent>();
        
        // Setup default action mappings if none exist
        if (actionMappings.Count == 0)
        {
            actionMappings.Add(new ActionMapping 
            { 
                actionType = ActionType.Movement, 
                actionIndex = 0,
                actionSpaceSize = movementDirections + 1 // +1 for no movement
            });
            
            actionMappings.Add(new ActionMapping 
            { 
                actionType = ActionType.Interaction, 
                actionIndex = 1,
                actionSpaceSize = 2 // Binary: interact or not
            });
        }
    }

    public int GetTotalActionSpaceSize()
    {
        int totalSize = 0;
        foreach (var mapping in actionMappings)
        {
            totalSize += mapping.actionSpaceSize;
        }
        return totalSize;
    }

    public Vector2 GetCurrentMovement()
    {
        if (agent != null)
        {
            return new Vector2(agent.velocity.x, agent.velocity.y).normalized;
        }
        return Vector2.zero;
    }

    private Vector2 DiscreteDirectionToVector(int action, int directions)
    {
        if (action == 0) return Vector2.zero;
        
        float angle = (360f / directions) * (action - 1);
        return Quaternion.Euler(0, 0, angle) * Vector2.right;
    }

    private int VectorToDiscreteDirection(Vector2 vector, int directions)
    {
        if (vector.magnitude < 0.1f) return 0;
        
        float angle = Vector2.SignedAngle(Vector2.right, vector);
        if (angle < 0) angle += 360f;
        
        int action = Mathf.RoundToInt(angle / (360f / directions)) + 1;
        return action;
    }

    public void ExecuteActions(Unity.MLAgents.Actuators.ActionBuffers actions)
    {
        var discreteActions = actions.DiscreteActions;
        
        foreach (var mapping in actionMappings)
        {
            switch (mapping.actionType)
            {
                case ActionType.Movement:
                    ExecuteMovement(discreteActions[mapping.actionIndex]);
                    break;
                    
                case ActionType.Interaction:
                    ExecuteInteraction(discreteActions[mapping.actionIndex]);
                    break;
                    
                case ActionType.TaskSelection:
                    ExecuteTaskSelection(discreteActions[mapping.actionIndex]);
                    break;
                    
                case ActionType.EmotionalResponse:
                    ExecuteEmotionalResponse(discreteActions[mapping.actionIndex]);
                    break;
            }
        }
    }

    private void ExecuteMovement(int action)
    {
        if (!useDiscretizedMovement) return;
        Vector2 movement = DiscreteDirectionToVector(action, movementDirections);
        Vector3 targetPosition = transform.position + new Vector3(movement.x, movement.y, 0);
        agent.SetDestination(targetPosition);
    }

    private void ExecuteInteraction(int action)
    {
        if (action == 1)
        {
            agentBrain.SendMessage("TryInteract");
        }
    }

    private void ExecuteTaskSelection(int action)
    {
        // Implement task selection logic
    }

    private void ExecuteEmotionalResponse(int action)
    {
        // Implement emotional response logic
    }
}