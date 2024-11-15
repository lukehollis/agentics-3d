using UnityEngine;
using System.Collections.Generic;

namespace Agentics.Utils
{
    public static class AgentUtils
    {
        public static Vector2 GridToWorldPosition(Vector2Int gridPos, float cellSize)
        {
            return new Vector2(
                gridPos.x * cellSize - (cellSize * 0.5f),
                gridPos.y * cellSize - (cellSize * 0.5f)
            );
        }

        public static float CalculateActionUtility(
            string actionType,
            float priority,
            float emotionalWeight,
            float timeWeight)
        {
            return (priority + emotionalWeight + timeWeight) / 3f;
        }
    }
}