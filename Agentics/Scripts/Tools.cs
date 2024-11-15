using UnityEngine;

namespace Agentics.Tools
{
    public interface IAgentTool
    {
        string ToolType { get; }
        bool CanUse(Vector3 position);
        float UseAt(Vector3 position, object[] args);
        void OnToolComplete();
    }
}