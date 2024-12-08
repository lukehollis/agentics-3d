using System.Collections.Generic;


namespace Agentics
{
    [System.Serializable]
    public class DayPlan
    {
        public string day_overview;
        public List<DayPlanAction> actions;
    }

    [System.Serializable]
    public class DayPlanAction
    {
        public string action;
        public string emoji;
        public string location;
        public List<ActionTask> tasks;
    }

    // Like a subtask for a bigger general action
    [System.Serializable]
    public class ActionTask 
    {
        public string task;
        public string emoji;
        public string sub_location;
        public string tool;
        public string animation;
        public ToolArgs tool_args;
    }

    [System.Serializable]
    public class ToolArgs
    {
        public string tile_description;

        // Add other fields as necessary
    }

    [System.Serializable]
    public class ActionTaskList
    {
        public List<ActionTask> tasks;
    }
}