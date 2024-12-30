using System;
using System.Collections.Generic;
using Agentics;

[Serializable]
public class ReceivedMessageData
{
    public string type;
    public int character_id;
    public string message;
    public TileResponseData response;
}

[Serializable]
public class ActionTasksRequest
{
    public string type;
    public int character_id;
    public string game_session_uuid; // Add this line
    public DayPlanAction action;
}

[Serializable]
public class ActionTasksResponse
{
    public string type;
    public int character_id;
    public string message;
}

[Serializable]
public class CharacterPlanRequest
{
    public string type;
    public int character_id;
    public string game_world_date;
    public string game_world_place_setting;
    public List<string> waypoint_names;
    public List<string> npc_names;
    public List<string> player_names;
}

[Serializable]
public class CharacterConversationMessage
{
    public string type;
    public string user_uuid;
    public string game_session_uuid; // Add this line
    public int character_id;
    public string message;
    public string game_world_date_time;
    public string game_world_place_setting;
    public string current_day_plan;
    public string current_action;
    public string current_location;
}

[Serializable]
public class MemoryEvent
{
    public string type;
    public int character_id;
    public string description;
    public string location;
    public int priority;
    public string user_id;
}

[Serializable]
public class CharacterSelfReflectionMessage
{
    public string type;
    public int character_id;
    public string image_data;
    public string day_plan;
    public string current_action;
    public string last_completed_task;
}

[Serializable]
public class TileGeneratedResponse
{
    public string type;
    public int character_id;
    public TileResponseData response;
}

[Serializable]
public class TileResponseData
{
    public string b64_json;
}

[Serializable]
public class TileGenerationRequest
{
    public string type;
    public int character_id;
    public string tile_description;
}

[Serializable]
public class SaveGameMessage
{
    public string type;
    public string user_uuid;
    public string game_session_uuid;
    public string save_data;
}

[Serializable]
public class LoadGameResponseData
{
    public string type;
    public SaveData message;
}
