using UnityEngine;
using Agentics;

[CreateAssetMenu(fileName = "WateringCan", menuName = "2D Farming/Items/Watering Can")]
public class WateringCan : Item
{
    public override bool CanUse(Vector3Int target)
    {
        return GameController.Instance.EnvironmentManager.IsTilled(target);
    }

    public override bool Use(Vector3Int target)
    {
        GameController.Instance.EnvironmentManager.WaterAt(target);
        return true;
    }
}
