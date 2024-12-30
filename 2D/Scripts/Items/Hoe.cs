using UnityEngine;
using Agentics;

[CreateAssetMenu(fileName = "Hoe", menuName = "2D Farming/Items/Hoe")]
public class Hoe : Item
{
    public override bool CanUse(Vector3Int target)
    {
        return GameController.Instance?.EnvironmentManager != null && GameController.Instance.EnvironmentManager.IsTillable(target);
    }

    public override bool Use(Vector3Int target)
    {
        GameController.Instance.EnvironmentManager.PlowTile(target);
        return true;
    }
}
