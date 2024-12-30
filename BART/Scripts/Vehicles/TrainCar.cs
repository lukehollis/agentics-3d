using UnityEngine;

public class TrainCar : MonoBehaviour
{
    public Transform frontConnector;
    public Transform rearConnector;
    public float carLength = 15f; // Distance between connectors
    
    private void OnDrawGizmosSelected()
    {
        if (frontConnector && rearConnector)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawSphere(frontConnector.position, 0.5f);
            Gizmos.DrawSphere(rearConnector.position, 0.5f);
            Gizmos.DrawLine(frontConnector.position, rearConnector.position);
        }
    }
}