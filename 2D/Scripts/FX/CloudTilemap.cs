using UnityEngine;
using UnityEngine.Tilemaps;

public class CloudTilemap : MonoBehaviour
{
    public Tilemap cloudTilemap;
    [SerializeField] private readonly float moveSpeed = 0.1f;

    private void Start()
    {
        // Get the reference to the Tilemap component
        cloudTilemap = GetComponent<Tilemap>();
    }

    private void Update()
    {
        // Move the cloud tiles to the right
        Vector3 currentPosition = cloudTilemap.transform.position;
        currentPosition.x += moveSpeed * Time.deltaTime;

        // Check if the cloud tiles have moved off-screen
        if (currentPosition.x > cloudTilemap.cellBounds.xMax)
        {
            // Reset the position to the left side of the tilemap
            currentPosition.x = cloudTilemap.cellBounds.xMin;
        }

        // Update the position of the cloud tilemap
        cloudTilemap.transform.position = currentPosition;
    }
}