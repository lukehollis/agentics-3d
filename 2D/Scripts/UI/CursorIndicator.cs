using UnityEngine;
using System.Collections;

public class CursorIndicator : MonoBehaviour
{
    public static CursorIndicator Instance { get; private set; }

    private SpriteRenderer spriteRenderer;
    private Coroutine fadeCoroutine;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {
        spriteRenderer = GetComponent<SpriteRenderer>();
        spriteRenderer.sortingOrder = 10; // Set a high sorting order
        gameObject.SetActive(false);
    }

    public void ShowAtPosition(Vector3 position, float tileSize=1f)
    {
        if (fadeCoroutine != null)
        {
            StopCoroutine(fadeCoroutine);
        }

        // Snap position to the center of the nearest tile
        position.x = Mathf.Floor(position.x / tileSize) * tileSize + tileSize / 2;
        position.y = Mathf.Floor(position.y / tileSize) * tileSize + tileSize / 2;
        position.z = -1; // Ensure the cursor is in front of other objects

        transform.position = position;
        gameObject.SetActive(true);
        spriteRenderer.color = new Color(spriteRenderer.color.r, spriteRenderer.color.g, spriteRenderer.color.b, 1f);
        fadeCoroutine = StartCoroutine(FadeOut());
    }

    private IEnumerator FadeOut()
    {
        float duration = 1f;
        float elapsedTime = 0f;
        Color initialColor = spriteRenderer.color;

        while (elapsedTime < duration)
        {
            elapsedTime += Time.deltaTime;
            float alpha = Mathf.Lerp(0.4f, 0f, elapsedTime / duration);
            spriteRenderer.color = new Color(initialColor.r, initialColor.g, initialColor.b, alpha);
            yield return null;
        }

        gameObject.SetActive(false);
    }
}