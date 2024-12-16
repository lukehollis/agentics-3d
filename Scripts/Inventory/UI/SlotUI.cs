using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Agentics;

public class SlotUI : MonoBehaviour
{
    public int slotID;
    public Image itemIcon;
    public TextMeshProUGUI quantityText;
    public Inventory inventory;
    [SerializeField] private GameObject highlightObject;
    [SerializeField] public InventoryItem currentItem;

    private void Awake()
    {
        // Find Image component in child named "Icon"
        itemIcon = transform.Find("Icon")?.GetComponent<Image>();
        if (itemIcon == null) Debug.LogError($"SlotUI: Missing Icon child with Image component on {gameObject.name}");

        // Find TextMeshProUGUI component in child named "Quantity"
        quantityText = transform.Find("Quantity")?.GetComponent<TextMeshProUGUI>();
        if (quantityText == null) Debug.LogError($"SlotUI: Missing DetailBottom child with TextMeshProUGUI component on {gameObject.name}");

        highlightObject = transform.Find("Checkmark")?.gameObject;
        if (highlightObject == null) Debug.LogError($"SlotUI: Missing Highlight child with GameObject component on {gameObject.name}");
    }


    public void SetItem(InventoryItem item)
    {
        if (item != null)
        {
            currentItem = item;
            itemIcon.sprite = item.Item.ItemSprite;
            itemIcon.color = new Color(1, 1, 1, 1);
            quantityText.text = item.Quantity.ToString();
        }
        else
        {
            SetEmpty();
        }
    }

    public void SetEmpty()
    {
        currentItem = null;
        itemIcon.sprite = null;
        itemIcon.color = new Color(1, 1, 1, 0);
        quantityText.text = "";
    }

    public void SetHighlight(bool isOn)
    {
        if (highlightObject != null)
        {
            highlightObject.SetActive(isOn);
        }
    }

    public void OnItemClick()
    {
        if (currentItem != null)
        {
            SimulationController.Instance.player.inventory.SetActiveItem(currentItem.Item);
        }
    }
}