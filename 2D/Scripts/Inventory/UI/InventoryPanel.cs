using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Agentics;

public class InventoryPanel : MonoBehaviour
{
    public Image itemImage;
    public TextMeshProUGUI itemTitle;
    public TextMeshProUGUI itemSubtitle;
    public TextMeshProUGUI itemDescription;

    private void Start() {
        ClearItemDetails();
        
        // Subscribe to the active item changed event
        var inventory = GameController.Instance.player.inventory;
        if (inventory != null) {
            inventory.OnActiveItemChanged += OnActiveItemChanged;
            
            // Set initial active item
            Item activeItem = inventory.GetActiveItem();
            if (activeItem != null) {
                SetItemDetails(activeItem.DetailImage, activeItem.DisplayName, activeItem.ScientificName, activeItem.Description);
            }
        }
    }

    private void OnDestroy() {
        // Unsubscribe from the event when the panel is destroyed
        var inventory = GameController.Instance.player.inventory;
        if (inventory != null) {
            inventory.OnActiveItemChanged -= OnActiveItemChanged;
        }
    }

    private void OnActiveItemChanged() {
        var inventory = GameController.Instance.player.inventory;
        Item activeItem = inventory.GetActiveItem();
        
        if (activeItem != null) {
            SetItemDetails(activeItem.DetailImage, activeItem.DisplayName, activeItem.ScientificName, activeItem.Description);
        } else {
            ClearItemDetails();
        }
    }

    public void SetItemDetails(Sprite image, string title, string scientificName, string description)
    {
        if (image != null) {
            itemImage.sprite = image;
        }
        if (title != null) {
            itemTitle.text = title;
        }
        if (scientificName != null) {
            itemSubtitle.text = scientificName;
        }
        if (description != null) {
            itemDescription.text = description;
        }
    }

    public void ClearItemDetails()
    {
        itemImage.sprite = null;
        itemTitle.text = "";
        itemSubtitle.text = "";
        itemDescription.text = "";
    }
}