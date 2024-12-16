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
        Item activeItem = SimulationController.Instance.player.inventory.GetActiveItem();
        if (activeItem != null) {
            SetItemDetails(activeItem.DetailImage, activeItem.DisplayName, activeItem.ScientificName, activeItem.Description);
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