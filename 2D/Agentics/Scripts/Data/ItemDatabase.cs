using UnityEngine;
using System.Linq;
using Agentics;

namespace Agentics {
    [CreateAssetMenu(fileName = "ItemDatabase", menuName = "2D Farming/Item Database")]
    public class ItemDatabase : BaseDatabase<Item>
    {
        public Item GetItem(string key)
        {
            return Entries.FirstOrDefault(item => item.Key == key);
        }
    }
}