using System.Collections.Generic;
using UnityEngine;
using System.Linq;

using Agentics;


/// <summary>
/// This is a modified version of the original from HappyHarvest.
/// This is a base class that allow to define a Database that will link a name/string id to a given object.
/// Useful for thing like linking item to their id so we can retrieve an item by its id (e.g. when reading save).
/// See ItemDatabase for sample of how those are created.
/// </summary>


namespace Agentics {

    public interface IDatabaseEntry
    {
        string Key { get; }
    }

    /// <summary>
    /// This is a base class that allow to define a Database that will link a name/string id to a given object.
    /// Useful for thing like linking item to their id so we can retrieve an item by its id (e.g. when reading save).
    /// See ItemDatabase for sample of how those are created.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class BaseDatabase<T> : ScriptableObject where T: class, IDatabaseEntry
    {
        [SerializeReference]
        public List<T> Entries; 

        private Dictionary<string, T> m_LookupDictionnary;

        public T GetFromID(string uniqueID)
        {
            if (m_LookupDictionnary.TryGetValue(uniqueID, out var entry))
            {
                return entry;
            }

            return null;
        }

        //This need to be called by whoever use this database to rebuild the lookup.
        //Used to use OnAfterDeserialize but we cannot control the order of deserialization, and Item could be
        //deserialized AFTER the database is, so the unique ID for that item was not ready yet.
        public void Init()
        {
            m_LookupDictionnary = new Dictionary<string, T>();

            //rebuild the lookup
            foreach (var entry in Entries)
            {
                if (entry == null)
                {
                    continue;
                }
                
                //TryAdd as there seems to be case where entries are duplicated. My guess is when drag and dropping, it
                //will first duplicate an entry, which trigger a deserialize THEN assign the new entry, which led to
                //error.
                m_LookupDictionnary.TryAdd(entry.Key, entry);
            }
        }

        public T Get(string key)
        {
            if (Entries == null) return null;
            return Entries.FirstOrDefault(entry => entry.Key == key);
        }
    }
}