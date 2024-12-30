using System;

namespace UnityEngine.Tilemaps
{
    internal static class TilePaletteIconsPreference
    {
        public static string GetTexturePath(Type tileType)
        {
            if (!tileType.IsSubclassOf(typeof(TileBase)))
                return String.Empty;

            return "UnityEngine/Tilemaps/Tile Icon";
        }
    }
}