using System;

namespace UnityEngine.Tilemaps
{
    [ExecuteInEditMode]
    public class TilePaletteHelper : MonoBehaviour
    {
        private Tilemap tilemap;
        private TileBase[] tiles;
     
#if UNITY_EDITOR        
        private void Start()
        {
            tilemap = GetComponentInChildren<Tilemap>();
            tilemap.CompressBounds();
        }

        private void OnDrawGizmos()
        {
            if (tilemap == null)
                return;
            
            var bounds = tilemap.cellBounds;
            var boundsSize = bounds.size.x * bounds.size.y * bounds.size.z;
            if (tiles == null || boundsSize != tiles.Length)
                Array.Resize(ref tiles, boundsSize);
            tilemap.GetTilesBlockNonAlloc(bounds, tiles);
            
            var i = 0;
            foreach (var position in bounds.allPositionsWithin)
            {
                var tile = tiles[i++];
                if (tile == null)
                    continue;

                if (tilemap.GetSprite(position) != null)
                    continue;
                
                var localPosition = tilemap.CellToLocalInterpolated(position + tilemap.tileAnchor);
                Gizmos.DrawIcon(localPosition, TilePaletteIconsPreference.GetTexturePath(tile.GetType()));
            }
        }
#endif        
    }
}