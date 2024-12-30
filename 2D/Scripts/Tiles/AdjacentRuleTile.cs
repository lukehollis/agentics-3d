using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Tilemaps;


/// <summary>
/// This special tile add a third rule : adjacent tiles, so you can match rules with other tile than this one.
/// See the cliff tile for an example.
/// </summary>
[CreateAssetMenu]
public class AdjacentRuleTile : RuleTile<AdjacentRuleTile.Neighbor>
{
    public class Neighbor : RuleTile.TilingRule.Neighbor
    {
        public const int Adjacent = 3;
    }

    public TileBase[] AdjacentTiles;

    public override bool RuleMatch(int neighbor, TileBase other)
    {

        switch (neighbor)
        {
            case Neighbor.Adjacent:
                return AdjacentTiles.Contains(other);
        }

        return base.RuleMatch(neighbor, other);
    }
}
