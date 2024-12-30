using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NeuralStateVisualization : MonoBehaviour
{
    void Start()
    {
        // Get particle system on this game object
        ParticleSystem[] rootParticles = GetComponents<ParticleSystem>();
        // Get all particle systems in children
        ParticleSystem[] childParticles = GetComponentsInChildren<ParticleSystem>(true);
        
        // Combine both arrays and process
        ParticleSystem[] allParticles = new ParticleSystem[rootParticles.Length + childParticles.Length];
        rootParticles.CopyTo(allParticles, 0);
        childParticles.CopyTo(allParticles, rootParticles.Length);

        foreach(ParticleSystem ps in allParticles)
        {
            ParticleSystemRenderer renderer = ps.GetComponent<ParticleSystemRenderer>();
            renderer.sortingOrder = 301;
            renderer.sortingLayerName = "UI3D";
        }
    }
}