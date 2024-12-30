using UnityEngine;

public class EmitParticles : MonoBehaviour
{
    [SerializeField] private ParticleSystem particles;
    [SerializeField] private int particleCount = 1;


    public void Emit()
    {
        particles.Emit(particleCount);
    }
}