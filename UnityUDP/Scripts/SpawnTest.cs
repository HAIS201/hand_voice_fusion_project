using UnityEngine;

public class SpawnTest : MonoBehaviour
{
    public float pointSize = 0.05f;

    void Start()
    {
        for (int i = 0; i < 21; i++)
        {
            var g = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            g.name = $"TEST_{i}";
            g.transform.SetParent(transform, false);
            g.transform.localScale = Vector3.one * pointSize;
            g.transform.localPosition = new Vector3(i * 0.2f, 0, 0);
        }
        Debug.Log("[SpawnTest] Spawned 21 test spheres.");
    }
}
