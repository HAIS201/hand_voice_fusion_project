using UnityEngine;
using System.Globalization; // InvariantCulture 사용을 위한 네임스페이스

public class HandTracking : MonoBehaviour
{
    [Header("네트워크 입력")]
    public UDPReceiver udpReceive;   // UDPReceiver 컴포넌트에서 data 문자열을 받아옴

    [Header("시각화 설정")]
    public float scale = 0.01f;      // 좌표 스케일링 비율
    public float xOffset = 5f;       // x축 오프셋 (씬에서 조금 옆으로 띄우기)
    public float pointSize = 0.03f;  // 구(점) 크기

    private readonly Transform[] points = new Transform[21];
    private bool spawned;

    void Start()
    {
        // 간단한 디버그용: 21개의 구를 만들어서 손 랜드마크 위치를 표시
        for (int i = 0; i < 21; i++)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = $"P{i}";
            go.transform.SetParent(transform, false);
            go.transform.localScale = Vector3.one * pointSize;

            // 충돌은 필요 없어서 제거
            var col = go.GetComponent<Collider>();
            if (col) Destroy(col);

            // 머티리얼 설정 (초록색 구로 보이게)
            var renderer = go.GetComponent<Renderer>();
            renderer.material = new Material(Shader.Find("Standard"));
            renderer.material.color = Color.green;

            points[i] = go.transform;
        }
        spawned = true;
        Debug.Log("[HandTracking] Spawned 21 points under HandRoot.");
    }

    void Update()
    {
        if (!spawned) return;

        if (udpReceive == null)
        {
            // 인스펙터에 UDPReceiver를 안 넣어줬을 때 경고용
            if (Time.frameCount % 120 == 0)
                Debug.LogWarning("[HandTracking] udpReceive is NULL. Drag the object with UDPReceiver here.");
            return;
        }

        // UDPReceiver에서 받은 최신 문자열
        string s = udpReceive.data;
        if (string.IsNullOrEmpty(s) || s.Length < 5) return;

        // 좌표 배열만 처리하고, "FORWARD"/"ATTACK" 같은 명령 문자열은 무시
        if (s[0] != '[') return;

        // 양 끝의 대괄호 제거
        s = s.Substring(1, s.Length - 2);

        var parts = s.Split(',');
        if (parts.Length < 63) return; // 21 * 3

        for (int i = 0; i < 21; i++)
        {
            float x = SafeParse(parts[i * 3 + 0]);
            float y = SafeParse(parts[i * 3 + 1]);
            float z = SafeParse(parts[i * 3 + 2]);

            // 주의: Python 쪽에서 y 축을 한 번 뒤집어서 보냈기 때문에,
            // 여기서는 기존에 맞춰둔 좌표계 방식 그대로 사용
            Vector3 p = new Vector3(xOffset - x * scale, y * scale, z * scale);
            points[i].localPosition = p;
        }
    }

    float SafeParse(string raw)
    {
        // 시스템 로캘에 상관없이 항상 같은 방식으로 float 파싱 (소수점/콤마 문제 방지)
        float v;
        if (float.TryParse(raw.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out v))
            return v;
        return 0f;
    }

    public Transform[] GetPoints() => points;
}
