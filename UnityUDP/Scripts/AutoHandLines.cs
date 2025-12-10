using UnityEngine;
using System.Collections.Generic;

[RequireComponent(typeof(HandTracking))]
public class AutoHandLines : MonoBehaviour
{
    [Header("라인 설정")]
    public float lineWidth = 0.01f;   // 손가락 뼈대를 그릴 라인 두께

    private HandTracking ht;

    // MediaPipe Hands 구조에 맞춘 손 뼈대 연결 정보
    private readonly int[][] edges = new int[][]
    {
        // 손바닥 메인 라인: 0-5-9-13-17
        new[]{0,5}, new[]{5,9}, new[]{9,13}, new[]{13,17},

        // 엄지: 0-1-2-3-4 (0-1은 옵션이지만 여기서는 연결)
        new[]{0,1}, new[]{1,2}, new[]{2,3}, new[]{3,4},

        // 검지: 5-6-7-8
        new[]{5,6}, new[]{6,7}, new[]{7,8},

        // 중지: 9-10-11-12
        new[]{9,10}, new[]{10,11}, new[]{11,12},

        // 약지: 13-14-15-16
        new[]{13,14}, new[]{14,15}, new[]{15,16},

        // 새끼손가락: 17-18-19-20
        new[]{17,18}, new[]{18,19}, new[]{19,20}
    };

    private List<LineRenderer> lines = new();

    void Start()
    {
        ht = GetComponent<HandTracking>();

        // 각 edge마다 LineRenderer를 하나씩 만들어서 손가락 뼈대를 그림
        for (int i = 0; i < edges.Length; i++)
        {
            var go = new GameObject($"Edge_{edges[i][0]}_{edges[i][1]}");
            go.transform.SetParent(transform, false);

            var lr = go.AddComponent<LineRenderer>();
            lr.useWorldSpace = false;                // HandRoot 기준 로컬 좌표 사용
            lr.positionCount = 2;
            lr.startWidth = lineWidth;
            lr.endWidth = lineWidth;
            lr.material = new Material(Shader.Find("Sprites/Default")); // 간단한 기본 머티리얼
            lines.Add(lr);
        }
    }

    void Update()
    {
        var pts = ht.GetPoints();
        if (pts == null || pts.Length < 21) return;

        // 21개 랜드마크 사이를 edge 정의에 따라 라인으로 연결
        for (int i = 0; i < edges.Length; i++)
        {
            int a = edges[i][0];
            int b = edges[i][1];
            var lr = lines[i];

            lr.SetPosition(0, pts[a].localPosition);
            lr.SetPosition(1, pts[b].localPosition);
        }
    }
}
