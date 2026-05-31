# -*- coding: utf-8 -*-
"""
조기 융합(Early Fusion)과 후기 융합(Late Fusion) 결과 비교 스크립트
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 프로젝트 루트 및 출력 디렉터리 설정
ROOT = Path(r"D:\pycharm\hand\fusion_project")
OUT_DIR = ROOT / "outputs"


def extract_number_from_line(line):
    """한 줄 문자열에서 첫 번째 숫자(float)를 추출"""
    import re
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    return float(numbers[0]) if numbers else None


def read_report(filename):
    """
    학습 보고서(txt)를 읽어서 주요 지표를 dict 형태로 반환
    - 조기/후기 융합 학습 스크립트에서 저장한 한글 로그를 파싱
    """
    report_path = OUT_DIR / filename

    if not report_path.exists():
        print(f"경고: 보고서 파일이 존재하지 않습니다: {filename}")
        return {}

    results = {}

    with open(report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # 테스트 세트 이동 정확도
        if "테스트 세트 이동 정확도" in line:
            results["test_move_acc"] = extract_number_from_line(line)

        # 테스트 세트 행동 정확도
        elif "테스트 세트 행동 정확도" in line:
            results["test_act_acc"] = extract_number_from_line(line)

        # C_fusion 테스트 세트 CSR
        elif "C_fusion 테스트 세트 CSR" in line:
            results["cfusion_csr"] = extract_number_from_line(line)

        # 환경별 CSR (예: "E1: 0.8000")
        elif line.startswith("E") and ":" in line:
            parts = line.split(":")
            if len(parts) == 2:
                env = parts[0].strip()
                value = extract_number_from_line(parts[1])
                if value is not None:
                    results[env] = value

    return results


def compare_fusion_strategies():
    """조기 융합과 후기 융합의 성능을 비교하고 시각화 및 리포트 저장"""
    print("=" * 80)
    print("조기 융합 vs 후기 융합 결과 비교 분석")
    print("=" * 80)

    # 조기 / 후기 융합 결과 읽기
    early_results = read_report("early_gru_report.txt")
    late_results = read_report("late_gru_report.txt")

    if not early_results:
        print("오류: 조기 융합 결과를 찾을 수 없습니다.")
        print("먼저 조기 융합 학습 스크립트를 실행해 리포트를 생성해 주세요.")
        return

    if not late_results:
        print("오류: 후기 융합 결과를 찾을 수 없습니다.")
        print("먼저 후기 융합 학습 스크립트를 실행해 리포트를 생성해 주세요.")
        return

    # 비교할 주요 지표 정의 (키, 표시 이름)
    key_metrics = [
        ("test_move_acc", "이동 정확도"),
        ("test_act_acc", "행동 정확도"),
        ("cfusion_csr", "C_fusion CSR"),
    ]

    # 주요 지표 표 형태로 정리
    comparison_data = []
    for key, label in key_metrics:
        early_val = early_results.get(key, 0.0)
        late_val = late_results.get(key, 0.0)
        improvement = ((late_val - early_val) / early_val * 100) if early_val > 0 else 0.0
        comparison_data.append([label, early_val, late_val, improvement])

    comparison = pd.DataFrame(
        comparison_data,
        columns=["지표", "조기 융합", "후기 융합", "개선율(%)"],
    )

    print("\n[1] 주요 지표 비교:")
    print(comparison.to_string(index=False, float_format="%.4f"))

    # 환경별 CSR 비교
    print("\n[2] 환경별 CSR 비교:")
    envs = ["E1", "E2", "E3", "E4"]
    env_data = []

    for env in envs:
        early_val = early_results.get(env, 0.0)
        late_val = late_results.get(env, 0.0)

        if early_val > 0 or late_val > 0:
            improvement = ((late_val - early_val) / early_val * 100) if early_val > 0 else 0.0
            env_data.append([env, early_val, late_val, improvement])

    if env_data:
        env_df = pd.DataFrame(
            env_data,
            columns=["환경", "조기 융합", "후기 융합", "개선율(%)"],
        )
        print(env_df.to_string(index=False, float_format="%.4f"))
    else:
        env_df = None
        print("환경별 CSR 정보가 충분하지 않습니다.")

    # ----------------- 시각화 -----------------
    try:
        plt.style.use('seaborn-v0_8')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('조기 융합 vs 후기 융합 성능 비교', fontsize=16, fontweight='bold')

        # 1. 주요 지표 막대그래프
        ax1 = axes[0, 0]
        x = np.arange(len(key_metrics))
        width = 0.35

        ax1.bar(x - width / 2, comparison["조기 융합"], width,
                label="조기 융합", alpha=0.8)
        ax1.bar(x + width / 2, comparison["후기 융합"], width,
                label="후기 융합", alpha=0.8)
        ax1.set_xlabel('지표', fontsize=12)
        ax1.set_ylabel('정확도 / CSR', fontsize=12)
        ax1.set_title('주요 지표 비교', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(comparison["지표"], rotation=45, ha='right')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(0, 1.0)

        # 막대 위에 수치 표시
        for i, (early, late) in enumerate(zip(comparison["조기 융합"], comparison["후기 융합"])):
            ax1.text(i - width / 2, early + 0.01, f'{early:.3f}',
                     ha='center', va='bottom', fontsize=9)
            ax1.text(i + width / 2, late + 0.01, f'{late:.3f}',
                     ha='center', va='bottom', fontsize=9)

        # 2. 개선율(%) 막대그래프
        ax2 = axes[0, 1]
        colors = ['green' if v >= 0 else 'red' for v in comparison["개선율(%)"]]
        ax2.bar(x, comparison["개선율(%)"], color=colors, alpha=0.7)
        ax2.set_xlabel('지표', fontsize=12)
        ax2.set_ylabel('개선율 (%)', fontsize=12)
        ax2.set_title('후기 융합 대비 조기 융합 개선율', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(comparison["지표"], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        for i, val in enumerate(comparison["개선율(%)"]):
            ax2.text(
                i,
                val + (1 if val >= 0 else -3),
                f'{val:+.1f}%',
                ha='center',
                va='bottom' if val >= 0 else 'top',
                fontsize=10,
            )

        # 3. 레이더 차트 (조기 vs 후기)
        if len(key_metrics) >= 3:
            ax3 = axes[1, 0]
            angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False).tolist()
            angles += angles[:1]  # 폐곡선

            early_values = comparison["조기 융합"].tolist()
            late_values = comparison["후기 융합"].tolist()
            early_values += early_values[:1]
            late_values += late_values[:1]

            ax3.plot(angles, early_values, 'o-', linewidth=2, label="조기 융합")
            ax3.plot(angles, late_values, 'o-', linewidth=2, label="후기 융합")
            ax3.fill(angles, early_values, alpha=0.25)
            ax3.fill(angles, late_values, alpha=0.25)
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(["이동\n정확도", "행동\n정확도", "C_fusion\nCSR"], fontsize=10)
            ax3.set_title('성능 레이더 차트', fontsize=14, fontweight='bold')
            ax3.legend(loc='upper right', fontsize=11)
            ax3.grid(True, alpha=0.3, linestyle='--')
            ax3.set_ylim(0, 1.0)

        # 4. 환경별 CSR 막대그래프
        if env_data and len(env_data) > 0:
            ax4 = axes[1, 1]
            env_names = [row[0] for row in env_data]
            env_early = [row[1] for row in env_data]
            env_late = [row[2] for row in env_data]
            env_x = np.arange(len(env_names))

            ax4.bar(env_x - width / 2, env_early, width,
                    label="조기 융합", alpha=0.8)
            ax4.bar(env_x + width / 2, env_late, width,
                    label="후기 융합", alpha=0.8)
            ax4.set_xlabel('환경', fontsize=12)
            ax4.set_ylabel('CSR', fontsize=12)
            ax4.set_title('환경별 C_fusion CSR 비교', fontsize=14, fontweight='bold')
            ax4.set_xticks(env_x)
            ax4.set_xticklabels(env_names)
            ax4.legend(fontsize=11)
            ax4.grid(True, alpha=0.3, linestyle='--')
            ax4.set_ylim(0, 1.0)

            for i, (early, late) in enumerate(zip(env_early, env_late)):
                ax4.text(i - width / 2, early + 0.01, f'{early:.3f}',
                         ha='center', va='bottom', fontsize=9)
                ax4.text(i + width / 2, late + 0.01, f'{late:.3f}',
                         ha='center', va='bottom', fontsize=9)
        else:
            axes[1, 1].axis('off')
            axes[1, 1].text(
                0.5,
                0.5,
                '환경별 CSR 데이터 없음',
                ha='center',
                va='center',
                fontsize=12,
            )

        plt.tight_layout()
        plt.savefig(OUT_DIR / "fusion_comparison.png", dpi=150, bbox_inches='tight')
        print(f"\n비교 그림이 저장되었습니다: {OUT_DIR / 'fusion_comparison.png'}")

        # ----------------- 텍스트 리포트 저장 -----------------
        with open(OUT_DIR / "fusion_comparison.txt", "w", encoding="utf-8") as f:
            f.write("조기 융합 vs 후기 융합 상세 비교 리포트\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. 주요 지표 비교:\n")
            f.write("-" * 40 + "\n")
            f.write(comparison.to_string(index=False, float_format="%.4f"))
            f.write("\n\n")

            if env_df is not None:
                f.write("2. 환경별 CSR 비교:\n")
                f.write("-" * 40 + "\n")
                f.write(env_df.to_string(index=False, float_format="%.4f"))
                f.write("\n\n")

            # 전체 평균 성능 및 개선율
            early_avg = comparison["조기 융합"].mean()
            late_avg = comparison["후기 융합"].mean()
            overall_improvement = ((late_avg - early_avg) / early_avg * 100) if early_avg > 0 else 0.0

            f.write("3. 전체 성능 분석:\n")
            f.write("-" * 40 + "\n")
            f.write(f"조기 융합 평균 성능: {early_avg:.4f}\n")
            f.write(f"후기 융합 평균 성능: {late_avg:.4f}\n")
            f.write(f"전체 평균 기준 개선율: {overall_improvement:+.1f}%\n\n")

            if overall_improvement > 0:
                f.write("결론: 이번 실험 설정에서는 '후기 융합' 전략이 더 우수한 성능을 보였습니다.\n")
            else:
                f.write("결론: 이번 실험 설정에서는 '조기 융합' 전략이 더 우수한 성능을 보였습니다.\n")

        print(f"상세 비교 리포트가 저장되었습니다: {OUT_DIR / 'fusion_comparison.txt'}")

    except Exception as e:
        print(f"시각화 생성 중 오류 발생: {e}")
        print("matplotlib 또는 환경 설정을 다시 확인해 주세요.")

if __name__ == "__main__":
    compare_fusion_strategies()
