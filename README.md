# DeepSeek V3 아키텍처 튜토리얼

이 저장소는 DeepSeek V3 모델의 아키텍처를 한국어로 설명하는 **인터랙티브 Jupyter 노트북 튜토리얼**입니다.

## 참고 자료
- 논문: [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)
- 코드: [HuggingFace Transformers - DeepSeek V3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py)

## 폴더 구조

```
deepseek-explained/
├── README.md
└── model_architecture/
    ├── 01_overview_tutorial.ipynb          # DeepSeek V3 전체 아키텍처 개요
    ├── 02_rmsnorm_rope_tutorial.ipynb      # RMSNorm과 RoPE 설명
    ├── 03_mla_attention_tutorial.ipynb     # Multi-head Latent Attention (MLA)
    ├── 04_moe_routing_tutorial.ipynb       # DeepSeekMoE와 라우팅 메커니즘
    └── 05_full_model_tutorial.ipynb        # 전체 모델 구조와 추론 흐름
```

## 튜토리얼 내용

| 노트북 | 주제 | 핵심 내용 |
|--------|------|----------|
| **01_overview** | 아키텍처 개요 | DeepSeek V3의 전체 구조, 주요 혁신 기술 소개 |
| **02_rmsnorm_rope** | 정규화 & 위치 인코딩 | RMSNorm 구현, RoPE (Rotary Position Embedding) 원리 |
| **03_mla_attention** | MLA 어텐션 | KV 압축, Decoupled RoPE, k_nope/k_rope concat |
| **04_moe_routing** | MoE 라우팅 | 전문가 라우팅, e_score_correction_bias, Load Balancing |
| **05_full_model** | 전체 모델 | 모든 컴포넌트 통합, 추론 흐름 |

## 실행 방법

```bash
# Jupyter 노트북 실행
cd model_architecture
jupyter notebook

# 또는 VS Code / Cursor에서 직접 열기
```

각 노트북은 다음을 포함합니다:
- 📚 개념 설명 (마크다운)
- 💻 실행 가능한 코드 예제
- 📝 이해도 테스트 퀴즈
- 🔍 HuggingFace 실제 구현과의 비교

## 주요 특징

DeepSeek V3는 다음과 같은 혁신적인 기술을 포함합니다:

### 1. Multi-head Latent Attention (MLA)
- KV 캐시를 저차원 잠재 벡터 `c_KV`로 압축
- **98%+ 메모리 절감** (131K 시퀀스 기준)
- Decoupled RoPE로 위치 정보 보존

### 2. DeepSeekMoE
- **Auxiliary loss 없는** 로드 밸런싱 (`e_score_correction_bias`)
- Sigmoid 기반 라우팅 (softmax 대신)
- 256개 전문가 중 8개만 활성화 → 효율적인 계산

### 3. Multi-Token Prediction (MTP)
- 한 번에 여러 토큰을 예측하여 추론 속도 향상

## 요구 사항

```bash
pip install torch jupyter
```
