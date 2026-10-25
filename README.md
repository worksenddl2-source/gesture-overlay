# ✋ Gesture Overlay

웹캠으로 손 제스처를 인식해 화면에 로고·이모지를 띄우는 웹 데모입니다.

**👉 데모: https://worksenddl2-source.github.io/gesture-overlay/**

| 제스처 | 표시 |
|---|---|
| `nike` | 스우시 로고 |
| `ok` | 👌 |
| `none` | (표시 없음) |

## 사용 방법

1. 위 데모 링크를 Chrome/Edge로 엽니다.
2. 카메라 권한을 **허용**합니다.
3. 카메라에 손을 보여주고 학습한 제스처를 취해 보세요.

> 영상은 브라우저 안에서만 처리되며 서버로 전송되지 않습니다.

## 동작 원리

```
웹캠 → 좌우 반전 → MediaPipe(브라우저)로 손 관절 21점 추출
     → 정규화(손목 기준·크기·좌우손) → 직접 학습한 MLP 분류기(classifier.js)
     → 최근 6프레임 다수결 → 제스처별 오버레이 표시
```

| 파일 | 역할 |
|---|---|
| `index.html` | 화면 구성 (영상, 확률 막대, 최소 확신도 슬라이더) |
| `app.js` | 웹캠·손 추적·스무딩·오버레이 |
| `classifier.js` | 정규화 + MLP 순전파 (Python scikit-learn 모델과 동일한 계산) |
| `model.json` | Python에서 학습한 분류기 가중치 |

## 로컬에서 실행

```bash
python -m http.server 8765 --bind 127.0.0.1
```

<http://127.0.0.1:8765> 접속 (`file://`로 직접 열면 카메라가 동작하지 않습니다).

## 제스처 바꾸기

모델은 Python(MediaPipe + scikit-learn)으로 수집·학습한 뒤 `export_model.py`로 `model.json`을 만들어 교체합니다.
새 제스처에 그림을 붙이려면 `app.js`의 `OVERLAYS`에 항목을 추가하세요.

```js
const OVERLAYS = {
  nike: SWOOSH,
  ok: `<span class="emoji">👌</span>`,
  call: `<span class="emoji">🤙</span>`,   // 예시
};
```

## 사용 기술

- [MediaPipe Tasks Vision (JS)](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/web_js) 1.1.0
- 순수 HTML/CSS/JavaScript (빌드 도구 없음)
