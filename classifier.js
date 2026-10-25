// Python(hand_features.py, scikit-learn MLP)과 똑같이 동작하는 JS 구현

// 21개 랜드마크 → 정규화된 63차원 벡터 (hand_features.landmarks_to_features와 동일)
export function landmarksToFeatures(landmarks, handedness) {
  const wrist = landmarks[0];
  const pts = landmarks.map((p) => [p.x - wrist.x, p.y - wrist.y, p.z - wrist.z]);
  const scale = Math.max(...pts.map(([x, y]) => Math.hypot(x, y)));
  const features = [];
  for (const [x, y, z] of pts) {
    const s = scale > 0 ? scale : 1;
    features.push(((handedness === "Left" ? -1 : 1) * x) / s, y / s, z / s);
  }
  return features;
}

const ACTIVATIONS = {
  relu: (v) => Math.max(0, v),
  tanh: Math.tanh,
  logistic: (v) => 1 / (1 + Math.exp(-v)),
  identity: (v) => v,
};

// MLP 순전파: 각 층마다 (입력 · 가중치 + 편향) → 활성화 함수
export function predictProba(model, features) {
  let x = features;
  const last = model.weights.length - 1;
  model.weights.forEach((W, layer) => {
    const b = model.biases[layer];
    const out = b.slice();
    for (let i = 0; i < x.length; i++) {
      const row = W[i];
      for (let j = 0; j < out.length; j++) out[j] += x[i] * row[j];
    }
    x = layer === last ? out : out.map(ACTIVATIONS[model.activation]);
  });

  if (model.outActivation === "logistic") {
    // 2클래스 모델은 출력이 1개 (= 두 번째 클래스일 확률)
    const p = ACTIVATIONS.logistic(x[0]);
    return [1 - p, p];
  }
  const max = Math.max(...x);
  const exp = x.map((v) => Math.exp(v - max));
  const sum = exp.reduce((a, c) => a + c, 0);
  return exp.map((v) => v / sum);
}
