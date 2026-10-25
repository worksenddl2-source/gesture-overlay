import {
  FilesetResolver,
  GestureRecognizer,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";
import { landmarksToFeatures, predictProba } from "./classifier.js";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const LANDMARK_MODEL =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task";
const SMOOTHING_FRAMES = 6; // 최근 N프레임 중 다수결로 결과를 정해 깜빡임 방지

// 제스처 이름 → 화면에 띄울 내용. 새 제스처를 학습했다면 여기에 추가하세요.
const SWOOSH = `
<svg viewBox="0 0 200 72" xmlns="http://www.w3.org/2000/svg" aria-label="swoosh">
  <path fill="#fff" d="M196 4 C150 22 70 56 42 64 C22 70 6 66 4 54 C2 44 10 32 22 22
    C14 34 12 44 18 50 C24 56 38 54 56 48 C90 36 150 14 196 4 Z"/>
</svg>`;
const OVERLAYS = {
  nike: SWOOSH,
  ok: `<span class="emoji">👌</span>`,
};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const overlayEl = document.getElementById("overlay");
const currentEl = document.getElementById("current");
const barsEl = document.getElementById("bars");
const thresholdEl = document.getElementById("threshold");
const thValueEl = document.getElementById("thValue");

thresholdEl.addEventListener("input", () => {
  thValueEl.textContent = Number(thresholdEl.value).toFixed(2);
});

function setStatus(text) {
  statusEl.textContent = text;
  statusEl.classList.toggle("hidden", !text);
}

async function loadClassifier() {
  const res = await fetch("model.json", { cache: "no-store" });
  if (!res.ok) throw new Error("model.json이 없습니다.\npython web/export_model.py 를 먼저 실행하세요.");
  return res.json();
}

function buildBars(classes) {
  barsEl.innerHTML = classes
    .map(
      (c) => `<div class="bar" data-c="${c}">
        <div class="bar-head"><span>${c}</span><span class="v">0%</span></div>
        <div class="bar-track"><div class="bar-fill"></div></div>
      </div>`
    )
    .join("");
}

function updateBars(classes, proba) {
  const best = proba.indexOf(Math.max(...proba));
  classes.forEach((c, i) => {
    const el = barsEl.querySelector(`[data-c="${CSS.escape(c)}"]`);
    el.querySelector(".bar-fill").style.width = `${(proba[i] * 100).toFixed(0)}%`;
    el.querySelector(".v").textContent = `${(proba[i] * 100).toFixed(0)}%`;
    el.classList.toggle("top", i === best);
  });
}

function drawHand(landmarks) {
  const { width: w, height: h } = canvas;
  ctx.strokeStyle = "rgba(61, 220, 151, .9)";
  ctx.lineWidth = 3;
  for (const { start, end } of GestureRecognizer.HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[start].x * w, landmarks[start].y * h);
    ctx.lineTo(landmarks[end].x * w, landmarks[end].y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "#ff5a5a";
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

let shownGesture = null;
function showOverlay(name, landmarks) {
  if (!name || !OVERLAYS[name]) {
    overlayEl.classList.remove("show");
    shownGesture = null;
    return;
  }
  if (shownGesture !== name) {
    overlayEl.innerHTML = OVERLAYS[name];
    overlayEl.classList.remove("show");
    void overlayEl.offsetWidth; // 애니메이션 재시작
    shownGesture = name;
  }
  // 손의 가장 윗부분 위쪽에 표시 (캔버스 좌표 → 화면 % 좌표)
  const xs = landmarks.map((p) => p.x);
  const top = Math.min(...landmarks.map((p) => p.y));
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  overlayEl.style.left = `${cx * 100}%`;
  overlayEl.style.top = `${Math.max(top - 0.03, 0.35) * 100}%`;
  overlayEl.classList.add("show");
}

async function main() {
  const [classifier, vision] = await Promise.all([
    loadClassifier(),
    FilesetResolver.forVisionTasks(WASM_PATH),
  ]);
  buildBars(classifier.classes);

  const recognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: { modelAssetPath: LANDMARK_MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 1,
  });

  setStatus("카메라 권한을 허용해 주세요");
  const video = document.createElement("video");
  video.playsInline = true;
  video.muted = true;
  video.srcObject = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  setStatus("");

  const history = [];
  let lastTime = -1;

  function frame() {
    if (video.currentTime !== lastTime) {
      lastTime = video.currentTime;

      // Python 버전(cv2.flip)과 똑같이 좌우 반전된 이미지를 분석해야 학습 데이터와 일치한다
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();

      const result = recognizer.recognizeForVideo(canvas, performance.now());
      const hand = result.landmarks[0];

      let name = null;
      if (hand) {
        drawHand(hand);
        const handedness = result.handedness[0][0].categoryName;
        const proba = predictProba(classifier, landmarksToFeatures(hand, handedness));
        updateBars(classifier.classes, proba);
        const best = proba.indexOf(Math.max(...proba));
        if (proba[best] >= Number(thresholdEl.value)) name = classifier.classes[best];
      } else {
        updateBars(classifier.classes, classifier.classes.map(() => 0));
      }

      history.push(name);
      if (history.length > SMOOTHING_FRAMES) history.shift();
      const votes = {};
      for (const h of history) votes[h] = (votes[h] || 0) + 1;
      const [stable, count] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
      const gesture = stable !== "null" && count > SMOOTHING_FRAMES / 2 ? stable : null;

      currentEl.textContent = gesture ?? (hand ? "?" : "-");
      showOverlay(hand ? gesture : null, hand);
    }
    requestAnimationFrame(frame);
  }
  frame();
}

main().catch((err) => {
  console.error(err);
  setStatus(`오류: ${err.message}`);
});
