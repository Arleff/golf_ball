"""
高尔夫球挥杆视频检测 Web 应用
Flask 后端：接收上传视频，运行 Hough + 卡尔曼检测，返回带标注视频与统计信息
"""
import os
import uuid
import threading
import time
import json
from pathlib import Path

from flask import (
    Flask, request, jsonify, send_file,
    render_template_string, abort
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

from golf_detector import process_video

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# 任务状态字典 {task_id: {...}}
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _run_detection(task_id: str, input_path: str, output_path: str):
    """后台线程执行检测"""
    with _tasks_lock:
        _tasks[task_id]["status"] = "processing"
        _tasks[task_id]["progress"] = 0

    def progress_cb(done, total):
        pct = int(done / max(total, 1) * 100)
        with _tasks_lock:
            _tasks[task_id]["progress"] = pct

    try:
        stats = process_video(input_path, output_path, progress_callback=progress_cb)
        with _tasks_lock:
            _tasks[task_id].update({
                "status": "done",
                "progress": 100,
                "stats": stats,
                "output_file": os.path.basename(output_path),
            })
    except Exception as exc:
        with _tasks_lock:
            _tasks[task_id].update({
                "status": "error",
                "error": str(exc),
            })
    finally:
        # 清理上传原文件
        try:
            os.remove(input_path)
        except Exception:
            pass


# ─── HTML 模板 ──────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>高尔夫球挥杆检测 · Golf Ball Tracker</title>
  <style>
    :root{
      --bg:#0f1117; --surface:#1c1e26; --card:#252836;
      --accent:#00e676; --accent2:#ffd740; --text:#e8eaf6;
      --muted:#9e9eb8; --radius:12px;
    }
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
    header{
      background:linear-gradient(135deg,#1a1f35 0%,#0d1b2a 100%);
      border-bottom:1px solid #2a2d3e;
      padding:20px 32px;display:flex;align-items:center;gap:14px
    }
    header .logo{font-size:2rem}
    header h1{font-size:1.5rem;font-weight:700;letter-spacing:.5px}
    header p{font-size:.85rem;color:var(--muted);margin-top:2px}
    .container{max-width:960px;margin:40px auto;padding:0 20px}
    .card{background:var(--card);border-radius:var(--radius);padding:28px;margin-bottom:24px;border:1px solid #2a2d3e}
    .card h2{font-size:1.1rem;margin-bottom:16px;color:var(--accent);display:flex;align-items:center;gap:8px}

    /* 上传区域 */
    .drop-zone{
      border:2px dashed #3a3d55;border-radius:var(--radius);
      padding:48px 24px;text-align:center;cursor:pointer;
      transition:all .2s;position:relative
    }
    .drop-zone:hover,.drop-zone.drag-over{border-color:var(--accent);background:rgba(0,230,118,.05)}
    .drop-zone input{display:none}
    .drop-icon{font-size:3rem;margin-bottom:12px}
    .drop-text{font-size:1rem;color:var(--muted)}
    .drop-text strong{color:var(--text)}
    .file-info{
      margin-top:16px;padding:12px 16px;background:#1a1d2e;
      border-radius:8px;display:none;align-items:center;gap:10px;
      font-size:.9rem;
    }
    .file-info.show{display:flex}
    .file-icon{font-size:1.4rem}

    /* 按钮 */
    .btn{
      display:inline-flex;align-items:center;gap:8px;
      padding:12px 28px;border-radius:8px;font-size:1rem;font-weight:600;
      border:none;cursor:pointer;transition:all .2s;text-decoration:none
    }
    .btn-primary{background:var(--accent);color:#0f1117}
    .btn-primary:hover{background:#00c853;transform:translateY(-1px)}
    .btn-primary:disabled{background:#3a3d55;color:var(--muted);cursor:not-allowed;transform:none}
    .btn-secondary{background:transparent;color:var(--accent);border:1px solid var(--accent)}
    .btn-secondary:hover{background:rgba(0,230,118,.08)}

    /* 进度 */
    .progress-wrap{margin-top:20px;display:none}
    .progress-wrap.show{display:block}
    .progress-bar-bg{background:#1a1d2e;border-radius:99px;height:10px;overflow:hidden}
    .progress-bar-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#00c853,#69f0ae);transition:width .4s}
    .progress-label{display:flex;justify-content:space-between;margin-top:6px;font-size:.85rem;color:var(--muted)}

    /* 状态标签 */
    .badge{display:inline-block;padding:4px 12px;border-radius:99px;font-size:.78rem;font-weight:600}
    .badge-waiting{background:#3a3d55;color:var(--muted)}
    .badge-processing{background:#1a2744;color:#82b1ff}
    .badge-done{background:#1b3a2a;color:var(--accent)}
    .badge-error{background:#3a1a1a;color:#ff5252}

    /* 统计卡片 */
    .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:14px;margin-top:12px}
    .stat-item{background:#1a1d2e;border-radius:8px;padding:14px 16px;text-align:center}
    .stat-value{font-size:1.5rem;font-weight:700;color:var(--accent2)}
    .stat-label{font-size:.75rem;color:var(--muted);margin-top:4px}

    /* 视频 */
    video{width:100%;border-radius:8px;margin-top:16px;background:#000}

    /* 算法说明 */
    .algo-steps{list-style:none;counter-reset:step}
    .algo-steps li{counter-increment:step;padding:10px 12px 10px 44px;position:relative;border-bottom:1px solid #2a2d3e}
    .algo-steps li:last-child{border-bottom:none}
    .algo-steps li::before{
      content:counter(step);position:absolute;left:12px;top:50%;transform:translateY(-50%);
      width:24px;height:24px;border-radius:50%;background:var(--accent);color:#0f1117;
      font-size:.8rem;font-weight:700;display:flex;align-items:center;justify-content:center;
      display:grid;place-items:center
    }
    .algo-steps li .step-title{font-weight:600;margin-bottom:2px}
    .algo-steps li .step-desc{font-size:.82rem;color:var(--muted)}

    @media(max-width:600px){
      header{padding:14px 16px}
      .container{padding:0 12px}
      .card{padding:18px}
    }
  </style>
</head>
<body>
<header>
  <span class="logo">⛳</span>
  <div>
    <h1>高尔夫球挥杆检测 &amp; 追踪</h1>
    <p>Golf Ball Detection &amp; Tracking · Hough Circles + Kalman Filter</p>
  </div>
</header>

<div class="container">

  <!-- 上传区 -->
  <div class="card">
    <h2>📂 上传挥杆视频</h2>
    <div class="drop-zone" id="dropZone">
      <input type="file" id="fileInput" accept="video/*"/>
      <div class="drop-icon">🎬</div>
      <div class="drop-text"><strong>点击选择</strong>或拖拽视频文件到此处</div>
      <div class="drop-text" style="font-size:.8rem;margin-top:6px">支持 MP4 · AVI · MOV · MKV · WebM（最大 500 MB）</div>
    </div>
    <div class="file-info" id="fileInfo">
      <span class="file-icon">🎞️</span>
      <div>
        <div id="fileName" style="font-weight:600"></div>
        <div id="fileSize" style="font-size:.8rem;color:var(--muted)"></div>
      </div>
    </div>
    <div style="margin-top:20px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">
      <button class="btn btn-primary" id="uploadBtn" disabled onclick="uploadVideo()">
        ▶ 开始检测
      </button>
      <span id="statusBadge" class="badge badge-waiting">等待上传</span>
    </div>
    <div class="progress-wrap" id="progressWrap">
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="progressBar" style="width:0%"></div></div>
      <div class="progress-label"><span id="progressText">处理中…</span><span id="progressPct">0%</span></div>
    </div>
  </div>

  <!-- 结果区 -->
  <div class="card" id="resultCard" style="display:none">
    <h2>✅ 检测结果</h2>
    <div class="stats-grid" id="statsGrid"></div>
    <video id="resultVideo" controls playsinline></video>
    <div style="margin-top:16px;display:flex;gap:12px;flex-wrap:wrap">
      <a class="btn btn-primary" id="downloadBtn" href="#" download>⬇ 下载结果视频</a>
      <button class="btn btn-secondary" onclick="reset()">🔄 重新上传</button>
    </div>
  </div>

  <!-- 算法说明 -->
  <div class="card">
    <h2>⚙️ 算法流程</h2>
    <ol class="algo-steps">
      <li>
        <div class="step-title">读取视频帧</div>
        <div class="step-desc">逐帧解码，支持所有主流格式（基于 OpenCV VideoCapture）</div>
      </li>
      <li>
        <div class="step-title">ROI 裁剪（±150 px）</div>
        <div class="step-desc">围绕上一帧预测中心裁剪感兴趣区域，与论文 test_net.py 策略一致，大幅减少背景干扰</div>
      </li>
      <li>
        <div class="step-title">Hough 圆形变换检测</div>
        <div class="step-desc">在灰度 ROI 中运行 HoughCircles，检测球形目标（高尔夫球为白色圆球）</div>
      </li>
      <li>
        <div class="step-title">颜色+形态学备用检测</div>
        <div class="step-desc">当 Hough 检测失败时，以 HSV 白色阈值 + 圆度筛选作为补充策略</div>
      </li>
      <li>
        <div class="step-title">卡尔曼滤波平滑</div>
        <div class="step-desc">常速度模型（状态 [x, y, vx, vy]），同论文结构，预测遮挡帧位置并平滑轨迹</div>
      </li>
      <li>
        <div class="step-title">轨迹可视化输出</div>
        <div class="step-desc">在原视频帧叠加 ROI 框、检测圆、彩色渐变轨迹线并输出 MP4</div>
      </li>
    </ol>
  </div>

</div>

<script>
const $ = id => document.getElementById(id);
let currentFile = null;
let pollTimer = null;

// 拖拽
const dz = $('dropZone');
dz.addEventListener('dragover', e=>{e.preventDefault();dz.classList.add('drag-over')});
dz.addEventListener('dragleave', ()=>dz.classList.remove('drag-over'));
dz.addEventListener('drop', e=>{
  e.preventDefault();dz.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if(f) setFile(f);
});
dz.addEventListener('click', ()=>$('fileInput').click());
$('fileInput').addEventListener('change', e=>{
  if(e.target.files[0]) setFile(e.target.files[0]);
});

function setFile(f){
  currentFile = f;
  $('fileName').textContent = f.name;
  $('fileSize').textContent = (f.size/1024/1024).toFixed(1)+' MB';
  $('fileInfo').classList.add('show');
  $('uploadBtn').disabled = false;
  setBadge('waiting','等待检测');
}

function setBadge(state, text){
  const b = $('statusBadge');
  b.className = 'badge badge-'+state;
  b.textContent = text;
}

async function uploadVideo(){
  if(!currentFile) return;
  $('uploadBtn').disabled = true;
  $('progressWrap').classList.add('show');
  $('resultCard').style.display='none';
  setBadge('processing','上传中…');

  const fd = new FormData();
  fd.append('video', currentFile);

  let resp;
  try{
    resp = await fetch('/api/upload', {method:'POST', body:fd});
  }catch(e){
    setBadge('error','网络错误');
    $('uploadBtn').disabled = false;
    return;
  }
  if(!resp.ok){
    const msg = await resp.text();
    setBadge('error','上传失败: '+msg);
    $('uploadBtn').disabled = false;
    return;
  }
  const {task_id} = await resp.json();
  setBadge('processing','检测中…');
  pollStatus(task_id);
}

function pollStatus(task_id){
  if(pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async()=>{
    const r = await fetch('/api/status/'+task_id);
    if(!r.ok) return;
    const d = await r.json();
    const pct = d.progress||0;
    $('progressBar').style.width = pct+'%';
    $('progressPct').textContent = pct+'%';
    $('progressText').textContent = d.status==='processing'?'检测中…':(d.status==='done'?'完成':'等待中');

    if(d.status==='done'){
      clearInterval(pollTimer);
      setBadge('done','✓ 检测完成');
      showResult(d);
    } else if(d.status==='error'){
      clearInterval(pollTimer);
      setBadge('error','错误: '+(d.error||'未知'));
      $('uploadBtn').disabled = false;
    }
  }, 800);
}

function showResult(d){
  const stats = d.stats||{};
  const items = [
    {v: stats.total_frames||'-',  l:'总帧数'},
    {v: stats.detected_frames||'-', l:'检测到帧数'},
    {v: (stats.detection_rate||0)+'%', l:'检测率'},
    {v: stats.fps||'-',          l:'帧率 FPS'},
    {v: stats.resolution||'-',   l:'分辨率'},
  ];
  $('statsGrid').innerHTML = items.map(it=>
    `<div class="stat-item"><div class="stat-value">${it.v}</div><div class="stat-label">${it.l}</div></div>`
  ).join('');

  const url = '/api/download/'+d.output_file;
  $('resultVideo').src = url;
  $('downloadBtn').href = url;
  $('downloadBtn').download = d.output_file;
  $('resultCard').style.display='block';
  $('resultCard').scrollIntoView({behavior:'smooth'});
}

function reset(){
  currentFile = null;
  $('fileInput').value='';
  $('fileInfo').classList.remove('show');
  $('uploadBtn').disabled = true;
  $('progressWrap').classList.remove('show');
  $('progressBar').style.width='0%';
  $('progressPct').textContent='0%';
  $('resultCard').style.display='none';
  setBadge('waiting','等待上传');
  if(pollTimer) clearInterval(pollTimer);
}
</script>
</body>
</html>
"""


# ─── API 路由 ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return "未找到视频文件", 400
    file = request.files["video"]
    if file.filename == "":
        return "文件名为空", 400
    if not allowed_file(file.filename):
        return f"不支持的文件格式，允许: {', '.join(ALLOWED_EXTENSIONS)}", 400

    task_id = str(uuid.uuid4())
    ext = file.filename.rsplit(".", 1)[1].lower()
    input_filename = f"{task_id}_input.{ext}"
    output_filename = f"{task_id}_output.mp4"

    input_path = str(UPLOAD_FOLDER / input_filename)
    output_path = str(OUTPUT_FOLDER / output_filename)

    file.save(input_path)

    with _tasks_lock:
        _tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "created_at": time.time(),
        }

    t = threading.Thread(
        target=_run_detection,
        args=(task_id, input_path, output_path),
        daemon=True,
    )
    t.start()

    return jsonify({"task_id": task_id})


@app.route("/api/status/<task_id>")
def status(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        abort(404)
    return jsonify(task)


@app.route("/api/download/<filename>")
def download(filename: str):
    safe_name = secure_filename(filename)
    path = OUTPUT_FOLDER / safe_name
    if not path.exists():
        abort(404)
    return send_file(str(path), mimetype="video/mp4", as_attachment=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
