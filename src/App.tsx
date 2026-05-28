/**
 * 役割: OCR/VLM/YOLO の解析結果を可視化・デバッグするためのダッシュボードUI
 * AI向け役割: 座標データの正規化、レイヤー別の表示切り替え、トークン使用履歴の管理、および各エンジンの解析ログをタブ表示する。
 */
import React, { useState, useRef, useEffect, type MouseEvent } from 'react';
import { fetchWithRetry } from './utils/fetchClient';

interface TokenUsage {
  promptTokenCount: number;
  candidatesTokenCount: number;
  totalTokenCount: number;
}

interface UsageHistory {
  timestamp: string;
  model: string;
  usage: TokenUsage;
}

interface AIModel {
  id: string;
  displayName: string;
}

interface BoundingBoxElement {
  text?: string;
  label?: string;
  bounding_box: [number, number, number, number];
}

type FetchStatus = 'idle' | 'loading' | 'success' | 'error';
type LogTab = 'vlm' | 'ocr' | 'yolo';

export default function App() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('画面内の主要なメニューやボタンをすべてリストアップしてください。');  
  const [availableModels, setAvailableModels] = useState<AIModel[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelStatus, setModelStatus] = useState<FetchStatus>('idle');
  
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [hoverCoords, setHoverCoords] = useState({ x: 0, y: 0 });
  
  const [aiBoxCoords, setAiBoxCoords] = useState<number[][]>([]);
  const [ocrBoxCoords, setOcrBoxCoords] = useState<BoundingBoxElement[]>([]); 
  const [yoloBoxCoords, setYoloBoxCoords] = useState<BoundingBoxElement[]>([]); 
  const [aiResponseText, setAiResponseText] = useState('');

  const [showOcr, setShowOcr] = useState(true);
  const [showVlm, setShowVlm] = useState(true);
  const [showYolo, setShowYolo] = useState(true);
  
  const [currentUsage, setCurrentUsage] = useState<TokenUsage | null>(null);
  const [usageLog, setUsageLog] = useState<UsageHistory[]>([]);
  
  const [activeLogTab, setActiveLogTab] = useState<LogTab>('vlm');
  
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  useEffect(() => {
    const fetchModels = async () => {
      setModelStatus('loading');
      setErrorMessage(null);
      try {
        const res = await fetchWithRetry(`${API_BASE_URL}/api/models`, {}, 2, 5000);
        const data = await res.json();
        if (data.models && data.models.length > 0) {
          setAvailableModels(data.models);
          setSelectedModel(data.models[0].id);
          setModelStatus('success');
        } else {
          throw new Error('利用可能なモデルが見つかりませんでした');
        }
      } catch (error: unknown) {
        console.error('モデル一覧の取得に失敗しました:', error);
        setModelStatus('error');
        setErrorMessage(error instanceof Error ? error.message : 'モデルの取得中に不明なエラーが発生しました');
      }
    };
    fetchModels();
  }, [API_BASE_URL]);

  const processImageFile = (file: File) => {
    if (!file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      setImageSrc(event.target?.result as string);
    };
    reader.readAsDataURL(file);
    setAiBoxCoords([]);
    setOcrBoxCoords([]);
    setYoloBoxCoords([]);
    setAiResponseText('');
    setCurrentUsage(null);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processImageFile(file);
  };

  const handleMouseMove = (e: MouseEvent<HTMLImageElement>) => {
    if (!imageRef.current) return;
    const rect = imageRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setHoverCoords({ x: x / rect.width, y: y / rect.height });
  };

  const handleRunAi = async () => {
    if (!imageSrc || isAnalyzing) return;
    
    setIsAnalyzing(true);
    setAiResponseText('AIに問い合わせ中...');
    setAiBoxCoords([]);
    setOcrBoxCoords([]);
    setYoloBoxCoords([]);
    setErrorMessage(null);

    try {
      // なぜ: Blobストリームの競合(ロック)を防ぐため、元の画像をArrayBuffer化し、通信ごとに独立したBlobを生成する
      const resBlob = await fetch(imageSrc);
      const arrayBuffer = await resBlob.arrayBuffer();
      const mimeType = resBlob.headers.get('content-type') || 'image/png';

      const vlmFormData = new FormData();
      vlmFormData.append('image', new Blob([arrayBuffer], { type: mimeType }), 'screenshot.png');
      vlmFormData.append('prompt', prompt);
      vlmFormData.append('model', selectedModel);

      const ocrFormData = new FormData();
      ocrFormData.append('image', new Blob([arrayBuffer], { type: mimeType }), 'screenshot.png');

      const yoloFormData = new FormData();
      yoloFormData.append('image', new Blob([arrayBuffer], { type: mimeType }), 'screenshot.png');

      // なぜ: Promise.allSettledによる同期ブロックを避け、完了したものから即座に画面を描画(State更新)するため個別のPromiseチェーンを構築する
      // なぜ: FormDataを用いたPOST通信のリトライはブラウザ仕様でTypeErrorを引き起こすため、ここではネイティブのfetchを使用する
      const vlmTask = fetch(`${API_BASE_URL}/api/analyze`, { method: 'POST', body: vlmFormData })
        .then(async (res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          if (data.text) {
            setAiResponseText(data.text);
            const regex = /\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]/g;
            const matches = [...data.text.matchAll(regex)];
            setAiBoxCoords(matches.map((m: string[]) => [parseFloat(m[1]), parseFloat(m[2]), parseFloat(m[3]), parseFloat(m[4])]));
          }
          if (data.usage) {
            setCurrentUsage(data.usage);
            setUsageLog(prev => [{ timestamp: new Date().toLocaleTimeString(), model: selectedModel, usage: data.usage }, ...prev].slice(0, 10));
          }
          return null;
        })
        .catch(err => {
          setAiResponseText('VLMの解析に失敗しました。');
          return `VLM通信エラー: ${err.message}`;
        });

      const ocrTask = fetch(`${API_BASE_URL}/api/scan`, { method: 'POST', body: ocrFormData })
        .then(async (res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          if (data.elements) setOcrBoxCoords(data.elements);
          return null;
        })
        .catch(err => `OCR通信エラー: ${err.message}`);

      const yoloTask = fetch(`${API_BASE_URL}/api/yolo`, { method: 'POST', body: yoloFormData })
        .then(async (res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          if (data.elements) setYoloBoxCoords(data.elements);
          return null;
        })
        .catch(err => `YOLO通信エラー: ${err.message}`);

      // すべての並列タスクの完了を待機し、返却されたエラー文字列（null以外）があれば表示
      const results = await Promise.all([vlmTask, ocrTask, yoloTask]);
      const errors = results.filter(Boolean);
      if (errors.length > 0) {
        setErrorMessage(errors.join(' / '));
      }

    } catch (error: unknown) {
      setAiResponseText('致命的なエラーが発生しました。');
      setErrorMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 font-sans text-gray-900 overflow-hidden">
      {/* 左サイドバー */}
      <div className="w-1/3 p-6 bg-white shadow-xl flex flex-col gap-6 overflow-y-auto border-r border-gray-200">
        <header>
          <h1 className="text-2xl font-black tracking-tight text-blue-600">VLM RPA ANALYZER</h1>
          <p className="text-xs text-gray-400 font-bold uppercase tracking-tighter">Graduation Research Dashboard</p>
        </header>

        {/* レイヤー切り替え */}
        <section className="bg-gray-50 p-4 rounded-2xl border border-gray-100 flex flex-col gap-3">
          <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest block">Display Layers</label>
          <div className="flex flex-wrap gap-2">
            <button 
              onClick={() => setShowOcr(!showOcr)}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${showOcr ? 'bg-blue-500 text-white shadow-md' : 'bg-white text-gray-400 border border-gray-200'}`}
            >
              OCR (Blue)
            </button>
            <button 
              onClick={() => setShowVlm(!showVlm)}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${showVlm ? 'bg-red-500 text-white shadow-md' : 'bg-white text-gray-400 border border-gray-200'}`}
            >
              VLM (Red)
            </button>
            <button 
              onClick={() => setShowYolo(!showYolo)}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${showYolo ? 'bg-green-500 text-white shadow-md' : 'bg-white text-gray-400 border border-gray-200'}`}
            >
              YOLO (Green)
            </button>
          </div>
        </section>

        {/* モデル選択 */}
        <section>
          <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest block mb-2">Active Model</label>
          <select 
            className="w-full border-2 border-gray-50 bg-gray-50 p-3 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none text-sm font-medium"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={modelStatus !== 'success'}
          >
            {availableModels.map(m => <option key={m.id} value={m.id}>{m.displayName}</option>)}
          </select>
          {errorMessage && (
            <p className="text-[10px] text-red-500 mt-2 font-medium bg-red-50 p-2 rounded-lg border border-red-100">
              ⚠️ {errorMessage}
            </p>
          )}
        </section>

        {/* 画像アップロード */}
        <section className="flex flex-col gap-2">
          <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Target Image</label>
          <div 
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => { e.preventDefault(); setIsDragging(false); const f = e.dataTransfer.files?.[0]; if (f) processImageFile(f); }}
            className={`border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all
              ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-100 hover:border-blue-400 hover:bg-gray-50'}`}
          >
            <p className="text-sm font-bold text-gray-500">Drop or Click to Upload</p>
            <input type="file" accept="image/*" ref={fileInputRef} onChange={handleImageUpload} className="hidden" />
          </div>
        </section>

        <section>
          <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest block mb-2">System Prompt</label>
          <textarea 
            className="w-full border-2 border-gray-50 bg-gray-50 p-4 rounded-2xl h-32 text-sm focus:ring-2 focus:ring-blue-500 outline-none resize-none font-medium leading-relaxed"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </section>

        <button 
          onClick={handleRunAi}
          disabled={!imageSrc || modelStatus !== 'success' || isAnalyzing}
          className={`w-full py-4 rounded-2xl font-bold shadow-lg transition-all text-sm flex items-center justify-center gap-2
            ${isAnalyzing
              ? 'bg-blue-400 cursor-not-allowed text-white shadow-none'
              : 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-100 active:scale-95 disabled:bg-gray-200 disabled:text-gray-400'
            }`}
        >
          {isAnalyzing && (
            <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          )}
          {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
        </button>

        {/* 履歴と統計 */}
        <section className="flex flex-col gap-4 mt-auto">
          {currentUsage && (
            <div className="bg-blue-50/50 p-4 rounded-2xl border border-blue-100">
              <h3 className="text-[10px] font-bold text-blue-400 uppercase tracking-widest mb-3">Usage Stats</h3>
              <div className="grid grid-cols-3 gap-2">
                <div className="bg-white p-2 rounded-lg text-center shadow-sm">
                  <p className="text-[9px] text-gray-400 font-bold">In</p>
                  <p className="text-xs font-black text-gray-700">{currentUsage.promptTokenCount}</p>
                </div>
                <div className="bg-white p-2 rounded-lg text-center shadow-sm">
                  <p className="text-[9px] text-gray-400 font-bold">Out</p>
                  <p className="text-xs font-black text-gray-700">{currentUsage.candidatesTokenCount}</p>
                </div>
                <div className="bg-white p-2 rounded-lg text-center shadow-sm border border-blue-200">
                  <p className="text-[9px] text-blue-400 font-bold">Total</p>
                  <p className="text-xs font-black text-blue-600">{currentUsage.totalTokenCount}</p>
                </div>
              </div>
            </div>
          )}

          {usageLog.length > 0 && (
            <div className="max-h-32 overflow-y-auto pr-2">
              <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest block mb-2">History</label>
              <div className="flex flex-col gap-1.5">
                {usageLog.map((log, i) => (
                  <div key={i} className="flex justify-between items-center text-[10px] bg-gray-50 p-2 rounded-lg border border-gray-100">
                    <span className="text-gray-400">{log.timestamp}</span>
                    <span className="font-bold text-blue-600">{log.usage.totalTokenCount} tokens</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>

      {/* メインプレビュー */}
      <main className="flex-1 p-8 flex flex-col gap-6 bg-gray-50/50 overflow-hidden">
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-black text-gray-800 tracking-tight">Preview Canvas</h2>
          <div className="flex gap-4">
            <div className="bg-gray-900 text-green-400 px-4 py-2 rounded-xl font-mono text-xs shadow-inner">X: {hoverCoords.x.toFixed(3)}</div>
            <div className="bg-gray-900 text-green-400 px-4 py-2 rounded-xl font-mono text-xs shadow-inner">Y: {hoverCoords.y.toFixed(3)}</div>
          </div>
        </div>

        <div className="flex-1 bg-white rounded-3xl border border-gray-200 flex items-center justify-center overflow-auto relative shadow-sm p-8">
          {!imageSrc ? (
            <div className="text-gray-300 flex flex-col items-center">
              <p className="font-bold text-xs uppercase tracking-widest">No image provided</p>
            </div>
          ) : (
            <div className="relative inline-block m-4">
              <img 
                ref={imageRef} 
                src={imageSrc} 
                alt="Target UI" 
                className="max-h-[60vh] max-w-full block cursor-crosshair shadow-2xl ring-4 ring-gray-100"
                style={{ width: 'auto', height: 'auto' }}
                onMouseMove={handleMouseMove}
              />

              {/* 解析中の半透明オーバーレイとスピナー */}
              {isAnalyzing && (
                <div className="absolute inset-0 bg-white/60 backdrop-blur-[2px] flex items-center justify-center z-50 rounded-lg">
                  <div className="flex flex-col items-center gap-3">
                    <svg className="animate-spin h-12 w-12 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span className="text-blue-600 font-bold text-sm tracking-widest uppercase drop-shadow-sm">Processing...</span>
                  </div>
                </div>
              )}
              
              {/* OCR レイヤー (青) */}
              {showOcr && ocrBoxCoords.map((el, i) => (
                <div 
                  key={`ocr-${i}`}
                  className="absolute border border-blue-400 bg-blue-400/10 pointer-events-none transition-all"
                  style={{
                    top: `${el.bounding_box[0] / 10}%`, 
                    left: `${el.bounding_box[1] / 10}%`,
                    height: `${(el.bounding_box[2] - el.bounding_box[0]) / 10}%`, 
                    width: `${(el.bounding_box[3] - el.bounding_box[1]) / 10}%`
                  }}
                >
                  <span className="absolute -top-4 left-0 text-[8px] bg-blue-500 text-white px-1 rounded shadow-sm z-30">{el.text}</span>
                </div>
              ))}

              {/* YOLO レイヤー (緑) */}
              {showYolo && yoloBoxCoords.map((el, i) => (
                <div 
                  key={`yolo-${i}`}
                  className="absolute border-2 border-green-500 bg-green-500/10 pointer-events-none transition-all z-20"
                  style={{
                    top: `${el.bounding_box[0] / 10}%`, 
                    left: `${el.bounding_box[1] / 10}%`,
                    height: `${(el.bounding_box[2] - el.bounding_box[0]) / 10}%`, 
                    width: `${(el.bounding_box[3] - el.bounding_box[1]) / 10}%`
                  }}
                >
                  <span className="absolute -bottom-4 left-0 text-[8px] bg-green-600 text-white px-1 rounded shadow-sm">{el.label}</span>
                </div>
              ))}

              {/* VLM レイヤー (赤) */}
              {showVlm && aiBoxCoords.map((coords, i) => {
                const scale = coords.some(c => c > 1) ? 1000 : 1;
                return (
                  <div 
                    key={`vlm-${i}`}
                    className="absolute border-2 border-red-500 bg-red-500/20 pointer-events-none transition-all z-40 shadow-[0_0_10px_rgba(239,68,68,0.5)]"
                    style={{
                      top: `${(coords[0] / scale) * 100}%`, 
                      left: `${(coords[1] / scale) * 100}%`,
                      height: `${((coords[2] - coords[0]) / scale) * 100}%`, 
                      width: `${((coords[3] - coords[1]) / scale) * 100}%`
                    }}
                  >
                    <span className="absolute -top-6 left-0 bg-red-500 text-[9px] text-white px-2 py-0.5 rounded font-black shadow-lg">#{i + 1} Selected</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* ログビューアー (タブ切り替え) */}
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-200 h-56 flex flex-col">
          <div className="flex border-b border-gray-100 mb-4 pb-2 gap-6">
            <button
              onClick={() => setActiveLogTab('vlm')}
              className={`text-[10px] font-bold uppercase tracking-widest pb-2 border-b-2 transition-all ${activeLogTab === 'vlm' ? 'border-red-500 text-red-600' : 'border-transparent text-gray-400 hover:text-gray-600'}`}
            >
              VLM Result
            </button>
            <button
              onClick={() => setActiveLogTab('ocr')}
              className={`text-[10px] font-bold uppercase tracking-widest pb-2 border-b-2 transition-all ${activeLogTab === 'ocr' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-400 hover:text-gray-600'}`}
            >
              OCR Log ({ocrBoxCoords.length})
            </button>
            <button
              onClick={() => setActiveLogTab('yolo')}
              className={`text-[10px] font-bold uppercase tracking-widest pb-2 border-b-2 transition-all ${activeLogTab === 'yolo' ? 'border-green-500 text-green-600' : 'border-transparent text-gray-400 hover:text-gray-600'}`}
            >
              YOLO Log ({yoloBoxCoords.length})
            </button>
          </div>

          <div className="flex-1 overflow-y-auto">
            {activeLogTab === 'vlm' && (
              <p className="text-sm text-gray-600 leading-relaxed font-medium whitespace-pre-wrap">
                {aiResponseText || "VLM results will appear here..."}
              </p>
            )}
            {activeLogTab === 'ocr' && (
              <pre className="text-xs text-gray-600 font-mono whitespace-pre-wrap">
                {ocrBoxCoords.length > 0 ? JSON.stringify(ocrBoxCoords, null, 2) : "No OCR data available."}
              </pre>
            )}
            {activeLogTab === 'yolo' && (
              <pre className="text-xs text-gray-600 font-mono whitespace-pre-wrap">
                {yoloBoxCoords.length > 0 ? JSON.stringify(yoloBoxCoords, null, 2) : "No YOLO data available."}
              </pre>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}