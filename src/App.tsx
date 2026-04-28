/**
 * 役割: OCR/VLM/YOLO の解析結果を可視化・デバッグするためのダッシュボードUI
 * AI向け役割: 座標データの正規化、レイヤー別の表示切り替え、およびトークン使用履歴の管理を行う。
 */
import React, { useState, useRef, useEffect, type MouseEvent } from 'react';

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

type FetchStatus = 'idle' | 'loading' | 'success' | 'error';

export default function App() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('画面内の主要なメニューやボタンをすべてリストアップしてください。');  
  const [availableModels, setAvailableModels] = useState<AIModel[]>([]);
  const [selectedModel, setSelectedModel] = useState('gemini-flash-latest');
  const [modelStatus, setModelStatus] = useState<FetchStatus>('idle');
  
  // TypeScript警告(6133)の解消: エラーメッセージを表示用に活用
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  const [hoverCoords, setHoverCoords] = useState({ x: 0, y: 0 });
  
  const [aiBoxCoords, setAiBoxCoords] = useState<number[][]>([]);
  const [ocrBoxCoords, setOcrBoxCoords] = useState<any[]>([]); 
  const [yoloBoxCoords, setYoloBoxCoords] = useState<any[]>([]); 
  const [aiResponseText, setAiResponseText] = useState('');

  const [showOcr, setShowOcr] = useState(true);
  const [showVlm, setShowVlm] = useState(true);
  const [showYolo, setShowYolo] = useState(true);
  
  const [currentUsage, setCurrentUsage] = useState<TokenUsage | null>(null);
  
  // TypeScript警告(6133)の解消: 使用履歴をUI下部に表示
  const [usageLog, setUsageLog] = useState<UsageHistory[]>([]);
  
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const fetchModels = async () => {
      setModelStatus('loading');
      setErrorMessage(null);
      try {
        const res = await fetch('http://localhost:3001/api/models');
        if (!res.ok) throw new Error(`サーバーエラー: ${res.status}`);
        const data = await res.json();
        if (data.models && data.models.length > 0) {
          setAvailableModels(data.models);
          setSelectedModel(data.models[0].id);
          setModelStatus('success');
        } else {
          throw new Error('利用可能なモデルが見つかりませんでした');
        }
      } catch (error: any) {
        console.error('モデル一覧の取得に失敗しました:', error);
        setModelStatus('error');
        setErrorMessage(error.message || 'モデルの取得中に不明なエラーが発生しました');
      }
    };
    fetchModels();
  }, []);

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
    if (!imageSrc) return;
    setAiResponseText('AIに問い合わせ中...');
    setAiBoxCoords([]);
    setOcrBoxCoords([]);
    setErrorMessage(null);

    try {
      const resBlob = await fetch(imageSrc);
      const blob = await resBlob.blob();

      const formData = new FormData();
      formData.append('image', blob, 'screenshot.png');
      formData.append('prompt', prompt);
      formData.append('model', selectedModel);

      const response = await fetch('http://localhost:3001/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`解析サーバーエラー: ${response.status}`);
      const data = await response.json();
      
      setAiResponseText(data.text);
      if (data.ocrElements) setOcrBoxCoords(data.ocrElements);
      if (data.yoloElements) setYoloBoxCoords(data.yoloElements);
      
      if (data.usage) {
        setCurrentUsage(data.usage);
        setUsageLog(prev => [{
          timestamp: new Date().toLocaleTimeString(),
          model: selectedModel,
          usage: data.usage
        }, ...prev].slice(0, 10));
      }

      const regex = /\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]/g;
      const matches = [...data.text.matchAll(regex)];
      const extractedCoords = matches.map(match => [
        parseFloat(match[1]), parseFloat(match[2]),
        parseFloat(match[3]), parseFloat(match[4]),
      ]);
      setAiBoxCoords(extractedCoords);
    } catch (error: any) {
      setAiResponseText('エラーが発生しました。');
      setErrorMessage(error.message);
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
          disabled={!imageSrc || modelStatus !== 'success'}
          className="w-full bg-blue-600 text-white py-4 rounded-2xl hover:bg-blue-700 disabled:bg-gray-200 font-bold shadow-lg shadow-blue-100 transition-all active:scale-95 text-sm"
        >
          Run Analysis
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
                className="max-h-[70vh] max-w-full block cursor-crosshair shadow-2xl ring-4 ring-gray-100"
                style={{ width: 'auto', height: 'auto' }}
                onMouseMove={handleMouseMove}
              />
              
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

              {/* YOLO レイヤー (緑) - 将来用 */}
              {showYolo && yoloBoxCoords.map((el: any, i: number) => (
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

        <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-200 h-40 overflow-y-auto">
          <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3">AI Analysis Data</h3>
          <p className="text-sm text-gray-600 leading-relaxed font-medium whitespace-pre-wrap">{aiResponseText || "Results will appear here..."}</p>
        </div>
      </main>
    </div>
  );
}