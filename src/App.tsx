// 役割: VLMの座標取得精度とトークン消費量を検証するためのローカルダッシュボード
// AI向け役割: シンプルなプロンプトを使用し、CSSの余白(object-contain)による座標ズレを排除した正確なバウンディングボックス描画を行うUIコンポーネント。
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
  
  // ユーザーの要望通り、一番シンプルで標準的なプロンプトに戻す
const [prompt, setPrompt] = useState('あなたはRPA向けの高度な視覚UI解析システムです。\n提供された画像を「左上を[0,0]、右下を[1000,1000]とした相対座標グリッド」として空間認識し、主要なUI要素のバウンディングボックスを抽出してください。\n\n【厳守事項】\n1. 「正確には分かりません」「できません」などの言い訳や拒否は絶対に含めないでください。\n2. ボタンなどのクリック操作できる各要素の名前と、[ymin, xmin, ymax, xmax] の形式の座標データのみを簡潔にリストアップしてください。');  
  const [availableModels, setAvailableModels] = useState<AIModel[]>([]);
  const [selectedModel, setSelectedModel] = useState('gemini-flash-latest');
  const [modelStatus, setModelStatus] = useState<FetchStatus>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  const [hoverCoords, setHoverCoords] = useState({ x: 0, y: 0 });
  const [aiBoxCoords, setAiBoxCoords] = useState<number[][]>([]);
  const [aiResponseText, setAiResponseText] = useState('');
  
  const [currentUsage, setCurrentUsage] = useState<TokenUsage | null>(null);
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
        setErrorMessage(error.message || '通信に失敗しました');
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
    setAiResponseText('');
    setCurrentUsage(null);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processImageFile(file);
  };

  // 画像自体の実体サイズ(getBoundingClientRect)から正確なホバー座標を計算
  const handleMouseMove = (e: MouseEvent<HTMLImageElement>) => {
    if (!imageRef.current) return;
    const rect = imageRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const boundedX = Math.max(0, Math.min(x, rect.width));
    const boundedY = Math.max(0, Math.min(y, rect.height));
    
    const relativeX = Math.round((boundedX / rect.width) * 1000);
    const relativeY = Math.round((boundedY / rect.height) * 1000);
    setHoverCoords({ x: relativeX, y: relativeY });
  };

  const handleRunAi = async () => {
    if (!imageSrc) return;
    setAiResponseText('AIに問い合わせ中...');
    setAiBoxCoords([]);

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

      if (!response.ok) throw new Error('サーバー通信エラー');

      const data = await response.json();
      setAiResponseText(data.text);
      
      if (data.usage) {
        setCurrentUsage(data.usage);
        setUsageLog(prev => [{
          timestamp: new Date().toLocaleTimeString(),
          model: selectedModel,
          usage: data.usage
        }, ...prev].slice(0, 10));
      }

      const regex = /\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]/g;
      const matches = [...data.text.matchAll(regex)];
      const extractedCoords = matches.map(match => [
        parseInt(match[1], 10), parseInt(match[2], 10),
        parseInt(match[3], 10), parseInt(match[4], 10),
      ]);
      setAiBoxCoords(extractedCoords);
    } catch (error: any) {
      setAiResponseText('エラー: ' + error.message);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 font-sans text-gray-900">
      <div className="w-1/3 p-6 bg-white shadow-xl flex flex-col gap-6 overflow-y-auto border-r border-gray-200">
        <header>
          <h1 className="text-2xl font-black tracking-tight text-blue-600">VLM RPA ANALYZER</h1>
          <p className="text-xs text-gray-400 font-bold uppercase tracking-tighter">Graduation Research Dashboard</p>
        </header>

        <section>
          <div className="flex justify-between items-end mb-2">
            <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Active Model</label>
            {modelStatus === 'success' && <span className="text-[9px] bg-green-100 text-green-600 px-2 py-0.5 rounded-full font-bold">Online</span>}
            {modelStatus === 'error' && <span className="text-[9px] bg-red-100 text-red-600 px-2 py-0.5 rounded-full font-bold">Offline</span>}
          </div>
          <select 
            className="w-full border-2 border-gray-50 bg-gray-50 p-3 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all text-sm font-medium"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={modelStatus !== 'success'}
          >
            {modelStatus === 'success' ? (
              availableModels.map(m => <option key={m.id} value={m.id}>{m.id}</option>)
            ) : (
              <option>Loading models...</option>
            )}
          </select>
          {modelStatus === 'error' && errorMessage && (
            <p className="text-[10px] text-red-500 mt-2 font-medium">{errorMessage}</p>
          )}
        </section>

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

        {currentUsage && (
          <section className="bg-blue-50/50 p-4 rounded-2xl border border-blue-100 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <h3 className="text-[10px] font-bold text-blue-400 uppercase tracking-widest mb-3">Usage Stats</h3>
            <div className="grid grid-cols-3 gap-2">
              <div className="bg-white p-2 rounded-lg text-center shadow-sm">
                <p className="text-[9px] text-gray-400 font-bold uppercase">In</p>
                <p className="text-sm font-black text-gray-700">{currentUsage.promptTokenCount}</p>
              </div>
              <div className="bg-white p-2 rounded-lg text-center shadow-sm">
                <p className="text-[9px] text-gray-400 font-bold uppercase">Out</p>
                <p className="text-sm font-black text-gray-700">{currentUsage.candidatesTokenCount}</p>
              </div>
              <div className="bg-white p-2 rounded-lg text-center shadow-sm border border-blue-200">
                <p className="text-[9px] text-blue-400 font-bold uppercase">Total</p>
                <p className="text-sm font-black text-blue-600">{currentUsage.totalTokenCount}</p>
              </div>
            </div>
          </section>
        )}

        {usageLog.length > 0 && (
          <section className="mt-2 flex-1 overflow-y-auto">
            <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest block mb-2">Usage History</label>
            <div className="flex flex-col gap-2">
              {usageLog.map((log, i) => (
                <div key={i} className="flex justify-between items-center text-[11px] bg-gray-50 p-2 rounded-lg border border-gray-100">
                  <span className="text-gray-400 font-mono">{log.timestamp}</span>
                  <span className="font-bold text-gray-600 truncate max-w-20">{log.model}</span>
                  <span className="bg-gray-200 px-2 py-0.5 rounded-full font-bold text-gray-700">{log.usage.totalTokenCount} tokens</span>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>

      <main className="flex-1 p-8 flex flex-col gap-6 bg-gray-50/50">
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-black text-gray-800 tracking-tight">Preview Canvas</h2>
          <div className="flex gap-4">
            <div className="bg-gray-900 text-green-400 px-4 py-2 rounded-xl font-mono text-xs shadow-inner">
              X: {hoverCoords.x.toString().padStart(4, '0')}
            </div>
            <div className="bg-gray-900 text-green-400 px-4 py-2 rounded-xl font-mono text-xs shadow-inner">
              Y: {hoverCoords.y.toString().padStart(4, '0')}
            </div>
          </div>
        </div>

        <div className="flex-1 bg-white rounded-3xl border border-gray-200 flex items-center justify-center overflow-hidden relative shadow-sm p-4">
          {!imageSrc ? (
            <div className="text-gray-300 flex flex-col items-center">
              <svg className="w-12 h-12 mb-2 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
              <p className="font-bold text-xs">No image provided</p>
            </div>
          ) : (
            // inline-block で img 要素とぴったり同じサイズのコンテナを作る
            <div className="relative inline-block">
              <img 
                ref={imageRef} 
                src={imageSrc} 
                alt="Target UI" 
                // 余白(object-contain)を作らず、max制約のみで縦横比を維持させる
                className="max-h-[70vh] max-w-full block cursor-crosshair rounded shadow-md"
                style={{ width: 'auto', height: 'auto' }}
                onMouseMove={handleMouseMove}
              />
              {aiBoxCoords.map((coords, i) => (
                <div 
                  key={i}
                  className="absolute border-2 border-red-500 bg-red-500/10 pointer-events-none transition-all duration-500"
                  style={{
                    top: `${coords[0] / 10}%`, 
                    left: `${coords[1] / 10}%`,
                    height: `${(coords[2] - coords[0]) / 10}%`, 
                    width: `${(coords[3] - coords[1]) / 10}%`
                  }}
                >
                  <span className="absolute -top-6 -left-0.5 bg-red-500 text-[9px] text-white px-2 py-0.5 rounded font-black shadow-lg">
                    #{i + 1}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-200 h-40 overflow-y-auto">
          <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3">AI Analysis Data</h3>
          <p className="text-sm text-gray-600 leading-relaxed font-medium whitespace-pre-wrap">
            {aiResponseText || "Results will appear here..."}
          </p>
        </div>
      </main>
    </div>
  );
}