// 役割: VLMの座標取得精度とトークン消費量を検証するためのローカルダッシュボード
// AI向け役割: モデル一覧の動的取得、複数要素の座標描画、およびトークン使用履歴の管理を行うUIコンポーネント。
import React, { useState, useRef, useEffect, type MouseEvent, type DragEvent } from 'react';

// トークン情報の型定義
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

export default function App() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('この画面のスクリーンショットを見てください。主要なボタンや入力欄などのUI要素の位置をすべて教えてください。出力は [ymin, xmin, ymax, xmax] のように、0から1000の相対座標の形式を含めてください。');
  
  // モデル関連
  const [availableModels, setAvailableModels] = useState<AIModel[]>([]);
  const [selectedModel, setSelectedModel] = useState('gemini-flash-latest');
  
  // 解析結果・座標関連
  const [hoverCoords, setHoverCoords] = useState({ x: 0, y: 0 });
  const [aiBoxCoords, setAiBoxCoords] = useState<number[][]>([]);
  const [aiResponseText, setAiResponseText] = useState('');
  
  // トークン管理関連
  const [currentUsage, setCurrentUsage] = useState<TokenUsage | null>(null);
  const [usageLog, setUsageLog] = useState<UsageHistory[]>([]);
  
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  // 初回レンダリング時に利用可能なモデルを取得
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch('http://localhost:3001/api/models');
        if (res.ok) {
          const data = await res.json();
          setAvailableModels(data.models);
          if (data.models.length > 0) setSelectedModel(data.models[0].id);
        }
      } catch (error) {
        console.error('モデル一覧の取得に失敗しました:', error);
      }
    };
    fetchModels();
  }, []);

  const processImageFile = (file: File) => {
    if (!file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (event) => setImageSrc(event.target?.result as string);
    reader.readAsDataURL(file);
    setAiBoxCoords([]);
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
    const relativeX = Math.round((x / rect.width) * 1000);
    const relativeY = Math.round((y / rect.height) * 1000);
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
      
      // トークン使用量を反映
      if (data.usage) {
        setCurrentUsage(data.usage);
        setUsageLog(prev => [{
          timestamp: new Date().toLocaleTimeString(),
          model: selectedModel,
          usage: data.usage
        }, ...prev].slice(0, 10)); // 直近10件を保持
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
      {/* サイドバー */}
      <div className="w-1/3 p-6 bg-white shadow-xl flex flex-col gap-6 overflow-y-auto">
        <header>
          <h1 className="text-2xl font-black tracking-tight text-blue-600">VLM RPA ANALYZER</h1>
          <p className="text-xs text-gray-500 font-medium">Graduation Research Dashboard</p>
        </header>

        {/* モデル選択 */}
        <section>
          <label className="block text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Active Model</label>
          <select 
            className="w-full border-2 border-gray-100 bg-gray-50 p-3 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {availableModels.length > 0 ? (
              availableModels.map(m => (
                <option key={m.id} value={m.id}>{m.id}</option>
              ))
            ) : (
              <option>Loading models...</option>
            )}
          </select>
        </section>

        {/* アップロードエリア */}
        <section>
          <label className="block text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Target Image</label>
          <div 
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => { e.preventDefault(); setIsDragging(false); const f = e.dataTransfer.files?.[0]; if (f) processImageFile(f); }}
            className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all
              ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-400 hover:bg-gray-50'}`}
          >
            <p className="text-sm font-semibold text-gray-600">Drag & Drop or Click</p>
            <input type="file" accept="image/*" ref={fileInputRef} onChange={handleImageUpload} className="hidden" />
          </div>
        </section>

        {/* プロンプト入力 */}
        <section>
          <label className="block text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">System Prompt</label>
          <textarea 
            className="w-full border-2 border-gray-100 bg-gray-50 p-4 rounded-2xl h-32 text-sm focus:ring-2 focus:ring-blue-500 outline-none resize-none"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </section>

        <button 
          onClick={handleRunAi}
          disabled={!imageSrc}
          className="w-full bg-blue-600 text-white py-4 rounded-2xl hover:bg-blue-700 disabled:bg-gray-200 font-bold shadow-lg shadow-blue-200 transition-all active:scale-95"
        >
          推論を実行
        </button>

        {/* 今回追加：トークン使用量表示エリア */}
        {currentUsage && (
          <section className="bg-blue-50 p-4 rounded-2xl border border-blue-100 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <h3 className="text-xs font-bold text-blue-600 uppercase tracking-widest mb-3">Last Request Usage</h3>
            <div className="grid grid-cols-3 gap-2">
              <div className="bg-white p-2 rounded-lg text-center shadow-sm">
                <p className="text-[10px] text-gray-400 font-bold uppercase">Prompt</p>
                <p className="text-sm font-black text-gray-700">{currentUsage.promptTokenCount}</p>
              </div>
              <div className="bg-white p-2 rounded-lg text-center shadow-sm">
                <p className="text-[10px] text-gray-400 font-bold uppercase">Response</p>
                <p className="text-sm font-black text-gray-700">{currentUsage.candidatesTokenCount}</p>
              </div>
              <div className="bg-white p-2 rounded-lg text-center shadow-sm border-2 border-blue-200">
                <p className="text-[10px] text-blue-400 font-bold uppercase">Total</p>
                <p className="text-sm font-black text-blue-600">{currentUsage.totalTokenCount}</p>
              </div>
            </div>
          </section>
        )}

        {/* 今回追加：履歴ログ */}
        {usageLog.length > 0 && (
          <section className="mt-2">
            <label className="block text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Usage History</label>
            <div className="flex flex-col gap-2">
              {usageLog.map((log, i) => (
                <div key={i} className="flex justify-between items-center text-[11px] bg-gray-50 p-2 rounded-lg border border-gray-100">
                  <span className="text-gray-400 font-mono">{log.timestamp}</span>
                  <span className="font-bold text-gray-600 truncate max-w-[80px]">{log.model}</span>
                  <span className="bg-gray-200 px-2 py-0.5 rounded-full font-bold">{log.usage.totalTokenCount} tokens</span>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>

      {/* メインプレビューエリア */}
      <main className="flex-1 p-8 flex flex-col gap-6 overflow-hidden">
        <div className="flex justify-between items-center bg-white px-6 py-4 rounded-2xl shadow-sm border border-gray-100">
          <h2 className="font-bold text-gray-500">Preview Canvas</h2>
          <div className="flex gap-4">
            <div className="bg-gray-900 text-green-400 px-4 py-2 rounded-xl font-mono text-xs shadow-inner">
              X: {hoverCoords.x.toString().padStart(4, '0')}
            </div>
            <div className="bg-gray-900 text-green-400 px-4 py-2 rounded-xl font-mono text-xs shadow-inner">
              Y: {hoverCoords.y.toString().padStart(4, '0')}
            </div>
          </div>
        </div>

        <div className="flex-1 bg-white rounded-3xl border-2 border-gray-100 flex items-center justify-center overflow-hidden relative shadow-inner p-4">
          {!imageSrc ? (
            <div className="text-gray-300 flex flex-col items-center">
              <div className="w-20 h-20 bg-gray-50 rounded-full flex items-center justify-center mb-4">
                <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
              </div>
              <p className="font-bold text-sm">No Image Loaded</p>
            </div>
          ) : (
            <div className="relative inline-block group">
              <img 
                ref={imageRef} src={imageSrc} alt="Target UI" 
                className="max-h-[70vh] w-auto cursor-crosshair rounded-xl shadow-2xl transition-all"
                onMouseMove={handleMouseMove}
              />
              {aiBoxCoords.map((coords, i) => (
                <div 
                  key={i}
                  className="absolute border-2 border-red-500 bg-red-500/10 pointer-events-none transition-all duration-500"
                  style={{
                    top: `${coords[0] / 10}%`, left: `${coords[1] / 10}%`,
                    height: `${(coords[2] - coords[0]) / 10}%`, width: `${(coords[3] - coords[1]) / 10}%`
                  }}
                >
                  <span className="absolute -top-6 -left-0.5 bg-red-500 text-[10px] text-white px-2 py-0.5 rounded-t-md font-black shadow-lg">
                    #{i + 1}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* AIレスポンスの下部表示 */}
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-100 max-h-48 overflow-y-auto">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3">AI Raw Output</h3>
          <p className="text-sm text-gray-700 leading-relaxed font-medium">
            {aiResponseText || "Waiting for execution..."}
          </p>
        </div>
      </main>
    </div>
  );
}