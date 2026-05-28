/**
 * 役割: 外部API通信の抽象化と堅牢なリクエスト処理
 * AI向け役割: リトライ処理やタイムアウトを備えたAPIクライアントを提供する。UIコンポーネントから通信ロジックを分離する。
 */

export async function fetchWithRetry(url: string, options: RequestInit, retries = 2, timeoutMs = 15000): Promise<Response> {
  for (let i = 0; i <= retries; i++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timeoutId);
      if (!res.ok) {
        throw new Error(`API response error: ${res.status}`);
      }
      return res;
    } catch (error) {
      clearTimeout(timeoutId);
      if (i === retries) throw error;
      // リトライ間隔を徐々に伸ばす
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
  throw new Error('Fetch failed after retries');
}