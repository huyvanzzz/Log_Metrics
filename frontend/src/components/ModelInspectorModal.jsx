import React from 'react';
import { X, Award, Activity, Gauge } from 'lucide-react';
import { THEMES } from '../theme';

export default function ModelInspectorModal({ modelId, modelDetail, onClose, currentTheme }) {
  if (!modelId) return null;
  const t = currentTheme || THEMES.light;

  const m = modelDetail?.metrics || {};
  const b = modelDetail?.benchmark?.data || {};
  const lossMeta = modelDetail?.loss || {};

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs animate-in fade-in duration-150">
      <div className={`${t.cardBg} border ${t.cardBorder} rounded-2xl w-full max-w-xl max-h-[85vh] overflow-y-auto shadow-2xl p-5 ${t.textPrimary}`}>
        {/* Header Modal */}
        <div className={`flex items-center justify-between pb-3 border-b ${t.cardBorder}`}>
          <div>
            <div className="text-[11px] font-bold uppercase tracking-wider text-sky-500">
              Chi Tiết Phương Pháp
            </div>
            <h2 className="text-lg font-bold mt-0.5">
              {modelId}
            </h2>
          </div>
          <button
            onClick={onClose}
            className={`p-1.5 rounded-lg border ${t.subtleBorder} ${t.subtleBg} hover:opacity-80 transition`}
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Nội dung chi tiết gọn gàng */}
        <div className="mt-4 space-y-4">
          {/* Card Điểm số NLP */}
          <div className={`p-3.5 rounded-xl border ${t.subtleBorder} ${t.subtleBg}`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-bold text-xs">
                <Award className="w-4 h-4" />
                <span>Chỉ Số Đánh Giá (NLP)</span>
              </div>
              <div className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-600 dark:text-emerald-300 font-mono font-bold text-xs">
                Overall: {m.Overall !== undefined ? m.Overall.toFixed(3) : '-'}
              </div>
            </div>

            <div className="grid grid-cols-4 gap-1.5 text-center font-mono text-xs">
              {['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr'].map((k) => (
                <div key={k} className={`p-1.5 rounded border ${t.tableBorder} ${t.cardBg}`}>
                  <div className={`text-[10px] ${t.textMuted}`}>{k}</div>
                  <div className="font-bold mt-0.5">{m[k] || '-'}</div>
                </div>
              ))}
              <div className={`p-1.5 rounded border border-emerald-500/30 bg-emerald-500/10`}>
                <div className="text-[10px] text-emerald-600 dark:text-emerald-400">CIDEr/10</div>
                <div className="font-bold mt-0.5 text-emerald-600 dark:text-emerald-300">
                  {m['CIDEr'] ? (m['CIDEr']/10).toFixed(2) : '-'}
                </div>
              </div>
            </div>
          </div>

          {/* Card Benchmark */}
          <div className={`p-3.5 rounded-xl border ${t.subtleBorder} ${t.subtleBg}`}>
            <div className="flex items-center gap-1.5 text-amber-600 dark:text-amber-400 font-bold text-xs mb-2">
              <Gauge className="w-4 h-4" />
              <span>Độ Trễ & Tốc Độ (Benchmark)</span>
            </div>

            {b.average_generation_ms_per_output_token ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5 text-xs font-mono text-center">
                <div className={`p-2 rounded border ${t.tableBorder} ${t.cardBg}`}>
                  <div className={`text-[10px] ${t.textMuted}`}>Độ trễ/token</div>
                  <div className="font-bold text-amber-500 mt-0.5">{b.average_generation_ms_per_output_token} ms</div>
                </div>
                <div className={`p-2 rounded border ${t.tableBorder} ${t.cardBg}`}>
                  <div className={`text-[10px] ${t.textMuted}`}>Tokens/giây</div>
                  <div className="font-bold text-sky-500 mt-0.5">{b.tokens_per_second} tk/s</div>
                </div>
                <div className={`p-2 rounded border ${t.tableBorder} ${t.cardBg}`}>
                  <div className={`text-[10px] ${t.textMuted}`}>E2E Mean</div>
                  <div className="font-bold mt-0.5">{b.e2e_mean_ms} ms</div>
                </div>
                <div className={`p-2 rounded border ${t.tableBorder} ${t.cardBg}`}>
                  <div className={`text-[10px] ${t.textMuted}`}>Peak VRAM</div>
                  <div className="font-bold mt-0.5">{b.peak_cuda_memory_mb} MB</div>
                </div>
              </div>
            ) : (
              <div className={`text-xs ${t.textMuted} text-center py-2`}>
                Chưa có dữ liệu benchmark.json
              </div>
            )}
          </div>

          {/* Card Training Progress */}
          <div className={`p-3.5 rounded-xl border ${t.subtleBorder} ${t.subtleBg}`}>
            <div className="flex items-center gap-1.5 text-sky-600 dark:text-sky-400 font-bold text-xs mb-2">
              <Activity className="w-4 h-4" />
              <span>Tiến Trình Huấn Luyện</span>
            </div>

            <div className="grid grid-cols-3 gap-1.5 text-xs text-center font-mono">
              <div className={`p-2 rounded border ${t.tableBorder} ${t.cardBg}`}>
                <div className={`text-[10px] ${t.textMuted} font-sans`}>Giai đoạn</div>
                <div className="font-bold mt-0.5 uppercase text-[11px]">
                  {lossMeta.stages ? lossMeta.stages.join(' ➔ ') : 'Finetune'}
                </div>
              </div>
              <div className={`p-2 rounded border ${t.tableBorder} ${t.cardBg}`}>
                <div className={`text-[10px] ${t.textMuted} font-sans`}>Tổng số bước</div>
                <div className="font-bold text-sky-500 mt-0.5">
                  {lossMeta.total_steps ? lossMeta.total_steps.toLocaleString() : '106,690'}
                </div>
              </div>
              <div className={`p-2 rounded border ${t.tableBorder} ${t.cardBg}`}>
                <div className={`text-[10px] ${t.textMuted} font-sans`}>Validation</div>
                <div className="font-bold text-emerald-500 mt-0.5">
                  {lossMeta.validations ? `${lossMeta.validations.length} mốc` : '-'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className={`mt-4 pt-3 border-t ${t.cardBorder} flex justify-end`}>
          <button
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg bg-sky-500 text-white text-xs font-semibold hover:bg-sky-600 transition"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
}
