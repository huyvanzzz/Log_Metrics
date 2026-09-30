import React from 'react';
import { Gauge, Cpu, AlertCircle } from 'lucide-react';
import { THEMES } from '../theme';

export default function BenchmarkTable({ benchmarks = [], onSelectModel, activeModelId, currentTheme }) {
  const t = currentTheme || THEMES.light;

  return (
    <div className={`${t.cardBg} border ${t.cardBorder} rounded-2xl p-5 ${t.cardShadow} transition`}>
      {/* Header */}
      <div className={`flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b ${t.cardBorder}`}>
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-amber-500/10 text-amber-500">
            <Gauge className="w-4 h-4" />
          </div>
          <div>
            <h3 className={`text-base font-bold ${t.textPrimary}`}>
              So Sánh Tốc Độ & Phần Cứng
            </h3>
            <p className={`text-xs ${t.textMuted}`}>
              Độ trễ suy luận (E2E, P50/P95), tốc độ sinh token và bộ nhớ GPU
            </p>
          </div>
        </div>

        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border ${t.subtleBorder} ${t.subtleBg} text-xs ${t.textSecondary}`}>
          <Cpu className="w-3.5 h-3.5 text-sky-500" />
          <span>GPU: Tesla T4</span>
        </div>
      </div>

      {/* Bảng */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className={`${t.tableHeaderBg} border-b ${t.tableHeaderBorder} ${t.textMuted} uppercase text-[11px] font-semibold tracking-wider`}>
              <th className="py-2.5 px-3">Mô hình</th>
              <th className="py-2.5 px-3 text-right">E2E Mean (ms)</th>
              <th className="py-2.5 px-3 text-right">E2E P50 (ms)</th>
              <th className="py-2.5 px-3 text-right">E2E P95 (ms)</th>
              <th className="py-2.5 px-3 text-right">Gen Mean (ms)</th>
              <th className="py-2.5 px-3 text-right text-amber-600 dark:text-amber-400 bg-amber-500/10">
                ⚡ ms/token
              </th>
              <th className="py-2.5 px-3 text-right text-sky-600 dark:text-sky-400 bg-sky-500/10">
                🚀 Tokens/s
              </th>
              <th className="py-2.5 px-3 text-right">Peak VRAM</th>
              <th className="py-2.5 px-3 text-right">GPU</th>
            </tr>
          </thead>
          <tbody className={`divide-y ${t.tableBorder}`}>
            {benchmarks.map((row) => {
              const isSelected = activeModelId === row.model_id;
              const d = row.data;
              const hasData = row.has_benchmark && d;

              return (
                <tr
                  key={row.model_id}
                  onClick={() => onSelectModel && onSelectModel(row.model_id)}
                  className={`transition-colors cursor-pointer ${
                    isSelected
                      ? 'bg-amber-500/10 border-l-3 border-amber-500'
                      : t.tableRowHover
                  }`}
                >
                  <td className={`py-2.5 px-3 font-semibold ${t.textPrimary} flex items-center gap-2`}>
                    <span className="w-2 h-2 rounded-full bg-slate-400"></span>
                    <span>{row.model_id}</span>
                  </td>

                  {hasData ? (
                    <>
                      <td className={`py-2.5 px-3 text-right font-mono ${t.textSecondary}`}>
                        {d.e2e_mean_ms?.toFixed(1)}
                      </td>
                      <td className={`py-2.5 px-3 text-right font-mono ${t.textSecondary}`}>
                        {d.e2e_p50_ms?.toFixed(1)}
                      </td>
                      <td className={`py-2.5 px-3 text-right font-mono ${t.textMuted}`}>
                        {d.e2e_p95_ms?.toFixed(1)}
                      </td>
                      <td className={`py-2.5 px-3 text-right font-mono ${t.textSecondary}`}>
                        {d.generation_mean_ms?.toFixed(1)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono font-bold text-amber-600 dark:text-amber-400 bg-amber-500/5">
                        {d.average_generation_ms_per_output_token?.toFixed(2)} ms
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono font-bold text-sky-600 dark:text-sky-400 bg-sky-500/5">
                        {d.tokens_per_second?.toFixed(1)} tk/s
                      </td>
                      <td className={`py-2.5 px-3 text-right font-mono ${t.textSecondary}`}>
                        {d.peak_cuda_memory_mb ? `${d.peak_cuda_memory_mb} MB` : '-'}
                      </td>
                      <td className={`py-2.5 px-3 text-right font-mono text-[11px] ${t.textMuted}`}>
                        {d.gpu || 'GPU'}
                      </td>
                    </>
                  ) : (
                    <td colSpan={8} className="py-2.5 px-3 text-center">
                      <div className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs ${t.textMuted}`}>
                        <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
                        <span>Chưa có file benchmark.json</span>
                      </div>
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
