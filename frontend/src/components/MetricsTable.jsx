import React from 'react';
import { Award, Calculator } from 'lucide-react';
import { THEMES } from '../theme';

export default function MetricsTable({ metrics = [], onSelectModel, activeModelId, currentTheme }) {
  const t = currentTheme || THEMES.light;
  const isLight = t.id === 'light' || t.id === 'paper';

  const bestScores = React.useMemo(() => {
    if (!metrics.length) return {};
    const keys = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr', 'Overall'];
    const maxMap = {};
    keys.forEach(k => {
      const vals = metrics.map(m => m[k] || 0);
      maxMap[k] = Math.max(...vals);
    });
    return maxMap;
  }, [metrics]);

  return (
    <div className={`${t.cardBg} border ${t.cardBorder} rounded-2xl p-5 ${t.cardShadow} transition`}>
      {/* Header */}
      <div className={`flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b ${t.cardBorder}`}>
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-500">
            <Award className="w-4 h-4" />
          </div>
          <div>
            <h3 className={`text-base font-bold ${t.textPrimary}`}>
              So Sánh Điểm Số NLP
            </h3>
            <p className={`text-xs ${t.textMuted}`}>
              Đối đầu tất cả mô hình • Highlight điểm cao nhất (Best Score)
            </p>
          </div>
        </div>

        {/* Công thức Overall */}
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border ${t.subtleBorder} ${t.subtleBg} text-emerald-600 dark:text-emerald-400 text-xs font-mono`}>
          <Calculator className="w-3.5 h-3.5 shrink-0" />
          <span>Overall = (BLEU-4 + METEOR + ROUGE_L + CIDEr/10) / 4</span>
        </div>
      </div>

      {/* Bảng */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className={`${t.tableHeaderBg} border-b ${t.tableHeaderBorder} ${t.textMuted} uppercase text-[11px] font-semibold tracking-wider`}>
              <th className="py-2.5 px-3">Mô hình</th>
              <th className="py-2.5 px-3 text-right">BLEU-1</th>
              <th className="py-2.5 px-3 text-right">BLEU-2</th>
              <th className="py-2.5 px-3 text-right">BLEU-3</th>
              <th className="py-2.5 px-3 text-right">BLEU-4</th>
              <th className="py-2.5 px-3 text-right">METEOR</th>
              <th className="py-2.5 px-3 text-right">ROUGE-L</th>
              <th className="py-2.5 px-3 text-right">CIDEr</th>
              <th className="py-2.5 px-3 text-right font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10">
                ⭐ Overall
              </th>
            </tr>
          </thead>
          <tbody className={`divide-y ${t.tableBorder}`}>
            {metrics.map((row) => {
              const isSelected = activeModelId === row.model_id;
              return (
                <tr
                  key={row.model_id}
                  onClick={() => onSelectModel && onSelectModel(row.model_id)}
                  className={`transition-colors cursor-pointer ${
                    isSelected
                      ? 'bg-sky-500/10 border-l-3 border-sky-500'
                      : t.tableRowHover
                  }`}
                >
                  <td className={`py-2.5 px-3 font-semibold ${t.textPrimary} flex items-center gap-2`}>
                    <span className="w-2 h-2 rounded-full bg-slate-400"></span>
                    <span>{row.model_id}</span>
                  </td>
                  {['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr'].map((col) => {
                    const val = row[col];
                    const isBest = val !== undefined && val === bestScores[col];
                    return (
                      <td key={col} className={`py-2.5 px-3 text-right font-mono ${t.textSecondary}`}>
                        <span
                          className={`inline-block px-1.5 py-0.5 rounded ${
                            isBest
                              ? 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 font-bold'
                              : ''
                          }`}
                        >
                          {val !== undefined ? val.toFixed(2) : '-'}
                        </span>
                      </td>
                    );
                  })}
                  <td className="py-2.5 px-3 text-right font-mono font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-500/5">
                    <span
                      className={`inline-block px-2 py-0.5 rounded ${
                        row.Overall === bestScores['Overall']
                          ? 'bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 font-extrabold shadow-xs'
                          : ''
                      }`}
                    >
                      {row.Overall !== undefined ? row.Overall.toFixed(3) : '-'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
