import React from 'react';
import { Award, Calculator } from 'lucide-react';
import { THEMES } from '../theme';

export default function MetricsTable({ metrics = [], onSelectModel, activeModelId, currentTheme }) {
  const t = currentTheme || THEMES.light;
  const isLight = t.id === 'light' || t.id === 'paper';

  const [sortConfig, setSortConfig] = React.useState({ key: 'Overall', direction: 'desc' });

  const handleSort = (key) => {
    setSortConfig(prev => {
      if (prev.key === key) {
        return { key, direction: prev.direction === 'desc' ? 'asc' : 'desc' };
      }
      return { key, direction: 'desc' };
    });
  };

  const sortedMetrics = React.useMemo(() => {
    if (!metrics.length) return [];
    return [...metrics].sort((a, b) => {
      const valA = a[sortConfig.key] ?? -999999;
      const valB = b[sortConfig.key] ?? -999999;
      if (sortConfig.direction === 'asc') {
        return valA > valB ? 1 : -1;
      } else {
        return valA < valB ? 1 : -1;
      }
    });
  }, [metrics, sortConfig]);

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
              Đối đầu tất cả mô hình • Sắp xếp mặc định theo điểm Overall (Cao ➔ Thấp)
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
            <tr className={`${t.tableHeaderBg} border-b ${t.tableHeaderBorder} ${t.textMuted} uppercase text-[11px] font-semibold tracking-wider select-none`}>
              <th className="py-2.5 px-3">Mô hình</th>
              {['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr'].map((col) => (
                <th
                  key={col}
                  onClick={() => handleSort(col)}
                  className="py-2.5 px-3 text-right cursor-pointer hover:opacity-80 transition"
                  title={`Sắp xếp theo ${col}`}
                >
                  <div className="inline-flex items-center gap-1 justify-end">
                    <span>{col}</span>
                    {sortConfig.key === col && (
                      <span className="text-[10px] text-emerald-500 font-bold">
                        {sortConfig.direction === 'asc' ? '▲' : '▼'}
                      </span>
                    )}
                  </div>
                </th>
              ))}
              <th
                onClick={() => handleSort('Overall')}
                className="py-2.5 px-3 text-right font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 cursor-pointer hover:bg-emerald-500/20 transition"
                title="Sắp xếp theo Overall"
              >
                <div className="inline-flex items-center gap-1 justify-end">
                  <span>⭐ Overall</span>
                  {sortConfig.key === 'Overall' && (
                    <span className="text-[10px] font-bold">
                      {sortConfig.direction === 'asc' ? '▲' : '▼'}
                    </span>
                  )}
                </div>
              </th>
            </tr>
          </thead>
          <tbody className={`divide-y ${t.tableBorder}`}>
            {sortedMetrics.map((row) => {
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
                    const isBest = val !== undefined && val !== null && val > 0 && val === bestScores[col];
                    return (
                      <td key={col} className={`py-2.5 px-3 text-right font-mono ${t.textSecondary}`}>
                        <span
                          className={`inline-block px-1.5 py-0.5 rounded ${
                            isBest
                              ? 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 font-bold'
                              : ''
                          }`}
                        >
                          {val !== undefined && val !== null ? val.toFixed(2) : '-'}
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
