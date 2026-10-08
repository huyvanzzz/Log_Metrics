import React from 'react';
import { Award, Calculator } from 'lucide-react';
import { THEMES } from '../theme';

export default function MetricsTable({
  metrics = [],
  selectedModels = [],
  onSelectModel,
  activeModelId,
  currentTheme
}) {
  const t = currentTheme || THEMES.light;
  const isLight = t.id === 'light' || t.id === 'paper';

  const [sortConfig, setSortConfig] = React.useState({ key: 'Overall', direction: 'desc' });
  const [filterMode, setFilterMode] = React.useState('all'); // 'all' | 'selected'
  const [limitTop10, setLimitTop10] = React.useState(true);

  const handleSort = (key) => {
    setSortConfig(prev => {
      if (prev.key === key) {
        return { key, direction: prev.direction === 'desc' ? 'asc' : 'desc' };
      }
      return { key, direction: 'desc' };
    });
  };

  const filteredMetrics = React.useMemo(() => {
    if (filterMode === 'selected' && selectedModels && selectedModels.length > 0) {
      const subset = metrics.filter(m => selectedModels.includes(m.model_id));
      return subset.length > 0 ? subset : metrics;
    }
    return metrics;
  }, [metrics, selectedModels, filterMode]);

  const sortedMetrics = React.useMemo(() => {
    if (!filteredMetrics.length) return [];
    return [...filteredMetrics].sort((a, b) => {
      const valA = a[sortConfig.key] ?? -999999;
      const valB = b[sortConfig.key] ?? -999999;
      if (sortConfig.direction === 'asc') {
        return valA > valB ? 1 : -1;
      } else {
        return valA < valB ? 1 : -1;
      }
    });
  }, [filteredMetrics, sortConfig]);

  const displayMetrics = React.useMemo(() => {
    if (limitTop10 && filterMode !== 'selected') {
      return sortedMetrics.slice(0, 10);
    }
    return sortedMetrics;
  }, [sortedMetrics, limitTop10, filterMode]);

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
            <div className="flex items-center gap-2">
              <h3 className={`text-base font-bold ${t.textPrimary}`}>
                So Sánh Điểm Số NLP
              </h3>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                {filterMode === 'selected' && selectedModels.length > 0
                  ? `Đã chọn: ${displayMetrics.length}/${metrics.length}`
                  : limitTop10 && sortedMetrics.length > 10
                  ? `Top 10 / ${sortedMetrics.length}`
                  : `Tất cả (${sortedMetrics.length})`}
              </span>
            </div>
            <p className={`text-xs ${t.textMuted}`}>
              Đối đầu mô hình • {filterMode === 'selected' && selectedModels.length > 0 ? `Chỉ hiển thị ${displayMetrics.length} mô hình đã chọn` : limitTop10 ? 'Hiển thị Top 10 cao nhất' : 'Hiển thị toàn bộ mô hình'} (Click cột để đổi sắp xếp)
            </p>
          </div>
        </div>

        {/* Nút lọc & Công thức Overall */}
        <div className="flex items-center gap-2 flex-wrap sm:flex-nowrap">
          {selectedModels.length > 0 && selectedModels.length < metrics.length && (
            <button
              onClick={() => setFilterMode(prev => prev === 'selected' ? 'all' : 'selected')}
              className={`px-2.5 py-1 rounded-lg text-xs font-semibold border transition cursor-pointer ${
                filterMode === 'selected'
                  ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-600 dark:text-emerald-400 font-bold'
                  : `${t.subtleBg} ${t.subtleBorder} ${t.textSecondary} hover:${t.textPrimary}`
              }`}
            >
              {filterMode === 'selected' ? `Hiện tất cả (${metrics.length})` : `Chỉ hiện đã chọn (${selectedModels.length})`}
            </button>
          )}

          {sortedMetrics.length > 10 && filterMode !== 'selected' && (
            <button
              onClick={() => setLimitTop10(!limitTop10)}
              className={`px-2.5 py-1 rounded-lg text-xs font-semibold border transition cursor-pointer ${
                limitTop10
                  ? `${t.subtleBg} ${t.subtleBorder} ${t.textSecondary} hover:${t.textPrimary}`
                  : 'bg-emerald-500/15 border-emerald-500/30 text-emerald-600 dark:text-emerald-400 font-bold'
              }`}
            >
              {limitTop10 ? `Xem tất cả (${sortedMetrics.length})` : 'Thu gọn Top 10'}
            </button>
          )}

          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border ${t.subtleBorder} ${t.subtleBg} text-emerald-600 dark:text-emerald-400 text-xs font-mono shrink-0`}>
            <Calculator className="w-3.5 h-3.5 shrink-0" />
            <span>Overall = (BLEU-4 + METEOR + ROUGE_L + CIDEr/10) / 4</span>
          </div>
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
            {displayMetrics.map((row) => {
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

      {sortedMetrics.length > 10 && limitTop10 && (
        <div className={`mt-3 text-center text-xs ${t.textMuted} flex items-center justify-center gap-2 border-t ${t.subtleBorder} pt-2`}>
          <span>* Đang hiển thị Top 10 mô hình có điểm cao nhất.</span>
          <button
            onClick={() => setLimitTop10(false)}
            className="text-emerald-600 dark:text-emerald-400 font-semibold hover:underline cursor-pointer"
          >
            Xem tất cả {sortedMetrics.length} mô hình ➔
          </button>
        </div>
      )}
    </div>
  );
}
