import React from 'react';
import { Gauge, Cpu, AlertCircle } from 'lucide-react';
import { THEMES } from '../theme';

export default function BenchmarkTable({
  benchmarks = [],
  selectedModels = [],
  onSelectModel,
  activeModelId,
  currentTheme
}) {
  const t = currentTheme || THEMES.light;

  const [sortConfig, setSortConfig] = React.useState({
    key: 'average_generation_ms_per_output_token',
    direction: 'asc'
  });
  const [filterMode, setFilterMode] = React.useState('all'); // 'all' | 'selected'
  const [limitTop10, setLimitTop10] = React.useState(true);

  const handleSort = (key) => {
    setSortConfig(prev => {
      if (prev.key === key) {
        return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
      }
      const defaultDir = key === 'tokens_per_second' ? 'desc' : 'asc';
      return { key, direction: defaultDir };
    });
  };

  const filteredBenchmarks = React.useMemo(() => {
    if (filterMode === 'selected' && selectedModels && selectedModels.length > 0) {
      const subset = benchmarks.filter(b => selectedModels.includes(b.model_id));
      return subset.length > 0 ? subset : benchmarks;
    }
    return benchmarks;
  }, [benchmarks, selectedModels, filterMode]);

  const sortedBenchmarks = React.useMemo(() => {
    if (!filteredBenchmarks.length) return [];
    return [...filteredBenchmarks].sort((a, b) => {
      const hasA = a.has_benchmark && a.data && a.data[sortConfig.key] !== undefined;
      const hasB = b.has_benchmark && b.data && b.data[sortConfig.key] !== undefined;

      if (!hasA && !hasB) return 0;
      if (!hasA) return 1;
      if (!hasB) return -1;

      const valA = a.data[sortConfig.key];
      const valB = b.data[sortConfig.key];

      if (sortConfig.direction === 'asc') {
        return valA > valB ? 1 : -1;
      } else {
        return valA < valB ? 1 : -1;
      }
    });
  }, [filteredBenchmarks, sortConfig]);

  const displayBenchmarks = React.useMemo(() => {
    if (limitTop10 && filterMode !== 'selected') {
      return sortedBenchmarks.slice(0, 10);
    }
    return sortedBenchmarks;
  }, [sortedBenchmarks, limitTop10, filterMode]);

  return (
    <div className={`${t.cardBg} border ${t.cardBorder} rounded-2xl p-5 ${t.cardShadow} transition`}>
      {/* Header */}
      <div className={`flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b ${t.cardBorder}`}>
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-amber-500/10 text-amber-500">
            <Gauge className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className={`text-base font-bold ${t.textPrimary}`}>
                So Sánh Tốc Độ & Phần Cứng
              </h3>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                {filterMode === 'selected' && selectedModels.length > 0
                  ? `Đã chọn: ${displayBenchmarks.length}/${benchmarks.length}`
                  : limitTop10 && sortedBenchmarks.length > 10
                  ? `Top 10 / ${sortedBenchmarks.length}`
                  : `Tất cả (${sortedBenchmarks.length})`}
              </span>
            </div>
            <p className={`text-xs ${t.textMuted}`}>
              Độ trễ suy luận • {filterMode === 'selected' && selectedModels.length > 0 ? `Chỉ hiển thị ${displayBenchmarks.length} mô hình đã chọn` : limitTop10 ? 'Hiển thị Top 10 nhanh nhất' : 'Hiển thị toàn bộ mô hình'} (Click cột để đổi sắp xếp)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap sm:flex-nowrap">
          {selectedModels.length > 0 && selectedModels.length < benchmarks.length && (
            <button
              onClick={() => setFilterMode(prev => prev === 'selected' ? 'all' : 'selected')}
              className={`px-2.5 py-1 rounded-lg text-xs font-semibold border transition cursor-pointer ${
                filterMode === 'selected'
                  ? 'bg-amber-500/15 border-amber-500/30 text-amber-600 dark:text-amber-400 font-bold'
                  : `${t.subtleBg} ${t.subtleBorder} ${t.textSecondary} hover:${t.textPrimary}`
              }`}
            >
              {filterMode === 'selected' ? `Hiện tất cả (${benchmarks.length})` : `Chỉ hiện đã chọn (${selectedModels.length})`}
            </button>
          )}

          {sortedBenchmarks.length > 10 && filterMode !== 'selected' && (
            <button
              onClick={() => setLimitTop10(!limitTop10)}
              className={`px-2.5 py-1 rounded-lg text-xs font-semibold border transition cursor-pointer ${
                limitTop10
                  ? `${t.subtleBg} ${t.subtleBorder} ${t.textSecondary} hover:${t.textPrimary}`
                  : 'bg-amber-500/15 border-amber-500/30 text-amber-600 dark:text-amber-400 font-bold'
              }`}
            >
              {limitTop10 ? `Xem tất cả (${sortedBenchmarks.length})` : 'Thu gọn Top 10'}
            </button>
          )}

          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border ${t.subtleBorder} ${t.subtleBg} text-xs ${t.textSecondary} shrink-0`}>
            <Cpu className="w-3.5 h-3.5 text-sky-500" />
            <span>GPU: Tesla T4</span>
          </div>
        </div>
      </div>

      {/* Bảng */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className={`${t.tableHeaderBg} border-b ${t.tableHeaderBorder} ${t.textMuted} uppercase text-[11px] font-semibold tracking-wider select-none`}>
              <th className="py-2.5 px-3">Mô hình</th>
              <th
                onClick={() => handleSort('e2e_mean_ms')}
                className="py-2.5 px-3 text-right cursor-pointer hover:opacity-80 transition"
                title="Sắp xếp theo E2E Mean"
              >
                <div className="inline-flex items-center gap-1 justify-end">
                  <span>E2E Mean (ms)</span>
                  {sortConfig.key === 'e2e_mean_ms' && (
                    <span className="text-[10px] text-amber-500 font-bold">
                      {sortConfig.direction === 'asc' ? '▲' : '▼'}
                    </span>
                  )}
                </div>
              </th>
              <th
                onClick={() => handleSort('e2e_p50_ms')}
                className="py-2.5 px-3 text-right cursor-pointer hover:opacity-80 transition"
              >
                <div className="inline-flex items-center gap-1 justify-end">
                  <span>E2E P50 (ms)</span>
                  {sortConfig.key === 'e2e_p50_ms' && (
                    <span className="text-[10px] text-amber-500 font-bold">
                      {sortConfig.direction === 'asc' ? '▲' : '▼'}
                    </span>
                  )}
                </div>
              </th>
              <th
                onClick={() => handleSort('e2e_p95_ms')}
                className="py-2.5 px-3 text-right cursor-pointer hover:opacity-80 transition"
              >
                <div className="inline-flex items-center gap-1 justify-end">
                  <span>E2E P95 (ms)</span>
                  {sortConfig.key === 'e2e_p95_ms' && (
                    <span className="text-[10px] text-amber-500 font-bold">
                      {sortConfig.direction === 'asc' ? '▲' : '▼'}
                    </span>
                  )}
                </div>
              </th>
              <th
                onClick={() => handleSort('generation_mean_ms')}
                className="py-2.5 px-3 text-right cursor-pointer hover:opacity-80 transition"
              >
                <div className="inline-flex items-center gap-1 justify-end">
                  <span>Gen Mean (ms)</span>
                  {sortConfig.key === 'generation_mean_ms' && (
                    <span className="text-[10px] text-amber-500 font-bold">
                      {sortConfig.direction === 'asc' ? '▲' : '▼'}
                    </span>
                  )}
                </div>
              </th>
              <th
                onClick={() => handleSort('average_generation_ms_per_output_token')}
                className="py-2.5 px-3 text-right font-bold text-amber-600 dark:text-amber-400 bg-amber-500/10 cursor-pointer hover:bg-amber-500/20 transition"
                title="Sắp xếp theo ms/token (Nhanh nhất lên đầu)"
              >
                <div className="inline-flex items-center gap-1 justify-end">
                  <span>⚡ ms/token</span>
                  {sortConfig.key === 'average_generation_ms_per_output_token' && (
                    <span className="text-[10px] font-bold">
                      {sortConfig.direction === 'asc' ? '▲' : '▼'}
                    </span>
                  )}
                </div>
              </th>
              <th
                onClick={() => handleSort('tokens_per_second')}
                className="py-2.5 px-3 text-right font-bold text-sky-600 dark:text-sky-400 bg-sky-500/10 cursor-pointer hover:bg-sky-500/20 transition"
                title="Sắp xếp theo Tokens/s"
              >
                <div className="inline-flex items-center gap-1 justify-end">
                  <span>🚀 Tokens/s</span>
                  {sortConfig.key === 'tokens_per_second' && (
                    <span className="text-[10px] font-bold">
                      {sortConfig.direction === 'asc' ? '▲' : '▼'}
                    </span>
                  )}
                </div>
              </th>
              <th
                onClick={() => handleSort('peak_cuda_memory_mb')}
                className="py-2.5 px-3 text-right cursor-pointer hover:opacity-80 transition"
              >
                <div className="inline-flex items-center gap-1 justify-end">
                  <span>Peak VRAM</span>
                  {sortConfig.key === 'peak_cuda_memory_mb' && (
                    <span className="text-[10px] text-amber-500 font-bold">
                      {sortConfig.direction === 'asc' ? '▲' : '▼'}
                    </span>
                  )}
                </div>
              </th>
              <th className="py-2.5 px-3 text-right">GPU</th>
            </tr>
          </thead>
          <tbody className={`divide-y ${t.tableBorder}`}>
            {displayBenchmarks.map((row) => {
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

      {sortedBenchmarks.length > 10 && limitTop10 && (
        <div className={`mt-3 text-center text-xs ${t.textMuted} flex items-center justify-center gap-2 border-t ${t.subtleBorder} pt-2`}>
          <span>* Đang hiển thị Top 10 mô hình có tốc độ nhanh nhất.</span>
          <button
            onClick={() => setLimitTop10(false)}
            className="text-amber-600 dark:text-amber-400 font-semibold hover:underline cursor-pointer"
          >
            Xem tất cả {sortedBenchmarks.length} mô hình ➔
          </button>
        </div>
      )}
    </div>
  );
}
