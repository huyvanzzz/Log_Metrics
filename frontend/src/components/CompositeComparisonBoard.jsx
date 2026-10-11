import React, { useRef, useState } from 'react';
import { toPng } from 'html-to-image';
import { Camera, Check, RefreshCw } from 'lucide-react';
import LossVisualizer from './LossVisualizer';
import MetricsTable from './MetricsTable';
import BenchmarkTable from './BenchmarkTable';
import TradeoffScatter from './TradeoffScatter';
import { THEMES } from '../theme';

export default function CompositeComparisonBoard({
  allModels = [],
  selectedModels = [],
  onToggleModel,
  lossData = {},
  metricsData = [],
  benchmarkData = [],
  tradeoffData = [],
  loading = false,
  onSelectModelDetail,
  activeModelId,
  currentTheme
}) {
  const boardRef = useRef(null);
  const [exporting, setExporting] = useState(false);
  const [exportSuccess, setExportSuccess] = useState(false);
  const t = currentTheme || THEMES.light;
  const isLight = t.id === 'light' || t.id === 'paper';

  const handleExportSnapshot = async () => {
    if (!boardRef.current) return;
    try {
      setExporting(true);
      setExportSuccess(false);

      await new Promise(r => setTimeout(r, 200));

      const dataUrl = await toPng(boardRef.current, {
        pixelRatio: 2,
        backgroundColor: isLight ? '#ffffff' : t.id === 'midnight' ? '#000000' : '#0f172a',
        cacheBust: true
      });

      const link = document.createElement('a');
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      link.download = `AD_Model_Comparison_${t.id}_${timestamp}.png`;
      link.href = dataUrl;
      link.click();

      setExportSuccess(true);
      setTimeout(() => setExportSuccess(false), 3000);
    } catch (err) {
      console.error('Lỗi khi xuất ảnh:', err);
      alert('Không thể tạo file ảnh: ' + err.message);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* Top action bar: Nút chụp ảnh xuất báo cáo */}
      <div className={`flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 rounded-2xl ${t.cardBg} border ${t.cardBorder} ${t.cardShadow}`}>
        <div>
          <h2 className={`text-base font-bold ${t.textPrimary}`}>
            Bảng So Sánh Tổng Hợp
          </h2>
          <p className={`text-xs ${t.textMuted}`}>
            Đồng bộ đầy đủ Loss 2 stage, Điểm NLP và Benchmark trong 1 ảnh
          </p>
        </div>

        <button
          onClick={handleExportSnapshot}
          disabled={exporting}
          className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-sm shrink-0 ${
            exportSuccess
              ? 'bg-emerald-600 text-white'
              : exporting
              ? 'bg-sky-600/50 text-sky-100 cursor-wait'
              : 'bg-sky-600 hover:bg-sky-700 text-white active:scale-95'
          }`}
        >
          {exporting ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span>Đang xuất ảnh HD...</span>
            </>
          ) : exportSuccess ? (
            <>
              <Check className="w-4 h-4" />
              <span>Đã lưu ảnh thành công!</span>
            </>
          ) : (
            <>
              <Camera className="w-4 h-4" />
              <span>📸 Xuất Ảnh Báo Cáo (HD PNG)</span>
            </>
          )}
        </button>
      </div>

      {/* KHUNG SO SÁNH CHÍNH (ĐƯỢC CHỤP RA ẢNH) */}
      <div
        ref={boardRef}
        id="composite-comparison-board"
        className={`space-y-5 p-4 md:p-6 rounded-3xl ${t.cardBg} border ${t.cardBorder} ${t.cardShadow}`}
      >
        {/* Banner trong ảnh */}
        <div className={`p-3.5 rounded-2xl border ${t.subtleBorder} ${t.subtleBg} flex flex-wrap items-center justify-between gap-2`}>
          <div>
            <h1 className={`text-lg font-extrabold ${t.textPrimary}`}>
              EM_VLM4AD Model Benchmark & Training Visualizer
            </h1>
            <p className={`text-xs ${t.textMuted}`}>
              Phân tích đối đầu mô hình xe tự hành Vision-Language • GPU: Tesla T4
            </p>
          </div>
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className={`px-2 py-0.5 rounded border ${t.subtleBorder} ${t.cardBg} ${t.textSecondary}`}>
              Đang so sánh: <strong className="text-sky-500">{selectedModels.length > 0 ? `${selectedModels.length} / ${allModels.length}` : allModels.length}</strong> mô hình
            </span>
            <span className={`px-2 py-0.5 rounded border ${t.subtleBorder} ${t.cardBg} ${t.textSecondary}`}>
              Merge: <strong className="text-emerald-500">Deduplicated</strong>
            </span>
          </div>
        </div>

        {/* 1. Loss Visualizer (2 stage Align & Finetune riêng biệt) */}
        <LossVisualizer
          allModels={allModels}
          selectedModels={selectedModels}
          onToggleModel={onToggleModel}
          lossData={lossData}
          loading={loading}
          onSelectModelDetail={onSelectModelDetail}
          currentTheme={currentTheme}
        />

        {/* 2. Trade-off Scatter Plot */}
        <TradeoffScatter
          tradeoffData={tradeoffData}
          selectedModels={selectedModels}
          onSelectModelDetail={onSelectModelDetail}
          currentTheme={currentTheme}
        />

        {/* 3. Metrics Table */}
        <MetricsTable
          metrics={metricsData}
          selectedModels={selectedModels}
          onSelectModel={onSelectModelDetail}
          activeModelId={activeModelId}
          currentTheme={currentTheme}
        />

        {/* 4. Benchmark Table */}
        <BenchmarkTable
          benchmarks={benchmarkData}
          selectedModels={selectedModels}
          onSelectModel={onSelectModelDetail}
          activeModelId={activeModelId}
          currentTheme={currentTheme}
        />

        {/* Watermark báo cáo */}
        <div className={`pt-2 text-center text-[10px] ${t.textMuted} font-mono flex items-center justify-between border-t ${t.subtleBorder}`}>
          <span>EM_VLM4AD Visualizer Report</span>
          <span>Overall = (BLEU-4 + METEOR + ROUGE_L + CIDEr/10) / 4</span>
        </div>
      </div>
    </div>
  );
}
