import React, { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { Activity, Layers, Columns2, Eye } from 'lucide-react';
import { THEMES } from '../theme';

const MODEL_COLORS = [
  '#0284c7', // Sky Blue 600
  '#e11d48', // Rose 600
  '#059669', // Emerald 600
  '#7c3aed', // Purple 600
  '#d97706', // Amber 600
  '#0891b2', // Cyan 600
];

export default function LossVisualizer({
  allModels = [],
  selectedModels = [],
  onToggleModel,
  lossData = {},
  loading = false,
  onSelectModelDetail,
  currentTheme
}) {
  const t = currentTheme || THEMES.light;
  const isLight = t.id === 'light' || t.id === 'paper';

  // Chế độ xem: 'split' (2 cột riêng biệt) | 'finetune' | 'align' | 'combined'
  const [stageMode, setStageMode] = useState('split');
  const [scaleType, setScaleType] = useState('log'); // 'linear' | 'log'
  const [showValLoss, setShowValLoss] = useState(true);

  // Tạo option cho ECharts - HOÀN TOÀN KHÔNG DÙNG TITLE TRONG CANVAS ĐỂ TRÁNH ĐÈ CHỮ
  const buildStageChartOption = (stageFilter) => {
    const series = [];
    const legendData = [];

    selectedModels.forEach((modelId, idx) => {
      const modelLog = lossData[modelId];
      if (!modelLog || !modelLog.steps) return;

      const baseColor = MODEL_COLORS[idx % MODEL_COLORS.length];

      // Lọc steps theo stageFilter
      let filteredSteps = [];
      if (stageFilter === 'all') {
        filteredSteps = modelLog.steps;
      } else {
        filteredSteps = modelLog.steps.filter(s => s.stage === stageFilter);
      }

      if (filteredSteps.length === 0) return;

      const dataPoints = filteredSteps
        .filter(s => s.loss !== null && s.loss !== undefined && (scaleType !== 'log' || s.loss > 0))
        .map(s => [s.step, s.loss, s]);

      legendData.push(modelId);
      series.push({
        name: modelId,
        type: 'line',
        data: dataPoints,
        smooth: 0.15,
        showSymbol: false,
        z: 2, // Đường loss ở lớp dưới
        lineStyle: {
          color: baseColor,
          width: 2.2,
          type: 'solid'
        },
        itemStyle: { color: baseColor }
      });

      // Validation points (ĐƯỢC ĐẶT LÊN LỚP TRÊN CÙNG Z: 10 ĐỂ KHÔNG BỊ LINE LOSS ĐÈ)
      if (showValLoss && modelLog.validations) {
        let filteredVals = modelLog.validations;
        if (stageFilter !== 'all') {
          filteredVals = modelLog.validations.filter(v => v.stage === stageFilter);
        }
        const valPoints = filteredVals
          .filter(v => v.val_loss !== null && (scaleType !== 'log' || v.val_loss > 0))
          .map(v => [v.step, v.val_loss, v]);

        if (valPoints.length > 0) {
          const valName = `${modelId} (Val)`;
          legendData.push(valName);
          series.push({
            name: valName,
            type: 'scatter',
            symbol: 'diamond',
            symbolSize: 11, // Tăng kích thước kim cương rõ nét hơn
            z: 10, // Ưu tiên hiển thị đè lên trên tất cả các đường line
            data: valPoints,
            itemStyle: {
              color: isLight ? '#ffffff' : '#0f172a',
              borderColor: baseColor,
              borderWidth: 2.5,
              shadowBlur: 10,
              shadowColor: baseColor
            },
            emphasis: {
              scale: 1.4,
              itemStyle: {
                shadowBlur: 16,
                borderWidth: 3.5
              }
            }
          });
        }
      }
    });

    return {
      backgroundColor: 'transparent',
      // Legend đặt gọn ở giữa trên cùng với type: 'scroll' để không bao giờ bị tràn hay đè chữ
      legend: {
        type: 'scroll',
        orient: 'horizontal',
        top: 4,
        left: 'center',
        textStyle: { color: t.chart.text, fontSize: 11 },
        itemWidth: 14,
        itemHeight: 8,
        pageIconColor: t.chart.text,
        pageTextStyle: { color: t.chart.text, fontSize: 10 }
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: t.chart.tooltipBg,
        borderColor: t.chart.tooltipBorder,
        borderWidth: 1,
        textStyle: { color: t.chart.tooltipText, fontSize: 11 },
        formatter: (params) => {
          if (!params || !params.length) return '';
          let header = `<div class="font-bold border-b pb-1 mb-1" style="border-color:${t.chart.border}">
            Step: ${params[0].value[0]?.toLocaleString()}
          </div>`;
          let items = params.map(p => {
            const lossVal = typeof p.value[1] === 'number' ? p.value[1].toFixed(4) : p.value[1];
            return `
              <div class="flex items-center justify-between gap-3 py-0.5 text-xs font-mono">
                <span class="flex items-center gap-1.5 font-sans">
                  <span class="w-2 h-2 rounded-full" style="background:${p.color}"></span>
                  <span>${p.seriesName}</span>
                </span>
                <strong>${lossVal}</strong>
              </div>
            `;
          }).join('');
          return header + items;
        }
      },
      // Grid top: 38 tạo khoảng trống 34px an toàn bên dưới legend
      grid: {
        top: 38,
        left: 45,
        right: 20,
        bottom: 30,
        containLabel: true
      },
      xAxis: {
        type: 'value',
        name: 'Step',
        nameLocation: 'middle',
        nameGap: 22,
        nameTextStyle: { color: t.chart.text, fontSize: 10 },
        splitLine: { lineStyle: { color: t.chart.split, type: 'dashed' } },
        axisLine: { lineStyle: { color: t.chart.border } },
        axisLabel: {
          color: t.chart.text,
          fontSize: 10,
          formatter: val => (val >= 1000 ? `${val / 1000}k` : val)
        }
      },
      yAxis: {
        type: scaleType === 'log' ? 'log' : 'value',
        logBase: 10,
        scale: true,
        name: 'Loss',
        nameLocation: 'end',
        nameGap: 8,
        nameTextStyle: { color: t.chart.text, fontSize: 10, align: 'right' },
        splitLine: { lineStyle: { color: t.chart.split } },
        axisLine: { lineStyle: { color: t.chart.border } },
        axisLabel: { color: t.chart.text, fontSize: 10 }
      },
      series: series
    };
  };

  const alignOption = useMemo(() => buildStageChartOption('align'), [selectedModels, lossData, scaleType, showValLoss, t]);
  const finetuneOption = useMemo(() => buildStageChartOption('finetune'), [selectedModels, lossData, scaleType, showValLoss, t]);
  const combinedOption = useMemo(() => buildStageChartOption('all'), [selectedModels, lossData, scaleType, showValLoss, t]);

  const onEvents = {
    click: (params) => {
      if (params.seriesName && onSelectModelDetail) {
        const rawName = params.seriesName.split(' ')[0];
        onSelectModelDetail(rawName);
      }
    }
  };

  return (
    <div className={`${t.cardBg} border ${t.cardBorder} rounded-2xl p-5 ${t.cardShadow} transition`}>
      {/* Header điều khiển trên cùng */}
      <div className={`flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b ${t.cardBorder}`}>
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-sky-500/10 text-sky-500">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <h3 className={`text-base font-bold ${t.textPrimary}`}>
              Đường Cong Loss
            </h3>
            <p className={`text-xs ${t.textMuted}`}>
              Tách biệt Align & Finetune • Max 5 mô hình • Nhấp vào line để soi chi tiết
            </p>
          </div>
        </div>

        {/* Nút tùy chọn hiển thị */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Chế độ Stage */}
          <div className={`p-1 rounded-lg border ${t.subtleBorder} ${t.subtleBg} flex items-center text-xs`}>
            <button
              onClick={() => setStageMode('split')}
              className={`px-2 py-1 rounded transition flex items-center gap-1 font-semibold ${
                stageMode === 'split' ? 'bg-sky-500 text-white shadow-xs' : `${t.textSecondary} hover:${t.textPrimary}`
              }`}
              title="Hiển thị 2 stage ở 2 cột riêng biệt"
            >
              <Columns2 className="w-3.5 h-3.5" />
              <span>Tách 2 Cột</span>
            </button>
            <button
              onClick={() => setStageMode('finetune')}
              className={`px-2 py-1 rounded transition font-medium ${
                stageMode === 'finetune' ? 'bg-sky-500 text-white shadow-xs' : `${t.textSecondary} hover:${t.textPrimary}`
              }`}
            >
              Chỉ Finetune
            </button>
            <button
              onClick={() => setStageMode('align')}
              className={`px-2 py-1 rounded transition font-medium ${
                stageMode === 'align' ? 'bg-sky-500 text-white shadow-xs' : `${t.textSecondary} hover:${t.textPrimary}`
              }`}
            >
              Chỉ Align
            </button>
            <button
              onClick={() => setStageMode('combined')}
              className={`px-2 py-1 rounded transition font-medium ${
                stageMode === 'combined' ? 'bg-sky-500 text-white shadow-xs' : `${t.textSecondary} hover:${t.textPrimary}`
              }`}
            >
              Ghép Liền
            </button>
          </div>

          {/* Scale Type */}
          <div className={`p-1 rounded-lg border ${t.subtleBorder} ${t.subtleBg} flex items-center text-xs`}>
            <button
              onClick={() => setScaleType('log')}
              className={`px-2.5 py-1 rounded transition font-medium ${
                scaleType === 'log' ? 'bg-indigo-500 text-white shadow-xs' : `${t.textSecondary} hover:${t.textPrimary}`
              }`}
            >
              Log Scale
            </button>
            <button
              onClick={() => setScaleType('linear')}
              className={`px-2.5 py-1 rounded transition font-medium ${
                scaleType === 'linear' ? 'bg-indigo-500 text-white shadow-xs' : `${t.textSecondary} hover:${t.textPrimary}`
              }`}
            >
              Linear
            </button>
          </div>

          {/* Val Points Toggle */}
          <button
            onClick={() => setShowValLoss(!showValLoss)}
            className={`px-2.5 py-1.5 rounded-lg text-xs font-medium border transition flex items-center gap-1 ${
              showValLoss
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-600 dark:text-emerald-400'
                : `${t.subtleBg} ${t.subtleBorder} ${t.textMuted}`
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>Val (♦)</span>
          </button>
        </div>
      </div>

      {/* Model Selector Bar */}
      <div className={`flex flex-wrap items-center gap-2 mb-4 p-2.5 rounded-xl border ${t.subtleBorder} ${t.subtleBg}`}>
        <span className={`text-xs font-semibold ${t.textSecondary} flex items-center gap-1.5 mr-1`}>
          <Layers className="w-3.5 h-3.5 text-sky-500" />
          <span>Mô hình hiển thị:</span>
        </span>
        {allModels.map((m, idx) => {
          const isSelected = selectedModels.includes(m.model_id);
          const color = MODEL_COLORS[idx % MODEL_COLORS.length];
          const canSelect = isSelected || selectedModels.length < 5;

          return (
            <button
              key={m.model_id}
              onClick={() => canSelect && onToggleModel(m.model_id)}
              disabled={!canSelect}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition flex items-center gap-1.5 ${
                isSelected
                  ? isLight
                    ? 'bg-white border-slate-300 text-slate-900 shadow-xs'
                    : 'bg-slate-800 border-slate-600 text-white shadow-xs'
                  : canSelect
                  ? `${t.cardBg} ${t.subtleBorder} ${t.textSecondary} hover:${t.textPrimary}`
                  : 'opacity-40 cursor-not-allowed'
              }`}
            >
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{
                  backgroundColor: isSelected ? color : '#94a3b8',
                  boxShadow: isSelected ? `0 0 5px ${color}` : 'none'
                }}
              />
              <span>{m.model_id}</span>
              {isSelected && <span className="text-[10px] opacity-70">✓</span>}
            </button>
          );
        })}
      </div>

      {/* VÙNG BIỂU ĐỒ - HOÀN TOÀN KHÔNG BỊ CHÈN CHỮ VÀO NHAU */}
      {stageMode === 'split' ? (
        /* CHẾ ĐỘ TÁCH 2 CỘT: TIÊU ĐỀ NẰM TRONG HEADER HTML ĐỘC LẬP TRÊN TỪNG KHUNG */
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Cột 1: Stage 1 Align */}
          <div className={`p-4 rounded-xl border ${t.subtleBorder} ${t.subtleBg}`}>
            {/* Header HTML riêng của Stage 1 (Không bao giờ bị chèn vào biểu đồ) */}
            <div className={`flex items-center justify-between pb-2 mb-2 border-b ${t.subtleBorder}`}>
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-sky-500 shadow-xs"></span>
                <span className={`font-bold text-sm ${t.textPrimary}`}>Stage 1: Align</span>
              </div>
              <span className={`text-xs font-medium ${t.textMuted}`}>Căn chỉnh kiến trúc</span>
            </div>

            {/* Canvas ECharts riêng */}
            <div className="h-[340px]">
              <ReactECharts
                option={alignOption}
                onEvents={onEvents}
                style={{ height: '100%', width: '100%' }}
                opts={{ renderer: 'canvas' }}
              />
            </div>
            <div className={`text-[11px] ${t.textMuted} text-center mt-2`}>
              Tiến trình huấn luyện khởi tạo căn chỉnh (Stage 1)
            </div>
          </div>

          {/* Cột 2: Stage 2 Finetune */}
          <div className={`p-4 rounded-xl border ${t.subtleBorder} ${t.subtleBg}`}>
            {/* Header HTML riêng của Stage 2 (Không bao giờ bị chèn vào biểu đồ) */}
            <div className={`flex items-center justify-between pb-2 mb-2 border-b ${t.subtleBorder}`}>
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500 shadow-xs"></span>
                <span className={`font-bold text-sm ${t.textPrimary}`}>Stage 2: Finetune</span>
              </div>
              <span className={`text-xs font-medium ${t.textMuted}`}>Huấn luyện tinh chỉnh</span>
            </div>

            {/* Canvas ECharts riêng */}
            <div className="h-[340px]">
              <ReactECharts
                option={finetuneOption}
                onEvents={onEvents}
                style={{ height: '100%', width: '100%' }}
                opts={{ renderer: 'canvas' }}
              />
            </div>
            <div className={`text-[11px] ${t.textMuted} text-center mt-2`}>
              Tiến trình huấn luyện tinh chỉnh toàn diện (Stage 2)
            </div>
          </div>
        </div>
      ) : (
        /* CHẾ ĐỘ XEM ĐƠN LẺ TOÀN MÀN HÌNH */
        <div className={`p-4 rounded-xl border ${t.subtleBorder} ${t.subtleBg}`}>
          <div className={`flex items-center justify-between pb-2 mb-2 border-b ${t.subtleBorder}`}>
            <span className={`font-bold text-sm ${t.textPrimary}`}>
              {stageMode === 'align'
                ? 'Stage 1: Align (Khởi Tạo & Căn Chỉnh)'
                : stageMode === 'finetune'
                ? 'Stage 2: Finetune (Huấn Luyện Tinh Chỉnh)'
                : 'Cả 2 Giai Đoạn (Ghép Nối Chuỗi)'}
            </span>
          </div>
          <div className="h-[380px]">
            <ReactECharts
              option={
                stageMode === 'align'
                  ? alignOption
                  : stageMode === 'finetune'
                  ? finetuneOption
                  : combinedOption
              }
              onEvents={onEvents}
              style={{ height: '100%', width: '100%' }}
              opts={{ renderer: 'canvas' }}
            />
          </div>
        </div>
      )}

      {/* Footer chú thích */}
      <div className={`mt-3 pt-2 border-t ${t.subtleBorder} flex flex-wrap items-center justify-between text-[11px] ${t.textMuted}`}>
        <div className="flex items-center gap-3">
          <span>• Line: <strong>Train Loss</strong></span>
          <span>• Điểm ♦: <strong>Validation Loss</strong></span>
        </div>
        <div className="italic">
          * Cuộn chuột trên biểu đồ để phóng to thu nhỏ (Zoom)
        </div>
      </div>
    </div>
  );
}
