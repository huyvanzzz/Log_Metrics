import React, { useState, useMemo, useRef, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import { Activity, Layers, Columns2, Eye, Folder, ChevronDown, Check, Search, X } from 'lucide-react';
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

  // State dropdown chọn mô hình theo thư mục
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const dropdownRef = useRef(null);

  // Đóng dropdown khi click ra ngoài
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    }
    if (isDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isDropdownOpen]);

  // Gom nhóm mô hình theo thư mục cha
  const groupedModels = useMemo(() => {
    const groups = {};
    allModels.forEach((m, idx) => {
      let groupName = m.group;
      if (!groupName) {
        if (m.folder_path) {
          const parts = m.folder_path.replace(/\\/g, '/').split('/').filter(Boolean);
          const logIdx = parts.findIndex(p => p.toLowerCase() === 'log');
          if (logIdx >= 0 && parts.length > logIdx + 1) {
            groupName = parts[logIdx + 1];
          }
        }
      }
      if (!groupName) {
        const parts = m.model_id.split('_');
        groupName = parts.length > 1 ? parts[0] : 'Khác';
      }

      let subName = m.sub_name;
      if (!subName) {
        if (m.model_id.startsWith(groupName + '_')) {
          subName = m.model_id.slice(groupName.length + 1);
        } else {
          subName = m.model_id;
        }
      }

      if (!groups[groupName]) {
        groups[groupName] = [];
      }
      groups[groupName].push({
        ...m,
        displayName: subName,
        globalIdx: idx,
        color: MODEL_COLORS[idx % MODEL_COLORS.length]
      });
    });
    return groups;
  }, [allModels]);

  // Lọc mô hình theo ô tìm kiếm
  const filteredGroupedModels = useMemo(() => {
    if (!searchTerm.trim()) return groupedModels;
    const term = searchTerm.toLowerCase();
    const result = {};
    Object.entries(groupedModels).forEach(([group, items]) => {
      const matched = items.filter(
        item =>
          item.model_id.toLowerCase().includes(term) ||
          item.displayName.toLowerCase().includes(term) ||
          group.toLowerCase().includes(term)
      );
      if (matched.length > 0) {
        result[group] = matched;
      }
    });
    return result;
  }, [groupedModels, searchTerm]);

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

      {/* Model Selector Bar with Folder Dropdown */}
      <div className={`relative mb-4 p-2.5 rounded-xl border ${t.subtleBorder} ${t.subtleBg} flex flex-wrap items-center gap-2`}>
        {/* Dropdown Toggle Button */}
        <div className="relative" ref={dropdownRef}>
          <button
            onClick={() => setIsDropdownOpen(prev => !prev)}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold border transition flex items-center gap-2 shadow-xs cursor-pointer ${
              isDropdownOpen
                ? 'bg-sky-500 text-white border-sky-500'
                : `${t.cardBg} ${t.cardBorder} ${t.textPrimary} hover:border-sky-500`
            }`}
          >
            <Folder className="w-3.5 h-3.5 text-sky-400" />
            <span>Chọn mô hình ({selectedModels.length}/5)</span>
            <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-200 ${isDropdownOpen ? 'rotate-180' : ''}`} />
          </button>

          {/* Popover Dropdown Menu từ trên xuống */}
          {isDropdownOpen && (
            <div
              className={`absolute top-full left-0 mt-2 w-80 sm:w-96 rounded-2xl border ${t.cardBorder} ${t.cardBg} ${t.cardShadow} shadow-2xl z-50 p-3.5 animate-in fade-in slide-in-from-top-2 duration-150`}
            >
              {/* Header Dropdown */}
              <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-200 dark:border-slate-800">
                <div className="flex items-center gap-1.5 text-xs font-bold text-sky-500">
                  <Layers className="w-4 h-4" />
                  <span>Chọn mô hình theo folder</span>
                </div>
                <div className="text-[11px] font-mono text-slate-400">
                  <span className={selectedModels.length >= 5 ? 'text-amber-500 font-bold' : 'text-emerald-500 font-bold'}>
                    {selectedModels.length}
                  </span>
                  <span>/5 mô hình</span>
                </div>
              </div>

              {/* Ô tìm kiếm nhanh */}
              <div className="relative mb-2.5">
                <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Tìm kiếm mô hình..."
                  className={`w-full pl-8 pr-7 py-1.5 text-xs rounded-lg border ${t.subtleBorder} ${t.subtleBg} ${t.textPrimary} focus:outline-none focus:ring-1 focus:ring-sky-500`}
                />
                {searchTerm && (
                  <button
                    onClick={() => setSearchTerm('')}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>

              {/* Danh sách nhóm theo folder */}
              <div className="max-h-[300px] overflow-y-auto pr-1 space-y-3 custom-scrollbar">
                {Object.keys(filteredGroupedModels).length === 0 ? (
                  <div className="text-center py-4 text-xs text-slate-400">
                    Không tìm thấy mô hình phù hợp
                  </div>
                ) : (
                  Object.entries(filteredGroupedModels).map(([group, items]) => (
                    <div key={group} className="space-y-1">
                      {/* Tiêu đề folder */}
                      <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-slate-100 dark:bg-slate-800/60 text-[11px] font-bold text-slate-600 dark:text-slate-300">
                        <Folder className="w-3.5 h-3.5 text-sky-500" />
                        <span className="truncate">{group}</span>
                        <span className="text-[10px] text-slate-400 ml-auto font-normal">({items.length})</span>
                      </div>

                      {/* Các mô hình con trong folder */}
                      <div className="space-y-0.5 pl-1">
                        {items.map((m) => {
                          const isSelected = selectedModels.includes(m.model_id);
                          const canSelect = isSelected || selectedModels.length < 5;

                          return (
                            <div
                              key={m.model_id}
                              onClick={() => {
                                if (canSelect) {
                                  onToggleModel(m.model_id);
                                }
                              }}
                              className={`flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs cursor-pointer transition select-none ${
                                isSelected
                                  ? 'bg-sky-500/10 border border-sky-500/30 text-sky-600 dark:text-sky-300 font-medium'
                                  : canSelect
                                  ? `hover:bg-slate-100 dark:hover:bg-slate-800/50 ${t.textSecondary}`
                                  : 'opacity-40 cursor-not-allowed'
                              }`}
                            >
                              <div className="flex items-center gap-2 min-w-0 pr-2">
                                {/* Checkbox trực quan */}
                                <div
                                  className={`w-4 h-4 rounded border flex items-center justify-center shrink-0 transition ${
                                    isSelected
                                      ? 'bg-sky-500 border-sky-500 text-white'
                                      : 'border-slate-300 dark:border-slate-600 bg-transparent'
                                  }`}
                                >
                                  {isSelected && <Check className="w-3 h-3 stroke-[3]" />}
                                </div>

                                {/* Dot màu tương ứng biểu đồ */}
                                <span
                                  className="w-2.5 h-2.5 rounded-full shrink-0"
                                  style={{
                                    backgroundColor: isSelected ? m.color : '#94a3b8',
                                    boxShadow: isSelected ? `0 0 5px ${m.color}` : 'none'
                                  }}
                                />

                                {/* Tên mô hình */}
                                <span className="truncate font-mono" title={m.model_id}>
                                  {m.displayName}
                                </span>
                              </div>

                              {isSelected ? (
                                <span className="text-[10px] font-bold text-sky-500 shrink-0">Đã chọn</span>
                              ) : !canSelect ? (
                                <span className="text-[10px] text-amber-500 shrink-0">Đủ 5</span>
                              ) : null}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))
                )}
              </div>

              {/* Footer Dropdown */}
              <div className="mt-2.5 pt-2 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
                <span>* Tối đa 5 mô hình</span>
                <button
                  onClick={() => setIsDropdownOpen(false)}
                  className="px-2.5 py-1 rounded-md bg-slate-200 dark:bg-slate-800 hover:opacity-80 text-xs font-semibold text-slate-700 dark:text-slate-200 cursor-pointer"
                >
                  Xong
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Các chip mô hình đã được chọn (gọn gàng, có dấu × để gỡ nhanh) */}
        <div className="flex flex-wrap items-center gap-1.5 flex-1 min-w-0">
          {selectedModels.map((modelId) => {
            const idx = allModels.findIndex(m => m.model_id === modelId);
            const color = MODEL_COLORS[(idx >= 0 ? idx : 0) % MODEL_COLORS.length];
            return (
              <span
                key={modelId}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border shadow-2xs transition ${
                  isLight
                    ? 'bg-white border-slate-300 text-slate-800'
                    : 'bg-slate-800 border-slate-600 text-slate-200'
                }`}
              >
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: color, boxShadow: `0 0 4px ${color}` }}
                />
                <span className="truncate max-w-[220px]" title={modelId}>
                  {modelId}
                </span>
                <button
                  onClick={() => onToggleModel(modelId)}
                  className="hover:text-rose-500 text-slate-400 p-0.5 rounded cursor-pointer transition leading-none"
                  title={`Bỏ chọn ${modelId}`}
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            );
          })}
          {selectedModels.length === 0 && (
            <span className="text-xs text-amber-500 italic flex items-center gap-1">
              Chưa chọn mô hình nào. Nhấp vào nút "Chọn mô hình" để tích chọn.
            </span>
          )}
        </div>
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
