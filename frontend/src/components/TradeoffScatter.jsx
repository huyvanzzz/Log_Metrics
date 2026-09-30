import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { Target, Zap, AlertCircle } from 'lucide-react';
import { THEMES } from '../theme';

const SCATTER_COLORS = [
  '#0284c7', // Sky Blue 600
  '#e11d48', // Rose 600
  '#059669', // Emerald 600
  '#7c3aed', // Purple 600
  '#d97706', // Amber 600
  '#0891b2', // Cyan 600
];

export default function TradeoffScatter({ tradeoffData = [], onSelectModelDetail, currentTheme }) {
  const t = currentTheme || THEMES.light;
  const isLight = t.id === 'light' || t.id === 'paper';

  const { validPoints, missingModels } = useMemo(() => {
    const valid = [];
    const missing = [];
    tradeoffData.forEach((p, idx) => {
      if (p.x_latency_per_token !== null && p.y_overall !== null) {
        valid.push({
          ...p,
          color: SCATTER_COLORS[idx % SCATTER_COLORS.length]
        });
      } else {
        missing.push(p);
      }
    });
    return { validPoints: valid, missingModels: missing };
  }, [tradeoffData]);

  const chartOption = useMemo(() => {
    const series = validPoints.map((p) => {
      return {
        name: p.model_id,
        type: 'scatter',
        symbolSize: 20,
        data: [[p.x_latency_per_token, p.y_overall, p]],
        itemStyle: {
          color: p.color,
          shadowBlur: 10,
          shadowColor: p.color,
          borderColor: isLight ? '#ffffff' : '#0f172a',
          borderWidth: 2
        },
        label: {
          show: true,
          formatter: (params) => {
            const raw = params.value[2];
            return `{title|${raw.model_id}}\n{sub|${raw.y_overall?.toFixed(2)} pts • ${raw.x_latency_per_token?.toFixed(1)} ms}`;
          },
          rich: {
            title: {
              color: isLight ? '#0f172a' : '#f8fafc',
              fontWeight: 'bold',
              fontSize: 11,
              padding: [0, 0, 2, 0]
            },
            sub: {
              color: isLight ? '#64748b' : '#94a3b8',
              fontSize: 10
            }
          },
          position: 'top',
          distance: 6
        }
      };
    });

    const xVals = validPoints.map(p => p.x_latency_per_token);
    const yVals = validPoints.map(p => p.y_overall);

    const minX = xVals.length ? Math.floor(Math.min(...xVals) * 0.8) : 0;
    const maxX = xVals.length ? Math.ceil(Math.max(...xVals) * 1.2) : 20;
    const minY = yVals.length ? Math.floor(Math.min(...yVals) * 0.9) : 35;
    const maxY = yVals.length ? Math.ceil(Math.max(...yVals) * 1.1) : 55;

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        backgroundColor: t.chart.tooltipBg,
        borderColor: t.chart.tooltipBorder,
        borderWidth: 1,
        textStyle: { color: t.chart.tooltipText, fontSize: 11 },
        formatter: (params) => {
          const item = params.value[2];
          if (!item) return '';
          const m = item.metrics || {};
          return `
            <div class="p-1">
              <div class="font-bold text-xs pb-1 mb-1.5 flex items-center gap-1.5 border-b" style="border-color:${t.chart.border}">
                <span class="w-2.5 h-2.5 rounded-full" style="background:${item.color}"></span>
                <span>${item.model_id}</span>
              </div>
              <div class="grid grid-cols-2 gap-x-3 gap-y-1 text-xs font-mono">
                <span class="font-sans text-slate-400">Độ trễ/token:</span>
                <strong class="text-right">${item.x_latency_per_token?.toFixed(2)} ms</strong>

                <span class="font-sans text-slate-400">Tốc độ sinh:</span>
                <strong class="text-right">${item.tokens_per_second?.toFixed(1)} tk/s</strong>

                <span class="font-sans text-slate-400">Điểm Overall:</span>
                <strong class="text-right text-emerald-500">${item.y_overall?.toFixed(3)}</strong>
              </div>
              <div class="mt-2 pt-1 border-t text-[10px] text-slate-400" style="border-color:${t.chart.border}">
                B4: <strong>${m['BLEU-4'] || '-'}</strong> | METEOR: <strong>${m['METEOR'] || '-'}</strong> | ROUGE: <strong>${m['ROUGE-L'] || '-'}</strong> | CIDEr: <strong>${m['CIDEr'] || '-'}</strong>
              </div>
            </div>
          `;
        }
      },
      grid: {
        top: 50,
        left: 45,
        right: 35,
        bottom: 40,
        containLabel: true
      },
      xAxis: {
        type: 'value',
        name: 'Độ trễ mỗi token (ms) ➔ [Càng nhỏ càng nhanh]',
        nameLocation: 'middle',
        nameGap: 24,
        nameTextStyle: { color: t.chart.text, fontSize: 11 },
        splitLine: { lineStyle: { color: t.chart.split, type: 'dashed' } },
        axisLine: { lineStyle: { color: t.chart.border } },
        axisLabel: { color: t.chart.text, fontSize: 10, formatter: val => `${val}ms` },
        min: minX,
        max: maxX
      },
      yAxis: {
        type: 'value',
        name: 'Điểm Overall ➔ [Càng cao càng tốt]',
        nameTextStyle: { color: t.chart.text, fontSize: 11 },
        splitLine: { lineStyle: { color: t.chart.split } },
        axisLine: { lineStyle: { color: t.chart.border } },
        axisLabel: { color: t.chart.text, fontSize: 10 },
        min: minY,
        max: maxY
      },
      series: series
    };
  }, [validPoints, t]);

  const onEvents = {
    click: (params) => {
      const item = params.value && params.value[2];
      if (item && onSelectModelDetail) {
        onSelectModelDetail(item.model_id);
      }
    }
  };

  return (
    <div className={`${t.cardBg} border ${t.cardBorder} rounded-2xl p-5 ${t.cardShadow} transition`}>
      {/* Header gọn gàng */}
      <div className={`flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b ${t.cardBorder}`}>
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-500">
            <Target className="w-4 h-4" />
          </div>
          <div>
            <h3 className={`text-base font-bold ${t.textPrimary}`}>
              Đánh Đổi Tốc Độ & Điểm Số (Trade-off)
            </h3>
            <p className={`text-xs ${t.textMuted}`}>
              Trục X: ms/token • Trục Y: Overall Score • Mỗi chấm tròn là 1 mô hình
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-600 dark:text-indigo-300 text-xs">
          <Zap className="w-3.5 h-3.5 text-amber-500" />
          <span>Vùng tối ưu: Góc trên bên trái (Nhanh & Điểm cao)</span>
        </div>
      </div>

      {/* Chart container */}
      <div className={`p-3 rounded-xl border ${t.subtleBorder} ${t.subtleBg} h-[340px]`}>
        {validPoints.length === 0 ? (
          <div className="w-full h-full flex flex-col items-center justify-center text-slate-400 gap-1 text-xs">
            <AlertCircle className="w-6 h-6 text-amber-500" />
            <span>Chưa có mô hình nào đủ cả 2 file benchmark và metrics</span>
          </div>
        ) : (
          <ReactECharts
            option={chartOption}
            onEvents={onEvents}
            style={{ height: '100%', width: '100%' }}
            opts={{ renderer: 'canvas' }}
          />
        )}
      </div>

      {missingModels.length > 0 && (
        <div className={`mt-2 text-xs ${t.textMuted} flex items-center gap-1.5`}>
          <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
          <span>{missingModels.map(m => m.model_id).join(', ')}: Chưa hiển thị do chưa có benchmark.json</span>
        </div>
      )}
    </div>
  );
}
