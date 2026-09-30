import React from 'react';
import { Eye, RefreshCw, Sun, Moon, FileText, Sparkles } from 'lucide-react';
import { THEMES } from '../theme';

export default function Header({
  onRefresh,
  refreshing,
  modelsCount,
  currentTheme,
  onSelectTheme
}) {
  const t = currentTheme;
  const isLight = t.id === 'light' || t.id === 'paper';

  const themeOptions = [
    { id: 'light', label: 'Sáng', icon: Sun, desc: 'Clean Light' },
    { id: 'paper', label: 'Giấy', icon: FileText, desc: 'Academic Paper' },
    { id: 'slate', label: 'Tối', icon: Moon, desc: 'Dark Slate' },
    { id: 'midnight', label: 'OLED', icon: Sparkles, desc: 'OLED Black' },
  ];

  return (
    <header className={`sticky top-0 z-40 ${t.cardBg}/95 backdrop-blur-md border-b ${t.cardBorder} px-4 lg:px-8 py-3 transition`}>
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Title */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center text-white shadow-xs">
            <Eye className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <h1 className={`text-base font-extrabold ${t.textPrimary} tracking-tight`}>
                EM_VLM4AD Visualizer
              </h1>
              <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                v1.1
              </span>
            </div>
          </div>
        </div>

        {/* Right Actions: Theme Selector & Refresh */}
        <div className="flex items-center gap-3">
          {/* Theme Selector Buttons */}
          <div className={`flex items-center p-1 rounded-xl border ${t.subtleBorder} ${t.subtleBg}`}>
            <span className={`text-[11px] font-semibold ${t.textMuted} px-2 hidden sm:inline`}>
              Theme:
            </span>
            {themeOptions.map((opt) => {
              const Icon = opt.icon;
              const isActive = t.id === opt.id;
              return (
                <button
                  key={opt.id}
                  onClick={() => onSelectTheme(opt.id)}
                  className={`px-2 py-1 rounded-lg text-xs font-medium transition flex items-center gap-1 ${
                    isActive
                      ? isLight
                        ? 'bg-white text-slate-900 shadow-xs border border-slate-200'
                        : 'bg-slate-800 text-white shadow-xs border border-slate-700'
                      : `${t.textMuted} hover:${t.textPrimary}`
                  }`}
                  title={opt.desc}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span className="text-[11px]">{opt.label}</span>
                </button>
              );
            })}
          </div>

          {/* Model count badge */}
          <div className={`hidden sm:flex items-center gap-1 px-2.5 py-1 rounded-lg border ${t.subtleBorder} ${t.subtleBg} text-xs ${t.textSecondary}`}>
            <span>{modelsCount} mô hình</span>
          </div>

          {/* Refresh button */}
          <button
            onClick={onRefresh}
            disabled={refreshing}
            className={`p-1.5 rounded-lg border ${t.subtleBorder} ${t.subtleBg} hover:opacity-80 transition text-slate-400`}
            title="Làm mới dữ liệu"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin text-sky-500' : ''}`} />
          </button>
        </div>
      </div>
    </header>
  );
}
