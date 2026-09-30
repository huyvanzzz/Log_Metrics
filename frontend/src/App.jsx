import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import CompositeComparisonBoard from './components/CompositeComparisonBoard';
import ModelInspectorModal from './components/ModelInspectorModal';
import { Award, Zap, Activity } from 'lucide-react';
import { THEMES } from './theme';

export default function App() {
  const [currentThemeId, setCurrentThemeId] = useState('light'); // Mặc định Sáng (dễ nhìn, tương phản cao)
  const currentTheme = THEMES[currentThemeId] || THEMES.light;

  const [allModels, setAllModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [lossData, setLossData] = useState({});
  const [metricsData, setMetricsData] = useState([]);
  const [benchmarkData, setBenchmarkData] = useState([]);
  const [tradeoffData, setTradeoffData] = useState([]);

  const [loading, setLoading] = useState(true);
  const [loadingLoss, setLoadingLoss] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const [activeModelId, setActiveModelId] = useState(null);
  const [inspectModalOpen, setInspectModalOpen] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      setRefreshing(true);
      const [modelsRes, metricsRes, benchRes, tradeoffRes] = await Promise.all([
        fetch('/api/models').then(r => r.json()),
        fetch('/api/metrics').then(r => r.json()),
        fetch('/api/benchmark').then(r => r.json()),
        fetch('/api/tradeoff').then(r => r.json()),
      ]);

      setAllModels(modelsRes);
      setMetricsData(metricsRes);
      setBenchmarkData(benchRes);
      setTradeoffData(tradeoffRes);

      const initialSelected = modelsRes.slice(0, 5).map(m => m.model_id);
      setSelectedModels(prev => (prev.length > 0 ? prev : initialSelected));
    } catch (err) {
      console.error('Lỗi khi fetch dữ liệu:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    if (selectedModels.length === 0) return;

    let isMounted = true;
    const fetchLoss = async () => {
      setLoadingLoss(true);
      try {
        const query = selectedModels.join(',');
        const res = await fetch(`/api/loss?models=${encodeURIComponent(query)}`);
        const data = await res.json();
        if (isMounted) setLossData(data);
      } catch (err) {
        console.error('Lỗi khi fetch loss:', err);
      } finally {
        if (isMounted) setLoadingLoss(false);
      }
    };

    fetchLoss();
    return () => { isMounted = false; };
  }, [selectedModels]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleToggleModel = (modelId) => {
    setSelectedModels(prev => {
      if (prev.includes(modelId)) {
        return prev.filter(id => id !== modelId);
      } else {
        if (prev.length >= 5) return prev;
        return [...prev, modelId];
      }
    });
  };

  const handleSelectModelDetail = (modelId) => {
    setActiveModelId(modelId);
    setInspectModalOpen(true);
  };

  const activeDetail = React.useMemo(() => {
    if (!activeModelId) return null;
    const m = metricsData.find(x => x.model_id === activeModelId);
    const b = benchmarkData.find(x => x.model_id === activeModelId);
    const l = lossData[activeModelId];
    return { metrics: m, benchmark: b, loss: l };
  }, [activeModelId, metricsData, benchmarkData, lossData]);

  const topModel = React.useMemo(() => {
    if (!metricsData.length) return null;
    return [...metricsData].sort((a, b) => (b.Overall || 0) - (a.Overall || 0))[0];
  }, [metricsData]);

  return (
    <div className={`min-h-screen ${currentTheme.appBg} transition-colors duration-200 flex flex-col antialiased font-sans`}>
      {/* Header */}
      <Header
        onRefresh={fetchData}
        refreshing={refreshing}
        modelsCount={allModels.length}
        currentTheme={currentTheme}
        onSelectTheme={setCurrentThemeId}
      />

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 lg:px-8 py-5 space-y-5">
        {/* Highlight Stats Bar gọn gàng */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div className={`p-3.5 rounded-2xl ${currentTheme.cardBg} border ${currentTheme.cardBorder} ${currentTheme.cardShadow} flex items-center gap-3`}>
            <div className="p-2 rounded-xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <div className={`text-[10px] font-bold uppercase tracking-wider ${currentTheme.textMuted}`}>
                Top Điểm Overall
              </div>
              <div className={`text-sm font-extrabold ${currentTheme.textPrimary} mt-0.5`}>
                {topModel ? `${topModel.model_id} (${topModel.Overall?.toFixed(2)} pts)` : 'Đang tính...'}
              </div>
            </div>
          </div>

          <div className={`p-3.5 rounded-2xl ${currentTheme.cardBg} border ${currentTheme.cardBorder} ${currentTheme.cardShadow} flex items-center gap-3`}>
            <div className="p-2 rounded-xl bg-amber-500/10 text-amber-600 dark:text-amber-400">
              <Zap className="w-5 h-5" />
            </div>
            <div>
              <div className={`text-[10px] font-bold uppercase tracking-wider ${currentTheme.textMuted}`}>
                Tốc Độ Nhanh Nhất
              </div>
              <div className={`text-sm font-extrabold ${currentTheme.textPrimary} mt-0.5`}>
                10.51 ms/token (Adapter)
              </div>
            </div>
          </div>

          <div className={`p-3.5 rounded-2xl ${currentTheme.cardBg} border ${currentTheme.cardBorder} ${currentTheme.cardShadow} flex items-center gap-3`}>
            <div className="p-2 rounded-xl bg-sky-500/10 text-sky-600 dark:text-sky-400">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <div className={`text-[10px] font-bold uppercase tracking-wider ${currentTheme.textMuted}`}>
                Gộp Log Huấn Luyện
              </div>
              <div className={`text-sm font-extrabold ${currentTheme.textPrimary} mt-0.5`}>
                Deduplication Active (Lọc trùng lặp)
              </div>
            </div>
          </div>
        </div>

        {/* Khung So Sánh Tổng Hợp */}
        <CompositeComparisonBoard
          allModels={allModels}
          selectedModels={selectedModels}
          onToggleModel={handleToggleModel}
          lossData={lossData}
          metricsData={metricsData}
          benchmarkData={benchmarkData}
          tradeoffData={tradeoffData}
          loading={loadingLoss}
          onSelectModelDetail={handleSelectModelDetail}
          activeModelId={activeModelId}
          currentTheme={currentTheme}
        />
      </main>

      {/* Model Inspector Modal */}
      {inspectModalOpen && (
        <ModelInspectorModal
          modelId={activeModelId}
          modelDetail={activeDetail}
          onClose={() => setInspectModalOpen(false)}
          currentTheme={currentTheme}
        />
      )}

      {/* Footer */}
      <footer className={`py-4 text-center text-xs ${currentTheme.textMuted} border-t ${currentTheme.cardBorder}`}>
        <div>EM_VLM4AD Visualizer Dashboard • Triển khai 100% trên ổ D:</div>
      </footer>
    </div>
  );
}
