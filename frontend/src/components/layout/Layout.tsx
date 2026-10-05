import React, { useEffect, useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { api } from '../../services/api';
import { X } from 'lucide-react';

const PAGE_INFO: Record<string, { title: string; subtitle: string }> = {
  '/': {
    title: 'Portfolio Intelligence',
    subtitle: 'Reinforcement learning for data-driven equity allocation',
  },
  '/market': {
    title: 'Market Analysis',
    subtitle: 'Cross-asset indicators, technical features, and historical momentum',
  },
  '/portfolio': {
    title: 'AI Portfolio',
    subtitle: 'DDPG model inference from the latest available market state',
  },
  '/performance': {
    title: 'Performance Analysis',
    subtitle: 'Held-out 2024 evaluation of DDPG policy against equal-weight benchmark',
  },
  '/training': {
    title: 'Training Lab',
    subtitle: 'DDPG training dynamics, experience replay, and validation checkpoints',
  },
  '/methodology': {
    title: 'Methodology & Architecture',
    subtitle: 'Markov Decision Process, 50-dimensional observation state, and neural policy update',
  },
  '/system': {
    title: 'System Diagnostics',
    subtitle: 'Live service health, PyTorch model integrity, and endpoint catalogue',
  },
};

export const Layout: React.FC = () => {
  const location = useLocation();
  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);
  const [isApiConnected, setIsApiConnected] = useState(true);
  const [latestDate, setLatestDate] = useState<string>('Dec 31, 2024');
  const [isRefreshing, setIsRefreshing] = useState(false);

  const currentInfo = PAGE_INFO[location.pathname] || {
    title: 'RL-APM Platform',
    subtitle: 'Reinforcement Learning for Portfolio Optimization',
  };

  const checkHealth = async () => {
    try {
      const health = await api.getHealth();
      setIsApiConnected(health.status === 'ok');
    } catch {
      setIsApiConnected(false);
    }
  };

  const fetchLatestDate = async () => {
    try {
      const summary = await api.getMarketSummary();
      if (summary.date) {
        setLatestDate(summary.date);
      }
    } catch {
      // Fallback retained
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await Promise.all([checkHealth(), fetchLatestDate()]);
    // Allow small delay for smooth visual feedback
    setTimeout(() => setIsRefreshing(false), 400);
  };

  useEffect(() => {
    checkHealth();
    fetchLatestDate();
    const timer = setInterval(checkHealth, 30000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col lg:flex-row">
      {/* Desktop Sidebar */}
      <div className="hidden lg:block shrink-0">
        <Sidebar />
      </div>

      {/* Mobile Drawer Overlay */}
      {isMobileNavOpen && (
        <div className="lg:hidden fixed inset-0 z-50 flex">
          <div
            className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs transition-opacity"
            onClick={() => setIsMobileNavOpen(false)}
          />
          <div className="relative w-64 max-w-[80vw] bg-white h-full shadow-2xl flex flex-col z-10">
            <button
              onClick={() => setIsMobileNavOpen(false)}
              className="absolute top-4 right-4 p-2 text-slate-400 hover:text-slate-600"
              aria-label="Close menu"
            >
              <X className="w-5 h-5" />
            </button>
            <Sidebar onCloseMobile={() => setIsMobileNavOpen(false)} />
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-x-hidden">
        <Header
          title={currentInfo.title}
          subtitle={currentInfo.subtitle}
          isApiConnected={isApiConnected}
          latestDate={latestDate}
          onRefresh={handleRefresh}
          isRefreshing={isRefreshing}
          onOpenMobileNav={() => setIsMobileNavOpen(true)}
        />
        <main className="flex-1 p-4 md:p-6 lg:p-8 max-w-7xl mx-auto w-full">
          <Outlet />
        </main>
      </div>
    </div>
  );
};
