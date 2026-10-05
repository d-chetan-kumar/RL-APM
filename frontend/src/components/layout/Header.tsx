import React from 'react';
import { RefreshCw, Menu, Calendar } from 'lucide-react';
import { Button } from '../ui/Button';

interface HeaderProps {
  title: string;
  subtitle: string;
  isApiConnected: boolean;
  latestDate?: string;
  onRefresh?: () => void;
  isRefreshing?: boolean;
  onOpenMobileNav?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  title,
  subtitle,
  isApiConnected,
  latestDate = 'Dec 31, 2024',
  onRefresh,
  isRefreshing = false,
  onOpenMobileNav,
}) => {
  return (
    <header className="h-16 bg-white border-b border-slate-200/80 px-6 flex items-center justify-between sticky top-0 z-20">
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenMobileNav}
          className="lg:hidden p-2 rounded-lg text-slate-500 hover:bg-slate-100 focus:outline-none"
          aria-label="Open navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div>
          <h1 className="text-base font-semibold text-slate-900 leading-none tracking-tight">
            {title}
          </h1>
          <p className="text-xs text-slate-500 mt-1 leading-none">
            {subtitle}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* API Connection Indicator */}
        <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-50 border border-slate-200/70 text-xs">
          <span
            className={`w-2 h-2 rounded-full ${
              isApiConnected ? 'bg-emerald-500' : 'bg-rose-500'
            }`}
          />
          <span className="text-[11px] font-medium text-slate-600">
            {isApiConnected ? 'API Connected' : 'API Offline'}
          </span>
        </div>

        {/* Latest Data Date */}
        {latestDate && (
          <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-50 border border-slate-200/70 text-xs">
            <Calendar className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-[11px] text-slate-500">Data:</span>
            <span className="text-[11px] font-medium text-slate-700">{latestDate}</span>
          </div>
        )}

        {/* Refresh Action */}
        {onRefresh && (
          <Button
            variant="outline"
            size="sm"
            onClick={onRefresh}
            isLoading={isRefreshing}
            icon={<RefreshCw className={`w-3.5 h-3.5 text-slate-500 ${isRefreshing ? 'animate-spin' : ''}`} />}
            className="h-8 text-xs font-medium text-slate-700"
          >
            <span className="hidden sm:inline">Refresh</span>
          </Button>
        )}
      </div>
    </header>
  );
};
