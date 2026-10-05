import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  LineChart,
  Brain,
  BarChart3,
  Activity,
  Network,
  ShieldCheck,
  TrendingUp,
} from 'lucide-react';
import { cn } from '../../lib/utils';

interface SidebarProps {
  onCloseMobile?: () => void;
}

const NAV_ITEMS = [
  { name: 'Overview', to: '/', icon: LayoutDashboard },
  { name: 'Market Analysis', to: '/market', icon: LineChart },
  { name: 'AI Portfolio', to: '/portfolio', icon: Brain },
  { name: 'Performance', to: '/performance', icon: BarChart3 },
  { name: 'Training Lab', to: '/training', icon: Activity },
  { name: 'Methodology', to: '/methodology', icon: Network },
  { name: 'System', to: '/system', icon: ShieldCheck },
];

export const Sidebar: React.FC<SidebarProps> = ({ onCloseMobile }) => {
  return (
    <aside className="w-64 bg-white border-r border-slate-200/80 flex flex-col justify-between h-screen sticky top-0 select-none z-30">
      <div>
        {/* Brand Header */}
        <div className="h-16 flex items-center px-6 border-b border-slate-100 gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-600 to-indigo-700 flex items-center justify-center text-white shadow-sm shadow-indigo-200">
            <TrendingUp className="w-5 h-5 stroke-[2.2]" />
          </div>
          <div>
            <span className="font-semibold text-sm tracking-tight text-slate-900 block leading-tight">
              RL-APM
            </span>
            <span className="text-[11px] text-slate-500 font-medium block leading-tight">
              AI Portfolio Research
            </span>
          </div>
        </div>

        {/* Navigation links */}
        <nav className="p-3 space-y-1 mt-2">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                onClick={onCloseMobile}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-xs font-medium transition-all duration-150',
                    isActive
                      ? 'bg-indigo-50/80 text-indigo-700 font-semibold shadow-xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon
                      className={cn(
                        'w-4 h-4 transition-colors',
                        isActive ? 'text-indigo-600' : 'text-slate-400 group-hover:text-slate-600'
                      )}
                    />
                    <span>{item.name}</span>
                  </>
                )}
              </NavLink>
            );
          })}
        </nav>
      </div>

      {/* Footer Info / Badge */}
      <div className="p-4 border-t border-slate-100">
        <div className="bg-slate-50 rounded-lg p-3 border border-slate-200/60">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[11px] font-semibold text-slate-700">DDPG Model Active</span>
          </div>
          <p className="text-[10px] text-slate-500 leading-normal">
            Held-out test validated. Autonomous continuous portfolio allocation.
          </p>
        </div>
      </div>
    </aside>
  );
};
