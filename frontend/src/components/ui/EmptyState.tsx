import React from 'react';
import { Database } from 'lucide-react';

interface EmptyStateProps {
  title?: string;
  description?: string;
  icon?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No data available',
  description = 'There are currently no records or results to display for this selection.',
  icon,
}) => {
  return (
    <div className="bg-white rounded-xl border border-slate-200/80 p-8 flex flex-col items-center justify-center text-center shadow-card">
      <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center text-slate-500 mb-3 border border-slate-200">
        {icon || <Database className="w-5 h-5" />}
      </div>
      <h4 className="text-sm font-semibold text-slate-800 mb-1">{title}</h4>
      <p className="text-xs text-slate-500 max-w-sm">{description}</p>
    </div>
  );
};
