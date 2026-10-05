import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { Button } from './Button';

interface ErrorCardProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorCard: React.FC<ErrorCardProps> = ({
  title = 'Backend connection error',
  message = 'Failed to communicate with the FastAPI backend service.',
  onRetry,
  className = '',
}) => {
  return (
    <div
      className={`bg-white rounded-xl border border-rose-200/80 p-6 flex flex-col items-center justify-center text-center shadow-card ${className}`}
    >
      <div className="w-10 h-10 rounded-full bg-rose-50 flex items-center justify-center text-rose-600 mb-3 border border-rose-100">
        <AlertCircle className="w-5 h-5" />
      </div>
      <h4 className="text-sm font-semibold text-slate-800 mb-1">{title}</h4>
      <p className="text-xs text-slate-500 max-w-md mb-4">{message}</p>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry} icon={<RefreshCw className="w-3.5 h-3.5" />}>
          Retry Connection
        </Button>
      )}
    </div>
  );
};
