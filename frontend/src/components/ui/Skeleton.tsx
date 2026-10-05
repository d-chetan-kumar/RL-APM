import React from 'react';
import { cn } from '../../lib/utils';

export const Skeleton: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className,
  ...props
}) => {
  return (
    <div
      className={cn('animate-pulse rounded-lg bg-slate-200/70', className)}
      {...props}
    />
  );
};

export const MetricCardSkeleton: React.FC = () => {
  return (
    <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-card space-y-3">
      <div className="flex items-center justify-between">
        <Skeleton className="h-4 w-28" />
        <Skeleton className="h-8 w-8 rounded-lg" />
      </div>
      <Skeleton className="h-7 w-36" />
      <Skeleton className="h-3 w-20" />
    </div>
  );
};

export const ChartSkeleton: React.FC<{ height?: string }> = ({ height = 'h-72' }) => {
  return (
    <div className={cn('w-full rounded-xl bg-white border border-slate-200/80 p-6 flex flex-col justify-between shadow-card', height)}>
      <div className="flex justify-between items-center mb-4">
        <Skeleton className="h-5 w-48" />
        <Skeleton className="h-4 w-24" />
      </div>
      <div className="flex-1 flex items-end gap-2 pt-6">
        {[40, 65, 30, 85, 60, 45, 75, 90, 55, 70].map((h, i) => (
          <Skeleton key={i} className="flex-1 rounded-t" style={{ height: `${h}%` }} />
        ))}
      </div>
    </div>
  );
};

export const TableSkeleton: React.FC<{ rows?: number }> = ({ rows = 5 }) => {
  return (
    <div className="w-full rounded-xl bg-white border border-slate-200/80 p-6 shadow-card space-y-4">
      <div className="flex justify-between items-center pb-2 border-b border-slate-100">
        <Skeleton className="h-5 w-48" />
        <Skeleton className="h-4 w-24" />
      </div>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex items-center justify-between py-2 border-b border-slate-50 last:border-0">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-24" />
        </div>
      ))}
    </div>
  );
};
