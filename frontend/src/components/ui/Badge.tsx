import React from 'react';
import { cn } from '../../lib/utils';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'brand' | 'emerald' | 'rose' | 'amber' | 'slate' | 'outline';
  size?: 'sm' | 'md';
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'brand',
  size = 'md',
  className,
  ...props
}) => {
  const variantStyles = {
    brand: 'bg-indigo-50 text-indigo-700 border-indigo-200/60',
    emerald: 'bg-emerald-50 text-emerald-700 border-emerald-200/60',
    rose: 'bg-rose-50 text-rose-700 border-rose-200/60',
    amber: 'bg-amber-50 text-amber-700 border-amber-200/60',
    slate: 'bg-slate-100 text-slate-700 border-slate-200',
    outline: 'bg-white text-slate-600 border-slate-300',
  };

  const sizeStyles = {
    sm: 'text-[11px] px-2 py-0.5 font-medium',
    md: 'text-xs px-2.5 py-0.5 font-medium',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border tracking-tight',
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
};
