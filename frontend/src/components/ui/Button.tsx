import React from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '../../lib/utils';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  icon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  icon,
  className,
  disabled,
  ...props
}) => {
  const variantStyles = {
    primary:
      'bg-indigo-600 text-white hover:bg-indigo-700 active:bg-indigo-800 shadow-sm border border-indigo-600',
    secondary:
      'bg-slate-100 text-slate-800 hover:bg-slate-200 border border-slate-200',
    outline:
      'bg-white text-slate-700 hover:bg-slate-50 border border-slate-300 shadow-sm',
    ghost:
      'bg-transparent text-slate-600 hover:bg-slate-100 hover:text-slate-900 border border-transparent',
  };

  const sizeStyles = {
    sm: 'text-xs px-3 py-1.5 rounded-lg font-medium',
    md: 'text-sm px-4 py-2 rounded-lg font-medium',
    lg: 'text-base px-5 py-2.5 rounded-xl font-medium',
  };

  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer',
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <Loader2 className="w-4 h-4 animate-spin text-current" />
      ) : (
        icon && <span className="w-4 h-4 flex items-center justify-center">{icon}</span>
      )}
      <span>{children}</span>
    </button>
  );
};
