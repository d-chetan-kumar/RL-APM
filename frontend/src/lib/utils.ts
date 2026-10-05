import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export const ASSET_COLORS: Record<string, string> = {
  AAPL: '#6366f1',
  MSFT: '#0ea5e9',
  GOOGL: '#f59e0b',
  AMZN: '#f97316',
  NVDA: '#10b981',
};

export const MODEL_COLORS = {
  ddpg: '#6366f1',
  benchmark: '#f59e0b',
} as const;
