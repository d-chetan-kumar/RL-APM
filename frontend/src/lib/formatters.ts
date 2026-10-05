const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 2,
});

export function formatCurrency(value: number | null | undefined): string {
  const numericValue = typeof value === 'number' && Number.isFinite(value) ? value : 0;
  return currencyFormatter.format(numericValue);
}

export function formatPercent(
  value: number | null | undefined,
  includeSign = true,
  fractionDigits = 2,
): string {
  const numericValue = typeof value === 'number' && Number.isFinite(value) ? value : 0;
  const formatted = `${(numericValue * 100).toFixed(fractionDigits)}%`;
  return includeSign && numericValue > 0 ? `+${formatted}` : formatted;
}

export function formatRatio(value: number | null | undefined): string {
  const numericValue = Number.isFinite(value) ? Number(value) : 0;
  return numericValue.toFixed(2);
}

export function formatNumber(value: number | null | undefined): string {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(
    Number.isFinite(value) ? Number(value) : 0,
  );
}

export function formatDate(value: string | Date): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }).format(new Date(value));
}

export function formatCompactNumber(value: number | null | undefined): string {
  const numericValue = typeof value === 'number' && Number.isFinite(value) ? value : 0;
  return new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(numericValue);
}

export function formatCompactCurrency(value: number | null | undefined): string {
  const numericValue = typeof value === 'number' && Number.isFinite(value) ? value : 0;
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(numericValue);
}

export function formatMetric(value: number | null | undefined): string {
  return Number.isFinite(value) ? Number(value).toFixed(2) : '—';
}

export function formatTimestamp(value: string | Date): string {
  return new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value));
}
