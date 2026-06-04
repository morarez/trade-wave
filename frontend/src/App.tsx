import { useMemo, useState } from 'react'

type DataTable = {
  columns: string[]
  index: string[]
  data: Array<Array<string | number | null>>
}

type StrategyResult = {
  summary: DataTable
  stats: DataTable
}

type ApiResult = {
  summary: DataTable
  per_symbol: Record<string, StrategyResult>
}

const DEFAULT_SYMBOLS = 'AAPL, SOFI, GOOG'

function formatCell(value: string | number | null, column?: string, percentPosition: 'prefix' | 'suffix' = 'suffix') {
  if (value === null || value === undefined) return '-'
  const col = (column || '').toLowerCase()

  // Columns that represent percentages
  const isPercentCol = /return|drawdown|win rate|cagr|annualized/i.test(col)

  const stringValue = String(value).trim()
  const parsedValue = Number(stringValue.replace(/%/g, ''))
  const isNumericString = stringValue !== '' && !Number.isNaN(parsedValue)

  if ((typeof value === 'number' || isNumericString) && isPercentCol) {
    const num = typeof value === 'number' ? (value as number) : parsedValue
    if (percentPosition === 'prefix') {
      return num < 0 ? `-%${Math.abs(num).toFixed(2)}` : `%${num.toFixed(2)}`
    }
    return num < 0 ? `-${Math.abs(num).toFixed(2)}%` : `${num.toFixed(2)}%`
  }

  if (typeof value === 'number') return (value as number).toFixed(2)
  return stringValue
}

function DataTableView({ table, percentPosition = 'suffix' }: { table: DataTable; percentPosition?: 'prefix' | 'suffix' }) {
  function normalizeLabel(label: string) {
    // Remove bracketed percent markers like ' [%]' or '[%]' from headers and row labels
    return label.replace(/\s*\[.*?%.*?\]/g, "").trim()
  }
  return (
    <div className="data-table">
      <table>
        <thead>
          <tr>
            <th></th>
            {table.columns.map((column) => (
              <th key={column}>{normalizeLabel(column)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.data.map((row, rowIndex) => (
            <tr key={rowIndex}>
              <td className="row-label">{normalizeLabel(table.index[rowIndex])}</td>
              {row.map((value, cellIndex) => (
                <td key={cellIndex}>{formatCell(value, table.columns[cellIndex], percentPosition)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function App() {
  const [symbols, setSymbols] = useState(DEFAULT_SYMBOLS)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ApiResult | null>(null)

  const symbolList = useMemo(() => symbols.trim(), [symbols])

  const runBacktest = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: symbolList }),
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || 'Backtest failed')
      }
      const data = (await response.json()) as ApiResult
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>Trade Wave Backtester</h1>
          <p className="subtitle">A modern TypeScript UI for local strategy evaluation.</p>
        </div>
        <button className="secondary-button" type="button" onClick={() => setSymbols(DEFAULT_SYMBOLS)}>
          Reset symbols
        </button>
      </header>

      <section className="hero-card">
        <div>
          <label htmlFor="symbol-input">Symbols</label>
          <div className="hero-input-row">
            <input
              id="symbol-input"
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              placeholder="AAPL, MSFT, GOOG"
            />
            <button className="primary-button hero-action-button" disabled={loading} onClick={runBacktest}>
              {loading ? 'Running...' : 'Run Backtest'}
            </button>
          </div>
          <p className="hint">Enter comma or space separated symbols. Leave blank for defaults.</p>
        </div>
      </section>

      {error && <div className="status-card error-card">{error}</div>}
      {result && (
        <section className="results-grid">
          <div className="card summary-card">
            <div className="card-header">
              <h2>Overall Summary</h2>
            </div>
            <DataTableView table={result.summary} percentPosition="suffix" />
          </div>

          {Object.entries(result.per_symbol).map(([strategy, strategyResult]) => (
            <div className="card strategy-card" key={strategy}>
              <div className="card-header">
                <div>
                  <h2>{strategy.replace('_', ' ')}</h2>
                  <p className="small-text">Strategy performance and trade metrics</p>
                </div>
                <span className="tag">{strategy}</span>
              </div>

              <div className="card-row">
                <div>
                  <h3>Detailed stats</h3>
                  <DataTableView table={strategyResult.stats} percentPosition="suffix" />
                </div>
              </div>
            </div>
          ))}
        </section>
      )}
    </div>
  )
}

export default App
