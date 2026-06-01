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

function formatCell(value: string | number | null) {
  if (value === null || value === undefined) return '-'
  if (typeof value === 'number') return value.toFixed(2)
  return String(value)
}

function DataTableView({ table }: { table: DataTable }) {
  return (
    <div className="data-table">
      <table>
        <thead>
          <tr>
            <th></th>
            {table.columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.data.map((row, rowIndex) => (
            <tr key={rowIndex}>
              <td className="row-label">{table.index[rowIndex]}</td>
              {row.map((value, cellIndex) => (
                <td key={cellIndex}>{formatCell(value)}</td>
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
            <DataTableView table={result.summary} />
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
                  <h3>Summary</h3>
                  <DataTableView table={strategyResult.summary} />
                </div>
                <div>
                  <h3>Detailed stats</h3>
                  <DataTableView table={strategyResult.stats} />
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
