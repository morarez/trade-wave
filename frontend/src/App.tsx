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

function formatCell(
  value: string | number | null,
  metricLabel?: string,
  columnLabel?: string,
  percentPosition: 'prefix' | 'suffix' = 'suffix'
) {
  if (value === null || value === undefined) return '-'
  const metric = (metricLabel || '').toLowerCase()
  const column = (columnLabel || '').toLowerCase()

  // Metrics or columns that represent percentages
  const isPercentMetric = /return|drawdown|win rate|cagr|annualized/i.test(metric) || /return|drawdown|win rate|cagr|annualized/i.test(column)

  const stringValue = String(value).trim()
  const parsedValue = Number(stringValue.replace(/%/g, ''))
  const isNumericString = stringValue !== '' && !Number.isNaN(parsedValue)

  if ((typeof value === 'number' || isNumericString) && isPercentMetric) {
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
                <td key={cellIndex}>{formatCell(value, table.index[rowIndex], table.columns[cellIndex], percentPosition)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

type Tab = 'backtest' | 'train'

function App() {
  const [tab, setTab] = useState<Tab>('backtest')
  const [symbols, setSymbols] = useState(DEFAULT_SYMBOLS)
  const [strategies, setStrategies] = useState('')
  const [startDate, setStartDate] = useState('2025-01-01')
  const [endDate, setEndDate] = useState('')
  const [cash, setCash] = useState('10000')
  const [modelPath, setModelPath] = useState('ai/models/pipeline_model.pkl')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ApiResult | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  // Training state
  const [trainMode, setTrainMode] = useState('final')
  const [trainSymbols, setTrainSymbols] = useState('AAPL,MSFT,GOOG')
  const [trainStart, setTrainStart] = useState('2015-01-01')
  const [trainModelPath, setTrainModelPath] = useState('ai/models/pipeline_model.pkl')
  const [trainSize, setTrainSize] = useState('500')
  const [testSize, setTestSize] = useState('100')
  const [stepSize, setStepSize] = useState('50')

  const symbolList = useMemo(() => symbols.trim(), [symbols])

  const runBacktest = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    setMessage(null)

    try {
      const body: any = { symbols: symbolList }
      if (strategies.trim()) body.strategies = strategies
      if (startDate) body.start = startDate
      if (endDate) body.end = endDate
      if (cash) body.cash = parseFloat(cash)
      if (modelPath.trim()) body.models = [modelPath]

      const response = await fetch('/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
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

  const runTrain = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    setMessage(null)

    try {
      const body: any = {
        mode: trainMode,
        symbols: trainSymbols,
        start: trainStart,
        model_path: trainModelPath,
      }
      if (trainMode === 'walk_forward') {
        body.train_size = parseInt(trainSize)
        body.test_size = parseInt(testSize)
        body.step_size = parseInt(stepSize)
      }

      const response = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || 'Training failed')
      }
      const data = await response.json()
      setMessage(data.message || 'Training completed successfully')
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
          <h1>Trade Wave</h1>
          <p className="subtitle">AI-driven backtesting and model training platform</p>
        </div>
        <button className="secondary-button" type="button" onClick={() => setSymbols(DEFAULT_SYMBOLS)}>
          Reset
        </button>
      </header>

      <nav className="tab-navigation">
        <button className={`tab-button ${tab === 'backtest' ? 'active' : ''}`} onClick={() => setTab('backtest')}>
          Backtest
        </button>
        <button className={`tab-button ${tab === 'train' ? 'active' : ''}`} onClick={() => setTab('train')}>
          Train Model
        </button>
      </nav>

      {tab === 'backtest' && (
        <section className="tab-content">
          <section className="hero-card">
            <div>
              <label htmlFor="symbol-input">Symbols</label>
              <div className="form-group">
                <input
                  id="symbol-input"
                  value={symbols}
                  onChange={(e) => setSymbols(e.target.value)}
                  placeholder="AAPL, MSFT, GOOG"
                />
                <p className="hint">Comma or space separated stock symbols</p>
              </div>

              <label htmlFor="strategy-input">Strategies (optional)</label>
              <div className="form-group">
                <input
                  id="strategy-input"
                  value={strategies}
                  onChange={(e) => setStrategies(e.target.value)}
                  placeholder="sma_rsi, bollinger_rsi, macd_trend"
                />
                <p className="hint">Leave blank to run all available strategies</p>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="start-date">Start Date</label>
                  <input
                    id="start-date"
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="end-date">End Date (optional)</label>
                  <input
                    id="end-date"
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="cash">Starting Cash</label>
                  <input
                    id="cash"
                    type="number"
                    value={cash}
                    onChange={(e) => setCash(e.target.value)}
                    placeholder="10000"
                  />
                </div>
              </div>

              <label htmlFor="model-input">AI Model Path (optional)</label>
              <div className="form-group">
                <input
                  id="model-input"
                  value={modelPath}
                  onChange={(e) => setModelPath(e.target.value)}
                  placeholder="ai/models/pipeline_model.pkl"
                />
                <p className="hint">Path to trained AI model for comparison</p>
              </div>

              <button className="primary-button" disabled={loading} onClick={runBacktest}>
                {loading ? 'Running Backtest...' : 'Run Backtest'}
              </button>
            </div>
          </section>

          {error && <div className="status-card error-card">{error}</div>}
          {message && <div className="status-card success-card">{message}</div>}
          {result && (
            <section className="results-grid">
              <div className="card summary-card">
                <div className="card-header">
                  <h2>Overall Summary</h2>
                </div>
                <DataTableView table={result.summary} percentPosition="suffix" />
              </div>
              {Object.entries(result.per_symbol).map(([symbolName, symbolData]) => (
                <div key={symbolName} className="card">
                  <div className="card-header">
                    <h3>{symbolName}</h3>
                  </div>
                  <div className="card-header" style={{ marginTop: '16px' }}>
                    <h4>Detailed Stats</h4>
                  </div>
                  <DataTableView table={symbolData.stats} percentPosition="suffix" />
                </div>
              ))}
            </section>
          )}
        </section>
      )}

      {tab === 'train' && (
        <section className="tab-content">
          <section className="hero-card">
            <div>
              <label htmlFor="train-mode">Training Mode</label>
              <div className="form-group">
                <select id="train-mode" value={trainMode} onChange={(e) => setTrainMode(e.target.value)}>
                  <option value="final">Final Split (train on full data)</option>
                  <option value="walk_forward">Walk-Forward Validation</option>
                </select>
              </div>

              <label htmlFor="train-symbols">Symbols</label>
              <div className="form-group">
                <input
                  id="train-symbols"
                  value={trainSymbols}
                  onChange={(e) => setTrainSymbols(e.target.value)}
                  placeholder="AAPL,MSFT,GOOG"
                />
                <p className="hint">Comma-separated stock symbols for training</p>
              </div>

              <label htmlFor="train-start">Training Start Date</label>
              <div className="form-group">
                <input
                  id="train-start"
                  type="date"
                  value={trainStart}
                  onChange={(e) => setTrainStart(e.target.value)}
                />
              </div>

              {trainMode === 'walk_forward' && (
                <div className="form-row">
                  <div className="form-group">
                    <label htmlFor="train-size">Training Window Size</label>
                    <input
                      id="train-size"
                      type="number"
                      value={trainSize}
                      onChange={(e) => setTrainSize(e.target.value)}
                      placeholder="500"
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor="test-size">Test Window Size</label>
                    <input
                      id="test-size"
                      type="number"
                      value={testSize}
                      onChange={(e) => setTestSize(e.target.value)}
                      placeholder="100"
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor="step-size">Step Size</label>
                    <input
                      id="step-size"
                      type="number"
                      value={stepSize}
                      onChange={(e) => setStepSize(e.target.value)}
                      placeholder="50"
                    />
                  </div>
                </div>
              )}

              <label htmlFor="train-model-path">Model Save Path</label>
              <div className="form-group">
                <input
                  id="train-model-path"
                  value={trainModelPath}
                  onChange={(e) => setTrainModelPath(e.target.value)}
                  placeholder="ai/models/pipeline_model.pkl"
                />
              </div>

              <button className="primary-button" disabled={loading} onClick={runTrain}>
                {loading ? 'Training...' : 'Start Training'}
              </button>
            </div>
          </section>

          {error && <div className="status-card error-card">{error}</div>}
          {message && <div className="status-card success-card">{message}</div>}
        </section>
      )}
    </div>
  )
}

export default App
