import { useRef, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
import './App.css'

type MaintenanceAction = {
  status: string
  severity?: string
  component?: string
  possible_causes?: string[]
  actions?: string[]
}

type Diagnostic = {
  recording_id: string
  window: number
  samples: number
  fault: number
  fault_type: string
  confidence?: {
    fault: number
    fault_type: number
  }
  maintenance_action: MaintenanceAction
}

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [diagnostics, setDiagnostics] = useState<Diagnostic[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedDiagnostic, setSelectedDiagnostic] = useState<Diagnostic | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const selectFile = (candidate: File | undefined) => {
    setError('')
    setDiagnostics([])
    setSelectedDiagnostic(null)
    if (!candidate) return
    if (!candidate.name.toLowerCase().endsWith('.csv')) {
      setFile(null)
      setError('Please choose a CSV file.')
      return
    }
    setFile(candidate)
  }

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    selectFile(event.target.files?.[0])
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
    selectFile(event.dataTransfer.files[0])
  }

  const runDiagnostics = async () => {
    if (!file) return
    setIsLoading(true)
    setError('')
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await fetch(`${API_URL}/diagnostics`, {
        method: 'POST',
        body: formData,
      })
      const payload = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(payload?.detail ?? 'Unable to process this CSV file.')
      }
      if (!Array.isArray(payload)) {
        throw new Error('The diagnostic service returned an invalid response.')
      }
      const results = payload as Diagnostic[]
      setDiagnostics(results)
      setSelectedDiagnostic(results.length === 1 ? results[0] : null)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Unable to connect to the API.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="page">
      <section className="card">
        <header>
          <h1>Upload vibration data</h1>
          <p className="intro">
            Upload a CSV recording to detect bearing faults and receive maintenance guidance.
          </p>
        </header>

        <div
          className={`dropzone ${isDragging ? 'dragging' : ''}`}
          onDragOver={(event) => {
            event.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click()
          }}
        >
          <input ref={inputRef} type="file" accept=".csv,text/csv" onChange={handleFileChange} hidden />
          <span className="upload-symbol">↑</span>
          <strong>{file ? file.name : 'Choose a CSV file'}</strong>
          <span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB selected` : 'Drop it here or click to browse'}</span>
        </div>

        <p className="hint">Required columns: DE_Time, FE_Time, RecordingID · Minimum 2,048 rows</p>

        {error && <p className="error">{error}</p>}

        <button className="button" type="button" disabled={!file || isLoading} onClick={runDiagnostics}>
          {isLoading ? 'Analyzing...' : 'Run diagnosis'}
        </button>

        {diagnostics.length > 0 && (
          <section className="results">
            <div className="results-header">
              <div>
                <p className="label">RESULTS</p>
                <h2>{diagnostics.length} diagnostic result{diagnostics.length === 1 ? '' : 's'}</h2>
              </div>
              <span className="result-count">{diagnostics.filter((item) => item.fault !== 0).length} fault(s)</span>
            </div>
            <div className="diagnostic-grid">
              {diagnostics.map((diagnostic, index) => {
                const action = diagnostic.maintenance_action
                const hasFault = diagnostic.fault !== 0
                return (
                  <button
                    className={`diagnostic-tile ${hasFault ? 'fault' : 'healthy'} ${selectedDiagnostic === diagnostic ? 'selected' : ''}`}
                    key={`${diagnostic.recording_id}-${diagnostic.window}-${index}`}
                    type="button"
                    onClick={() => setSelectedDiagnostic(diagnostic)}
                  >
                    <span className="tile-window">SAMPLES {diagnostic.samples}</span>
                    <strong>{diagnostic.recording_id}</strong>
                    <span className="tile-fault">{diagnostic.fault_type}</span>
                    <span className="status">{action?.status ?? (hasFault ? 'FAULT DETECTED' : 'HEALTHY')}</span>
                  </button>
                )
              })}
            </div>
            {selectedDiagnostic && (
              <article className={`diagnostic ${selectedDiagnostic.fault !== 0 ? 'fault' : 'healthy'}`}>
                <div className="diagnostic-top">
                  <div>
                    <strong>{selectedDiagnostic.recording_id}</strong>
                    <span>Samples {selectedDiagnostic.samples}</span>
                  </div>
                  <button className="close-details" type="button" onClick={() => setSelectedDiagnostic(null)} aria-label="Close diagnostic details">×</button>
                </div>
                <h3>{selectedDiagnostic.fault_type}</h3>
                {selectedDiagnostic.confidence && (
                  <p className="confidence">
                    Model confidence: <strong>Fault {Math.round(selectedDiagnostic.confidence.fault * 100)}%</strong>
                    {' · '}
                    <strong>Type {Math.round(selectedDiagnostic.confidence.fault_type * 100)}%</strong>
                  </p>
                )}
                {selectedDiagnostic.maintenance_action && (
                  <>
                    {(selectedDiagnostic.maintenance_action.severity || selectedDiagnostic.maintenance_action.component) && (
                      <p className="severity">
                        {selectedDiagnostic.maintenance_action.severity && (
                          <>Severity: <strong>{selectedDiagnostic.maintenance_action.severity}</strong></>
                        )}
                        {selectedDiagnostic.maintenance_action.severity && selectedDiagnostic.maintenance_action.component && ' · '}
                        {selectedDiagnostic.maintenance_action.component && (
                          <>Component: <strong>{selectedDiagnostic.maintenance_action.component}</strong></>
                        )}
                      </p>
                    )}
                    {(selectedDiagnostic.maintenance_action.possible_causes?.length ||
                      selectedDiagnostic.maintenance_action.actions?.length) && (
                      <div className="details">
                        {selectedDiagnostic.maintenance_action.possible_causes?.length ? (
                          <div>
                            <h4>Possible causes</h4>
                            <ul>
                              {selectedDiagnostic.maintenance_action.possible_causes.map((cause) => (
                                <li key={cause}>{cause}</li>
                              ))}
                            </ul>
                          </div>
                        ) : null}
                        {selectedDiagnostic.maintenance_action.actions?.length ? (
                          <div>
                            <h4>Recommended actions</h4>
                            <ol>
                              {selectedDiagnostic.maintenance_action.actions.map((item) => (
                                <li key={item}>{item}</li>
                              ))}
                            </ol>
                          </div>
                        ) : null}
                      </div>
                    )}
                  </>
                )}
              </article>
            )}
          </section>
        )}
      </section>
    </main>
  )
}

export default App
