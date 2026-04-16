import React, { useState, useCallback, useEffect, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import { removeBackground, removeBackgroundFromUrl } from '../../api'
import {
    ProcessingTypeTabs,
    ProcessingModeDescription,
    SettingsPanel,
    ImageUploadZone,
    ErrorMessage
} from './'
import { PrimaryButton } from '../ui/PrimaryButton'
import { SecondaryButton } from '../ui/SecondaryButton'
import { Hourglass, CheckCircle2, AlertCircle, Download, Star, Link, Upload } from 'lucide-react'

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms))

const statusCopy = {
    pending: 'Queued',
    processing: 'Processing',
    success: 'Completed',
    error: 'Failed'
}

const statusBadgeClasses = {
    pending: 'bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-200',
    processing: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-200',
    success: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-200',
    error: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-200'
}

/** Minimal URL input tab — sits above (or alongside) the existing dropzone */
const UrlInputPanel = ({ onSubmit, disabled }) => {
    const [url, setUrl] = useState('')

    const handleSubmit = (e) => {
        e.preventDefault()
        const trimmed = url.trim()
        if (!trimmed) return
        onSubmit(trimmed)
        setUrl('')
    }

    return (
        <form onSubmit={handleSubmit} className="flex gap-2">
            <div className="relative flex-1">
                <Link className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                <input
                    type="url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://example.com/image.jpg"
                    disabled={disabled}
                    className="w-full pl-9 pr-3 py-2.5 text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 dark:focus:ring-purple-400 disabled:opacity-50 disabled:cursor-not-allowed"
                />
            </div>
            <PrimaryButton
                type="submit"
                disabled={disabled || !url.trim()}
                className="px-4 py-2.5 text-sm shrink-0"
            >
                Process
            </PrimaryButton>
        </form>
    )
}

/** Toggle between Upload and URL tabs */
const InputModeTabs = ({ mode, onChange }) => (
    <div className="flex gap-1 p-1 bg-gray-100 dark:bg-gray-700/50 rounded-lg w-fit mb-4">
        <button
            onClick={() => onChange('upload')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                mode === 'upload'
                    ? 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 shadow-sm'
                    : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
            }`}
        >
            <Upload className="w-3.5 h-3.5" />
            Upload
        </button>
        <button
            onClick={() => onChange('url')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                mode === 'url'
                    ? 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 shadow-sm'
                    : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
            }`}
        >
            <Link className="w-3.5 h-3.5" />
            URL
        </button>
    </div>
)

/**
 * RemoveBackground component - handles image background removal
 */
export const RemoveBackground = () => {
    const [uploads, setUploads] = useState([])
    const [processingType, setProcessingType] = useState('local')
    const [apiKey, setApiKey] = useState('')
    const [error, setError] = useState(null)
    const [queueRunning, setQueueRunning] = useState(false)
    const [starCount, setStarCount] = useState(null)
    const [inputMode, setInputMode] = useState('upload') // 'upload' | 'url'

    const uploadsRef = useRef(uploads)
    useEffect(() => {
        uploadsRef.current = uploads
    }, [uploads])

    const revokeObjectUrls = useCallback((items) => {
        items.forEach(item => {
            if (item.preview) URL.revokeObjectURL(item.preview)
            if (item.resultUrl) URL.revokeObjectURL(item.resultUrl)
        })
    }, [])

    useEffect(() => {
        return () => { revokeObjectUrls(uploadsRef.current) }
    }, [revokeObjectUrls])

    // Fetch GitHub star count
    useEffect(() => {
        const CACHE_KEY = 'withoutbg_star_count'
        const CACHE_TIMESTAMP_KEY = 'withoutbg_star_count_timestamp'
        const CACHE_DURATION = 1000 * 60 * 60

        const fetchStarCount = async () => {
            try {
                const response = await fetch('https://api.github.com/repos/withoutbg/withoutbg')
                if (response.ok) {
                    const data = await response.json()
                    const count = data.stargazers_count
                    setStarCount(count)
                    localStorage.setItem(CACHE_KEY, count.toString())
                    localStorage.setItem(CACHE_TIMESTAMP_KEY, Date.now().toString())
                }
            } catch (err) {
                // Silently fail if we can't fetch the star count
                console.error('Failed to fetch star count:', err)
            }
        }

        // Check cache first
        const cachedCount = localStorage.getItem(CACHE_KEY)
        const cachedTimestamp = localStorage.getItem(CACHE_TIMESTAMP_KEY)

        if (cachedCount && cachedTimestamp) {
            const age = Date.now() - parseInt(cachedTimestamp, 10)
            if (age < CACHE_DURATION) {
                setStarCount(parseInt(cachedCount, 10))
                return
            }
        }

        // Fetch fresh data if cache is missing or expired
        void fetchStarCount()
    }, [])

    const onDrop = useCallback((acceptedFiles) => {
        setError(null)
        const imageFiles = acceptedFiles.filter(file => file.type.startsWith('image/'))

        if (imageFiles.length === 0) {
            setError('Please upload valid image files')
            return
        }

        if (imageFiles.length > 10) {
            setError('Maximum 10 images allowed')
            return
        }

        revokeObjectUrls(uploadsRef.current)

        const timestamp = Date.now()
        const mapped = imageFiles.map((file, index) => ({
            id: `${timestamp}-${index}-${file.name}`,
            file,
            preview: URL.createObjectURL(file),
            name: file.name,
            status: 'pending',
            resultUrl: null,
            error: null,
            sourceType: 'file'
        }))

        uploadsRef.current = mapped
        setUploads(mapped)
    }, [revokeObjectUrls])

    // Handle URL submission — adds a single item to the queue
    const onUrlSubmit = useCallback((imageUrl) => {
        setError(null)

        if (processingType === 'api' && !apiKey) {
            setError('API key is required for API processing')
            return
        }

        revokeObjectUrls(uploadsRef.current)

        const id = `url-${Date.now()}-${imageUrl}`
        // Derive a display name from the URL path
        const name = imageUrl.split('/').pop().split('?')[0] || 'image'

        const item = {
            id,
            file: null,
            imageUrl,
            preview: imageUrl,   // use the URL directly as preview src
            name,
            status: 'pending',
            resultUrl: null,
            error: null,
            sourceType: 'url'
        }

        uploadsRef.current = [item]
        setUploads([item])
    }, [apiKey, processingType, revokeObjectUrls])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.webp'] },
        multiple: true
    })

    const updateUpload = useCallback((index, updater) => {
        setUploads(prev => {
            if (!prev[index]) return prev

            const updated = [...prev]
            const current = updated[index]
            const nextValue = typeof updater === 'function' ? updater(current) : { ...current, ...updater }

            if (nextValue.resultUrl && current.resultUrl && nextValue.resultUrl !== current.resultUrl) {
                URL.revokeObjectURL(current.resultUrl)
            }

            updated[index] = nextValue
            uploadsRef.current = updated
            return updated
        })
    }, [])

    const processQueue = useCallback(async () => {
        if (queueRunning) return

        if (processingType === 'api' && !apiKey) {
            setError('API key is required for API processing')
            return
        }

        setQueueRunning(true)

        try {
            while (true) {
                const nextIndex = uploadsRef.current.findIndex(item => item.status === 'pending')
                if (nextIndex === -1) break

                const current = uploadsRef.current[nextIndex]
                if (!current) break

                setError(null)
                updateUpload(nextIndex, (item) => ({ ...item, status: 'processing', error: null }))

                try {
                    let blob

                    if (current.sourceType === 'url') {
                        // Call the URL variant of the API helper
                        blob = await removeBackgroundFromUrl(current.imageUrl, {
                            format: 'png',
                            apiKey: processingType === 'api' ? apiKey : undefined
                        })
                    } else {
                        blob = await removeBackground(current.file, {
                            format: 'png',
                            apiKey: processingType === 'api' ? apiKey : undefined
                        })
                    }

                    const resultUrl = URL.createObjectURL(blob)
                    updateUpload(nextIndex, (item) => ({ ...item, status: 'success', resultUrl, error: null }))
                } catch (err) {
                    const message = err?.response?.data?.detail || err?.message || 'Failed to process image'
                    updateUpload(nextIndex, (item) => ({ ...item, status: 'error', error: message }))
                    setError(message)
                }

                if (processingType === 'api') {
                    await delay(3000)
                }
            }
        } finally {
            setQueueRunning(false)
        }
    }, [apiKey, processingType, queueRunning, updateUpload])

    useEffect(() => {
        if (uploads.length === 0) return

        const hasPending = uploads.some(item => item.status === 'pending')
        if (hasPending) {
            if (processingType === 'api' && !apiKey) {
                setError('API key is required for API processing')
                return
            }
            void processQueue()
        }
    }, [uploads, processQueue, apiKey, processingType])

    const downloadImage = useCallback((resultUrl, filename) => {
        const link = document.createElement('a')
        link.href = resultUrl
        link.download = `withoutbg-${filename}`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
    }, [])

    const downloadAll = useCallback(() => {
        uploads.forEach((item, index) => {
            if (item.status === 'success' && item.resultUrl) {
                setTimeout(() => { downloadImage(item.resultUrl, item.name) }, index * 150)
            }
        })
    }, [downloadImage, uploads])

    const reset = useCallback(() => {
        revokeObjectUrls(uploadsRef.current)
        uploadsRef.current = []
        setUploads([])
        setError(null)
        setQueueRunning(false)
    }, [revokeObjectUrls])

    const handleStarClick = useCallback(() => {
        localStorage.removeItem('withoutbg_star_count')
        localStorage.removeItem('withoutbg_star_count_timestamp')
    }, [])

    const handleProcessingTypeChange = (type) => {
        setProcessingType(type)
        if (type === 'local') setApiKey('')
        reset()
    }

    const hasResults = uploads.some(item => item.status === 'success')
    const processingCount = uploads.filter(item => item.status === 'processing').length
    const pendingCount = uploads.filter(item => item.status === 'pending').length
    const successCount = uploads.filter(item => item.status === 'success').length
    const errorCount = uploads.filter(item => item.status === 'error').length
    const allProcessed = uploads.length > 0 && processingCount === 0 && pendingCount === 0

    const summaryParts = []
    if (processingCount) summaryParts.push(`${processingCount} processing`)
    if (pendingCount) summaryParts.push(`${pendingCount} queued`)
    if (successCount) summaryParts.push(`${successCount} completed`)
    if (errorCount) summaryParts.push(`${errorCount} failed`)
    const progressSummary = summaryParts.join(' • ')

    return (
        <div className="max-w-5xl mx-auto">
            <ProcessingTypeTabs
                processingType={processingType}
                onProcessingTypeChange={handleProcessingTypeChange}
            />

            <div className="bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-b-lg rounded-tr-lg shadow-sm p-6">
                <ProcessingModeDescription processingType={processingType} />

                <SettingsPanel
                    processingType={processingType}
                    apiKey={apiKey}
                    onApiKeyChange={setApiKey}
                />

                {uploads.length === 0 && (
                    <>
                        <InputModeTabs mode={inputMode} onChange={setInputMode} />

                        {inputMode === 'upload' ? (
                            <ImageUploadZone
                                getRootProps={getRootProps}
                                getInputProps={getInputProps}
                                isDragActive={isDragActive}
                            />
                        ) : (
                            <UrlInputPanel
                                onSubmit={onUrlSubmit}
                                disabled={queueRunning}
                            />
                        )}
                    </>
                )}

                <ErrorMessage error={error} />

                {uploads.length > 0 && (
                    <div className="space-y-6 mt-6">
                        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                            <div>
                                <p className="text-sm text-gray-600 dark:text-gray-400">
                                    {progressSummary || 'Preparing images...'}
                                </p>
                                <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                                    {processingType === 'api'
                                        ? 'API mode adds a 3s pause between requests to stay within rate limits.'
                                        : 'Images are processed one by one by the local backend.'}
                                </p>
                            </div>
                            <div className="flex flex-wrap gap-3">
                                {hasResults && (
                                    <PrimaryButton
                                        onClick={downloadAll}
                                        className="px-4 py-2 flex items-center gap-2"
                                    >
                                        <Download className="w-4 h-4" />
                                        Download All
                                    </PrimaryButton>
                                )}
                                <SecondaryButton onClick={reset} className="px-4 py-2">
                                    Reset
                                </SecondaryButton>
                            </div>
                        </div>

                        <div className="space-y-6">
                            {uploads.map((item) => (
                                <div
                                    key={item.id}
                                    className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl p-4 shadow-sm transition-shadow"
                                >
                                    <div className="flex items-center justify-between mb-3">
                                        <div className="flex items-center gap-1.5 min-w-0 pr-4">
                                            {item.sourceType === 'url' && (
                                                <Link className="w-3.5 h-3.5 text-gray-400 shrink-0" />
                                            )}
                                            <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                                                {item.name}
                                            </p>
                                        </div>
                                        <span className={`text-xs font-semibold px-2.5 py-1 rounded-full shrink-0 ${statusBadgeClasses[item.status]}`}>
                                            {statusCopy[item.status]}
                                        </span>
                                    </div>

                                    <div className="space-y-4">
                                        <div className="overflow-hidden rounded-xl border border-gray-200 dark:border-gray-700">
                                            <div className="grid gap-px bg-gray-200/60 dark:bg-gray-700/60 sm:grid-cols-2">
                                                {/* Original Panel */}
                                                <div className="flex flex-col bg-white dark:bg-gray-900">
                                                    <p className="px-4 pt-4 text-xs font-semibold tracking-wide text-gray-500 dark:text-gray-400">Original</p>
                                                    <div className="flex-1 px-4 pb-4">
                                                        <div className="relative h-full min-h-[12rem] overflow-hidden rounded-lg">
                                                            <img
                                                                src={item.preview}
                                                                alt={item.name}
                                                                className={`h-full w-full object-cover ${item.status === 'pending' ? 'opacity-40 grayscale' : ''}`}
                                                            />
                                                            {item.status === 'pending' && (
                                                                <div className="absolute inset-0 flex flex-col items-center justify-center bg-purple-950/40 dark:bg-purple-900/40 backdrop-blur-sm">
                                                                    <Hourglass className="w-16 h-16 text-purple-300" />
                                                                    <span className="mt-2 text-sm font-semibold text-purple-100">In queue</span>
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>

                                                {/* Result Panel */}
                                                <div className="relative flex flex-col bg-white dark:bg-gray-900">
                                                    <p className="px-4 pt-4 text-xs font-semibold tracking-wide text-gray-500 dark:text-gray-400">Result</p>
                                                    <div className="flex-1 px-4 pb-4">
                                                        <div className="checkerboard relative flex h-full min-h-[12rem] w-full items-center justify-center overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
                                                            {item.status === 'processing' && (
                                                                <>
                                                                    <div className="absolute inset-0 bg-gray-900/30 dark:bg-gray-950/50" />
                                                                    <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/70 to-transparent dark:via-white/20 animate-shimmer" />
                                                                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white">
                                                                        <div className="text-sm font-semibold uppercase tracking-wide">Processing</div>
                                                                        <div className="mt-1 text-xs text-gray-100/80">Hang tight ✨</div>
                                                                    </div>
                                                                </>
                                                            )}
                                                            {item.status === 'success' && (
                                                                <>
                                                                    <img
                                                                        src={item.resultUrl}
                                                                        alt={`${item.name} without background`}
                                                                        className="h-full w-full object-contain"
                                                                    />
                                                                    <div className="absolute top-2 right-2">
                                                                        <CheckCircle2 className="h-6 w-6 text-emerald-400 drop-shadow-sm" />
                                                                    </div>
                                                                </>
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        {item.status === 'success' && (
                                            <div className="flex justify-center">
                                                <PrimaryButton
                                                    onClick={() => downloadImage(item.resultUrl, item.name)}
                                                    className="px-6 py-2 flex items-center gap-2"
                                                >
                                                    <Download className="w-4 h-4" />
                                                    Download
                                                </PrimaryButton>
                                            </div>
                                        )}

                                        {item.status === 'error' && (
                                            <div className="flex items-start gap-2 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-3 py-2">
                                                <AlertCircle className="h-5 w-5 mt-0.5 text-red-500 dark:text-red-300" />
                                                <div>
                                                    <p className="text-sm font-medium text-red-700 dark:text-red-200">Processing failed</p>
                                                    <p className="text-xs text-red-600 dark:text-red-300/80">{item.error}</p>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>

                        {allProcessed && successCount > 0 && (
                            <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 border-2 border-purple-200 dark:border-purple-700/50 rounded-xl p-6 shadow-sm mt-6">
                                <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                                    <div className="text-center sm:text-left">
                                        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-1">
                                            Love the results?
                                        </h3>
                                        <p className="text-sm text-gray-600 dark:text-gray-400">
                                            Star withoutbg on GitHub to support the project!
                                        </p>
                                    </div>
                                    <a
                                        href="https://github.com/withoutbg/withoutbg"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        onClick={handleStarClick}
                                        className="inline-flex items-center gap-2 px-6 py-3 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 rounded-lg font-medium hover:bg-gray-800 dark:hover:bg-gray-200 shadow-md hover:shadow-lg transform hover:scale-105 transition-all duration-200"
                                    >
                                        <Star className="w-5 h-5" />
                                        <span>Star on GitHub{starCount !== null && ` (${starCount.toLocaleString()})`}</span>
                                    </a>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    )
}
