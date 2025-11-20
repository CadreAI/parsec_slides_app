'use client'

import * as React from 'react'
import { cn } from '@/lib/utils'

interface ToastProps {
    message: string
    type?: 'default' | 'success' | 'error'
    duration?: number
    onClose: () => void
}

export function Toast({ message, type = 'default', duration = 3000, onClose }: ToastProps) {
    React.useEffect(() => {
        const timer = setTimeout(() => {
            onClose()
        }, duration)

        return () => clearTimeout(timer)
    }, [duration, onClose])

    return (
        <div
            className={cn(
                'fixed bottom-4 right-4 z-50 flex items-center gap-3 rounded-lg border bg-background px-4 py-3 shadow-lg transition-all',
                type === 'success' && 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950',
                type === 'error' && 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950'
            )}
        >
            <div className="flex-1">
                <p className={cn('text-sm font-medium', type === 'success' && 'text-green-900 dark:text-green-100', type === 'error' && 'text-red-900 dark:text-red-100')}>
                    {message}
                </p>
            </div>
            <button
                onClick={onClose}
                className={cn(
                    'rounded-md p-1 hover:bg-background/50',
                    type === 'success' && 'hover:bg-green-100 dark:hover:bg-green-900',
                    type === 'error' && 'hover:bg-red-100 dark:hover:bg-red-900'
                )}
            >
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>
    )
}

interface ToastContextType {
    showToast: (message: string, type?: 'default' | 'success' | 'error') => void
}

const ToastContext = React.createContext<ToastContextType | undefined>(undefined)

export function ToastProvider({ children }: { children: React.ReactNode }) {
    const [toast, setToast] = React.useState<{ message: string; type: 'default' | 'success' | 'error' } | null>(null)

    const showToast = React.useCallback((message: string, type: 'default' | 'success' | 'error' = 'default') => {
        setToast({ message, type })
    }, [])

    return (
        <ToastContext.Provider value={{ showToast }}>
            {children}
            {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
        </ToastContext.Provider>
    )
}

export function useToast() {
    const context = React.useContext(ToastContext)
    if (!context) {
        throw new Error('useToast must be used within ToastProvider')
    }
    return context
}

