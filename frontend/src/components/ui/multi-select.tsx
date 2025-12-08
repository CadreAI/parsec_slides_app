'use client'

import * as React from 'react'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { cn } from '@/lib/utils'

interface MultiSelectProps {
    options: string[]
    selected: string[]
    onChange: (selected: string[]) => void
    placeholder?: string
    className?: string
    disabled?: boolean
}

export function MultiSelect({ options, selected, onChange, placeholder = 'Select options...', className, disabled = false }: MultiSelectProps) {
    const [isOpen, setIsOpen] = React.useState(false)
    const containerRef = React.useRef<HTMLDivElement>(null)

    React.useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setIsOpen(false)
            }
        }

        document.addEventListener('mousedown', handleClickOutside)
        return () => {
            document.removeEventListener('mousedown', handleClickOutside)
        }
    }, [])

    const handleToggle = (option: string) => {
        if (selected.includes(option)) {
            onChange(selected.filter((item) => item !== option))
        } else {
            onChange([...selected, option])
        }
    }

    const handleSelectAll = () => {
        if (selected.length === options.length) {
            // Deselect all
            onChange([])
        } else {
            // Select all
            onChange([...options])
        }
    }

    const allSelected = options.length > 0 && selected.length === options.length

    return (
        <div ref={containerRef} className={cn('relative', className)}>
            <button
                type="button"
                onClick={() => !disabled && setIsOpen(!isOpen)}
                disabled={disabled}
                className={cn(
                    'border-input shadow-xs focus-visible:border-ring focus-visible:ring-ring/50 dark:border-input dark:bg-input/30 flex h-9 w-full items-center justify-between rounded-md border bg-transparent px-3 py-1 text-base transition-[color,box-shadow] focus-visible:outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50 md:text-sm',
                    selected.length === 0 && 'text-muted-foreground'
                )}
            >
                <span className="truncate">{selected.length === 0 ? placeholder : selected.length === 1 ? selected[0] : `${selected.length} selected`}</span>
                <svg className={cn('h-4 w-4 transition-transform', isOpen && 'rotate-180')} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
            </button>

            {isOpen && (
                <div className="bg-background absolute z-50 mt-1 w-full rounded-md border shadow-md">
                    {options.length > 0 && (
                        <div className="border-b p-2">
                            <button
                                type="button"
                                onClick={handleSelectAll}
                                className="hover:bg-accent flex w-full items-center space-x-2 rounded-sm px-2 py-1.5 text-sm font-medium"
                            >
                                <Checkbox checked={allSelected} onChange={handleSelectAll} />
                                <Label className="cursor-pointer font-medium">{allSelected ? 'Deselect All' : 'Select All'}</Label>
                            </button>
                        </div>
                    )}
                    <div className="max-h-60 overflow-auto p-2">
                        {options.map((option) => (
                            <div key={option} className="hover:bg-accent flex items-center space-x-2 rounded-sm px-2 py-1.5">
                                <Checkbox id={`multi-select-${option}`} checked={selected.includes(option)} onChange={() => handleToggle(option)} />
                                <Label htmlFor={`multi-select-${option}`} className="flex-1 cursor-pointer font-normal">
                                    {option}
                                </Label>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {selected.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-2">
                    {selected.map((item) => (
                        <span key={item} className="bg-primary/10 text-primary inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs font-medium">
                            {item}
                            <button type="button" onClick={() => handleToggle(item)} className="hover:bg-primary/20 rounded-full">
                                <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </span>
                    ))}
                </div>
            )}
        </div>
    )
}
