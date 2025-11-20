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

    return (
        <div ref={containerRef} className={cn('relative', className)}>
            <button
                type="button"
                onClick={() => !disabled && setIsOpen(!isOpen)}
                disabled={disabled}
                className={cn(
                    'flex h-9 w-full items-center justify-between rounded-md border border-input bg-transparent px-3 py-1 text-base shadow-xs transition-[color,box-shadow] focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm dark:border-input dark:bg-input/30',
                    selected.length === 0 && 'text-muted-foreground'
                )}
            >
                <span className="truncate">{selected.length === 0 ? placeholder : selected.length === 1 ? selected[0] : `${selected.length} selected`}</span>
                <svg className={cn('h-4 w-4 transition-transform', isOpen && 'rotate-180')} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
            </button>

            {isOpen && (
                <div className="absolute z-50 mt-1 w-full rounded-md border bg-background shadow-md">
                    <div className="max-h-60 overflow-auto p-2">
                        {options.map((option) => (
                            <div key={option} className="flex items-center space-x-2 rounded-sm px-2 py-1.5 hover:bg-accent">
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
                        <span key={item} className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-1 text-xs font-medium text-primary">
                            {item}
                            <button type="button" onClick={() => handleToggle(item)} className="rounded-full hover:bg-primary/20">
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
