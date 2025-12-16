'use client'

import * as React from 'react'
import { Check, ChevronsUpDown, Search } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

interface ComboboxOption {
    value: string
    label: string
}

interface ComboboxProps {
    options: ComboboxOption[]
    value?: string
    onChange?: (value: string) => void
    placeholder?: string
    searchPlaceholder?: string
    disabled?: boolean
    className?: string
}

export function Combobox({
    options,
    value,
    onChange,
    placeholder = 'Select an option...',
    searchPlaceholder = 'Search...',
    disabled = false,
    className
}: ComboboxProps) {
    const [open, setOpen] = React.useState(false)
    const [search, setSearch] = React.useState('')
    const dropdownRef = React.useRef<HTMLDivElement>(null)
    const buttonRef = React.useRef<HTMLButtonElement>(null)

    // Close dropdown when clicking outside
    React.useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (
                dropdownRef.current &&
                !dropdownRef.current.contains(event.target as Node) &&
                buttonRef.current &&
                !buttonRef.current.contains(event.target as Node)
            ) {
                setOpen(false)
            }
        }

        if (open) {
            document.addEventListener('mousedown', handleClickOutside)
            return () => document.removeEventListener('mousedown', handleClickOutside)
        }
    }, [open])

    // Filter options based on search
    const filteredOptions = React.useMemo(() => {
        if (!search) return options
        const searchLower = search.toLowerCase()
        return options.filter((option) => option.label.toLowerCase().includes(searchLower) || option.value.toLowerCase().includes(searchLower))
    }, [options, search])

    // Get selected option label
    const selectedOption = options.find((option) => option.value === value)

    const handleSelect = (optionValue: string) => {
        onChange?.(optionValue)
        setOpen(false)
        setSearch('')
    }

    return (
        <div className={cn('relative', className)}>
            <Button
                ref={buttonRef}
                type="button"
                variant="outline"
                role="combobox"
                aria-expanded={open}
                disabled={disabled}
                className="w-full justify-between"
                onClick={() => !disabled && setOpen(!open)}
            >
                <span className={cn('truncate', !selectedOption && 'text-muted-foreground')}>{selectedOption ? selectedOption.label : placeholder}</span>
                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
            </Button>

            {open && (
                <div
                    ref={dropdownRef}
                    className="border-input bg-popover text-popover-foreground absolute top-full z-50 mt-1 w-full rounded-md border shadow-md"
                >
                    <div className="flex items-center border-b px-3 py-2">
                        <Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
                        <Input
                            placeholder={searchPlaceholder}
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            className="h-8 border-0 p-0 shadow-none focus-visible:ring-0"
                        />
                    </div>
                    <div className="max-h-[300px] overflow-y-auto p-1">
                        {filteredOptions.length === 0 ? (
                            <div className="text-muted-foreground py-6 text-center text-sm">No results found.</div>
                        ) : (
                            filteredOptions.map((option) => (
                                <button
                                    key={option.value}
                                    type="button"
                                    onClick={() => handleSelect(option.value)}
                                    className={cn(
                                        'hover:bg-accent hover:text-accent-foreground relative flex w-full cursor-pointer select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none transition-colors',
                                        value === option.value && 'bg-accent'
                                    )}
                                >
                                    <Check className={cn('mr-2 h-4 w-4', value === option.value ? 'opacity-100' : 'opacity-0')} />
                                    <span className="truncate">{option.label}</span>
                                </button>
                            ))
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}
