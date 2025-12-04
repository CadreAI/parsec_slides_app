import * as React from 'react'

import { cn } from '@/lib/utils'

const Select = React.forwardRef<HTMLSelectElement, React.ComponentProps<'select'>>(({ className, children, ...props }, ref) => {
    return (
        <select
            className={cn(
                'border-input shadow-xs focus-visible:border-ring focus-visible:ring-ring/50 dark:border-input dark:bg-input/30 flex h-9 w-full rounded-md border bg-transparent px-3 py-1 text-base transition-[color,box-shadow] focus-visible:outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50 md:text-sm',
                className
            )}
            ref={ref}
            {...props}
        >
            {children}
        </select>
    )
})
Select.displayName = 'Select'

export { Select }
