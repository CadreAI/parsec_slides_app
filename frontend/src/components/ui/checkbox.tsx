import * as React from 'react'

import { cn } from '@/lib/utils'

const Checkbox = React.forwardRef<HTMLInputElement, React.ComponentProps<'input'>>(({ className, type = 'checkbox', ...props }, ref) => {
    return (
        <input
            type={type}
            className={cn(
                'border-input shadow-xs focus-visible:border-ring focus-visible:ring-ring/50 dark:bg-input/30 h-4 w-4 rounded outline-none transition-[color,box-shadow] focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50',
                className
            )}
            ref={ref}
            {...props}
        />
    )
})
Checkbox.displayName = 'Checkbox'

export { Checkbox }
