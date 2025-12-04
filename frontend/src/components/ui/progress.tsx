'use client'

import * as React from 'react'

export interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
    value?: number
    max?: number
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(({ value = 0, max = 100, ...props }, ref) => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100)

    return (
        <div ref={ref} className="bg-secondary relative h-4 w-full overflow-hidden rounded-full" {...props}>
            <div
                className="bg-primary h-full w-full flex-1 transition-all duration-300 ease-in-out"
                style={{ transform: `translateX(-${100 - percentage}%)` }}
            />
        </div>
    )
})
Progress.displayName = 'Progress'

export { Progress }
