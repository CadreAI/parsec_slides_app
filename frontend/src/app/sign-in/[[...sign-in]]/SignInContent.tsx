'use client'

import { SignIn, SignedOut } from '@clerk/nextjs'

export default function SignInContent() {
    return (
        <div className="flex min-h-screen flex-col items-center justify-center p-8">
            <SignedOut>
                <div className="mb-8 flex flex-col items-center justify-center gap-4 text-center">
                    <h1 className="text-6xl font-bold">Parsec Education</h1>
                    <h2 className="text-3xl font-bold">Deck Creation Application</h2>
                    <p className="text-muted-foreground text-sm">Sign in to continue</p>
                </div>
                <SignIn
                    routing="path"
                    path="/sign-in"
                    signUpUrl="/sign-up"
                    appearance={{
                        elements: {
                            card: 'shadow-none border-0 bg-transparent',
                            formButtonPrimary: 'bg-primary text-primary-foreground hover:bg-primary/90',
                            footer: 'hidden'
                        }
                    }}
                />
            </SignedOut>
        </div>
    )
}
